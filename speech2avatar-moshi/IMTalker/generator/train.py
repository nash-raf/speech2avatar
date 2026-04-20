import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.nn import Module
from torch.utils import data

from generator.FM import FMGenerator
from generator.dataset import AudioMotionSmirkGazeDataset
from generator.options.base_options import BaseOptions


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow = self.shadow.get(name)
                if shadow is None:
                    self.shadow[name] = param.data.clone().detach()
                    continue
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow.to(param.device)
                self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def append_dims(t: torch.Tensor, ndims: int) -> torch.Tensor:
    return t.reshape(*t.shape, *((1,) * ndims))


class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)


class L1loss(Module):
    def forward(self, pred, target, **kwargs):
        return F.l1_loss(pred, target)


def peak_weighted_l1(pred, target, power=1.0, eps=1e-6):
    with torch.no_grad():
        gt_vel = (target[:, 1:] - target[:, :-1]).abs().mean(dim=-1)
        w = gt_vel.pow(power)
        w = w / (w.mean(dim=1, keepdim=True) + eps)
        w = torch.cat([w[:, :1], w], dim=1)
    per_frame = (pred - target).abs().mean(dim=-1)
    return (w * per_frame).mean()


class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = FMGenerator(opt)
        self.opt = opt
        self.loss_fn = L1loss()
        self.ema = EMA(self.model, decay=getattr(opt, "ema_decay", 0.9999))

    def forward(self, batch, t):
        return self.model(batch, t=t)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step >= getattr(self.opt, "ema_start_step", 0):
            self.ema.update()

    def on_validation_epoch_start(self):
        self.ema.apply_shadow()

    def on_validation_epoch_end(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint and isinstance(checkpoint["ema_state_dict"], dict):
            self.ema.shadow = checkpoint["ema_state_dict"]

    def _prepare_flow_batch(self, batch):
        batch = dict(batch)
        m_now = batch["m_now"]
        noise = torch.randn_like(m_now)
        times = torch.rand(m_now.size(0), device=self.device)
        t = append_dims(times, m_now.ndim - 1)
        batch["m_now"] = t * m_now + (1 - t) * noise
        gt_flow = m_now - noise
        return batch, times, gt_flow

    def training_step(self, batch, batch_idx):
        batch, times, gt_flow = self._prepare_flow_batch(batch)
        pred_flow = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow, gt_flow)
        vel_loss = self.loss_fn(
            pred_flow[:, 1:] - pred_flow[:, :-1],
            gt_flow[:, 1:] - gt_flow[:, :-1],
        )

        if getattr(self.opt, "audio_adapter_mode", "none") == "temporal_conv_to_32":
            acc_loss = self.loss_fn(
                pred_flow[:, 2:] - 2 * pred_flow[:, 1:-1] + pred_flow[:, :-2],
                gt_flow[:, 2:] - 2 * gt_flow[:, 1:-1] + gt_flow[:, :-2],
            )
            peak_loss = peak_weighted_l1(
                pred_flow,
                gt_flow,
                power=self.opt.peak_power,
            )
            train_loss = (
                fm_loss
                + self.opt.lambda_vel * vel_loss
                + self.opt.lambda_acc * acc_loss
                + self.opt.lambda_peak * peak_loss
            )
            self.log("acc_loss", acc_loss, prog_bar=True, sync_dist=True)
            self.log("peak_loss", peak_loss, prog_bar=True, sync_dist=True)
        else:
            train_loss = fm_loss + vel_loss

        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)
        self.log("fm_loss", fm_loss, prog_bar=True, sync_dist=True)
        self.log("vel_loss", vel_loss, prog_bar=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        batch, times, gt_flow = self._prepare_flow_batch(batch)
        pred_flow = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow, gt_flow)
        vel_loss = self.loss_fn(
            pred_flow[:, 1:] - pred_flow[:, :-1],
            gt_flow[:, 1:] - gt_flow[:, :-1],
        )
        val_loss = fm_loss + vel_loss

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_fm_loss", fm_loss, prog_bar=True, sync_dist=True)
        self.log("val_vel_loss", vel_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def _normalize_candidate_state(self, raw_state):
        if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
            raw_state = raw_state["model"]
        if not isinstance(raw_state, dict):
            raise RuntimeError(f"Unsupported checkpoint structure: {type(raw_state)!r}")
        candidates = [raw_state]
        for prefix in ("model.", "student.", "teacher."):
            stripped = {k[len(prefix) :]: v for k, v in raw_state.items() if k.startswith(prefix)}
            if stripped:
                candidates.append(stripped)
        return candidates

    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        raw_state = ckpt.get("state_dict", ckpt)
        if "ema_state_dict" in ckpt and isinstance(ckpt["ema_state_dict"], dict):
            print("[INFO] Found EMA weights in checkpoint. Loading EMA weights for stability.")
            raw_state = ckpt["ema_state_dict"]
        else:
            print("[INFO] EMA weights not found. Loading standard state_dict.")

        model_state_dict = self.model.state_dict()
        best_state = None
        best_match_count = -1
        for candidate in self._normalize_candidate_state(raw_state):
            match_count = sum(
                1
                for k, v in candidate.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            )
            if match_count > best_match_count:
                best_match_count = match_count
                best_state = candidate

        if best_state is None or best_match_count <= 0:
            raise RuntimeError(f"Could not find compatible weights in checkpoint: {ckpt_path}")

        loadable_params = {}
        unmatched_keys = []
        for k, v in best_state.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                loadable_params[k] = v
            else:
                unmatched_keys.append(k)

        missing_keys, unexpected_keys = self.model.load_state_dict(loadable_params, strict=False)
        loaded_prefixes = sorted({k.split(".")[0] for k in loadable_params.keys()})
        fresh_prefixes = sorted({k.split(".")[0] for k in model_state_dict.keys()} - set(loaded_prefixes))
        print(f"[INFO] Loaded {len(loadable_params)} params from checkpoint.")
        print(f"[INFO] Pretrained modules loaded: {loaded_prefixes}")
        print(f"[INFO] Modules left at random init: {fresh_prefixes}")
        if missing_keys:
            print(f"[WARNING] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {unexpected_keys}")
        if unmatched_keys:
            print(f"[WARNING] {len(unmatched_keys)} keys skipped due to shape/name mismatch.")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_mode = getattr(self.opt, "optimizer", "auto")
        use_adamw = optimizer_mode == "adamw" or (
            optimizer_mode == "auto" and getattr(self.opt, "audio_adapter_mode", "none") != "none"
        )
        beta1 = self.opt.beta1
        beta2 = self.opt.beta2
        weight_decay = self.opt.weight_decay
        if beta1 is None:
            beta1 = 0.9 if use_adamw else 0.5
        if beta2 is None:
            beta2 = 0.95 if use_adamw else 0.999
        if weight_decay is None:
            weight_decay = 0.01 if use_adamw else 0.0

        if use_adamw:
            optimizer = optim.AdamW(params, lr=self.opt.lr, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(params, lr=self.opt.lr, betas=(beta1, beta2), weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.opt.iter,
            eta_min=self.opt.lr_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--dataset_path", "--dataset_pat", dest="dataset_path", default=None, type=str)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--lr_min", default=1e-5, type=float)
        parser.add_argument("--optimizer", default="auto", choices=["auto", "adam", "adamw"])
        parser.add_argument("--beta1", default=None, type=float)
        parser.add_argument("--beta2", default=None, type=float)
        parser.add_argument("--weight_decay", default=None, type=float)
        parser.add_argument("--ema_decay", default=0.9999, type=float)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--val_batch_size", default=8, type=int)
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--iter", default=5000000, type=int)
        parser.add_argument("--exp_path", type=str, default="./exps")
        parser.add_argument("--exp_name", type=str, default="debug")
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--devices", type=int, default=-1)
        parser.add_argument("--precision", type=str, default="32-true")
        parser.add_argument("--log_every_n_steps", type=int, default=50)
        parser.add_argument("--validate_at_start", action="store_true")
        return parser


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):
        self.train_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=-100)
        self.val_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=-100, end=-1)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            num_workers=self.opt.num_workers,
            batch_size=self.opt.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            num_workers=0,
            batch_size=self.opt.val_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )


def main():
    opt = TrainOptions().parse()
    if opt.rank == "cuda" and not torch.cuda.is_available():
        opt.rank = "cpu"

    system = System(opt)
    dm = DataModule(opt)

    if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
        system.load_ckpt(opt.resume_ckpt)

    if opt.bridge_ckpt:
        adapter_sd = torch.load(opt.bridge_ckpt, map_location="cpu")
        if isinstance(adapter_sd, dict) and "state_dict" in adapter_sd and isinstance(adapter_sd["state_dict"], dict):
            adapter_sd = adapter_sd["state_dict"]
        missing, unexpected = system.model.audio_adapter.load_state_dict(adapter_sd, strict=True)
        print(
            f"[INFO] Loaded bridge ckpt from {opt.bridge_ckpt}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    if opt.freeze_generator:
        for name, param in system.model.named_parameters():
            if not name.startswith("audio_adapter."):
                param.requires_grad = False
        print("[INFO] freeze_generator ON: only audio_adapter.* is trainable.")

    if opt.freeze_audio_projection:
        for param in system.model.audio_projection.parameters():
            param.requires_grad = False
        print("[INFO] freeze_audio_projection ON: audio_projection frozen.")

    if opt.unfreeze_early_adaln:
        for block_idx in (0, 1):
            for param in system.model.fmt.blocks[block_idx].adaLN_modulation.parameters():
                param.requires_grad = True
        print("[INFO] Unfroze adaLN_modulation of FMT blocks 0 and 1.")

    system.ema.decay = opt.ema_decay
    system.ema.register()
    trainable = sum(p.numel() for p in system.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in system.model.parameters())
    print(f"[INFO] trainable/total params: {trainable:,} / {total:,}")

    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, "checkpoints"),
        filename="{step:06d}",
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() and opt.rank != "cpu" else "cpu"
    devices = opt.devices if accelerator == "gpu" else 1
    strategy = "ddp_find_unused_parameters_true" if accelerator == "gpu" and torch.cuda.device_count() > 1 else "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_steps=opt.iter,
        val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        precision=opt.precision,
        log_every_n_steps=opt.log_every_n_steps,
    )

    if opt.validate_at_start:
        trainer.validate(system, datamodule=dm, verbose=True)
    trainer.fit(system, dm)


if __name__ == "__main__":
    main()
