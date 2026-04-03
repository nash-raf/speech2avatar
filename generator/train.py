import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.func import jvp
from torch.utils import data

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.dataset import AudioMotionSmirkGazeDataset
from generator.FM import FMGenerator
from generator.generate import DataProcessor, load_smirk_params
from generator.options.base_options import BaseOptions
from renderer.models import IMTRenderer


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))


class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = FMGenerator(opt)
        self.opt = opt
        self.ema = EMA(self.model, decay=0.9999)
        self.preview_renderer = None
        self.preview_processor = None
        self.preview_data = None
        self.last_preview_step = -1
        self.last_preview_video_step = -1
        self.train_batch_start_time = None
        self.validation_epoch_start_time = None
        self.preview_start_time = None

    def on_train_batch_start(self, batch, batch_idx):
        self.train_batch_start_time = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()
        if self.train_batch_start_time is not None:
            step_time = time.perf_counter() - self.train_batch_start_time
            batch_size = batch["m_now"].shape[0]
            samples_per_sec = batch_size / max(step_time, 1e-6)
            self.log("perf/train_step_sec", step_time, prog_bar=False, on_step=True, on_epoch=False)
            self.log("perf/train_samples_per_sec", samples_per_sec, prog_bar=False, on_step=True, on_epoch=False)

    def on_validation_epoch_start(self):
        self.ema.apply_shadow()
        self.validation_epoch_start_time = time.perf_counter()

    def on_validation_epoch_end(self):
        self._log_preview()
        if self.validation_epoch_start_time is not None:
            val_time = time.perf_counter() - self.validation_epoch_start_time
            self.log("perf/val_epoch_sec", val_time, prog_bar=False, on_step=False, on_epoch=True)
            print(f"[TIMING] validation_epoch={val_time:.2f}s at step={self.global_step}")
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.shadow = checkpoint["ema_state_dict"]

    def on_after_backward(self):
        if self.opt.max_grad_norm is not None and self.opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_grad_norm)

    def _prepare_batch(self, batch):
        alias_pairs = [
            ("gaze", "gaze_now"),
            ("pose", "pose_now"),
            ("cam", "cam_now"),
        ]
        for canonical_key, alias_key in alias_pairs:
            if canonical_key not in batch and alias_key in batch:
                batch[canonical_key] = batch[alias_key]
        return batch

    def _logit_normal_dist(self, batch_size, device):
        rnd_normal = torch.randn(batch_size, device=device)
        return torch.sigmoid(rnd_normal * self.opt.P_std + self.opt.P_mean)

    def _sample_tr(self, batch_size, device):
        t = self._logit_normal_dist(batch_size, device)
        r = self._logit_normal_dist(batch_size, device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(batch_size * self.opt.data_proportion)
        fm_mask = torch.arange(batch_size, device=device) < data_size
        r = torch.where(fm_mask, t, r)
        return t, r, fm_mask

    def _sample_cfg_scale(self, batch_size, device, s_max=7.0):
        u = torch.rand(batch_size, device=device)
        if self.opt.cfg_beta == 1.0:
            return torch.exp(u * math.log1p(s_max))

        smax = torch.tensor(s_max, device=device, dtype=torch.float32)
        beta = torch.tensor(self.opt.cfg_beta, device=device, dtype=torch.float32)
        log_base = (1.0 - beta) * torch.log1p(smax)
        log_inner = torch.log1p(u * torch.expm1(log_base))
        return torch.exp(log_inner / (1.0 - beta))

    def _sample_cfg_interval(self, batch_size, fm_mask, device):
        t_min = torch.rand(batch_size, device=device) * 0.5
        t_max = 0.5 + torch.rand(batch_size, device=device) * 0.5
        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        return t_min, t_max

    def _adaptive_weight(self, loss):
        adp_wt = (loss + self.opt.norm_eps) ** self.opt.norm_p
        return loss / adp_wt.detach()

    def _compute_loss(self, batch):
        batch = self._prepare_batch(batch)
        x = batch["m_now"]
        batch_size = x.size(0)
        device = x.device

        t, r, fm_mask = self._sample_tr(batch_size, device)
        noise = torch.randn_like(x)
        t_view = append_dims(t, x.ndim - 1)
        z_t = (1 - t_view) * x + t_view * noise
        v_t = noise - x

        omega = self._sample_cfg_scale(batch_size, device)
        t_min, t_max = self._sample_cfg_interval(batch_size, fm_mask, device)
        zero = torch.zeros_like(t)
        one = torch.ones_like(t)
        batch_with_noise = dict(batch)
        batch_with_noise["m_now"] = z_t

        _, v_c_fm = self.model(
            batch_with_noise,
            h=zero,
            omega=omega,
            t_min=zero,
            t_max=one,
            zero_audio_mask=None,
            return_v=True,
        )
        _, v_u = self.model(
            batch_with_noise,
            h=zero,
            omega=one,
            t_min=zero,
            t_max=one,
            zero_audio_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
            return_v=True,
        )
        v_g_fm = v_t + append_dims(1 - 1 / omega, v_t.ndim - 1) * (v_c_fm - v_u)

        omega_interval = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        _, v_c = self.model(
            batch_with_noise,
            h=zero,
            omega=omega_interval,
            t_min=t_min,
            t_max=t_max,
            zero_audio_mask=None,
            return_v=True,
        )
        v_g = v_t + append_dims(1 - 1 / omega_interval, v_t.ndim - 1) * (v_c - v_u)
        v_g = torch.where(append_dims(fm_mask, v_g.ndim - 1), v_g_fm, v_g)

        audio_drop_mask = torch.rand(batch_size, device=device) < self.opt.audio_dropout_prob
        v_g = torch.where(append_dims(audio_drop_mask, v_g.ndim - 1), v_t, v_g)
        v_c_for_jvp = torch.where(append_dims(audio_drop_mask, v_c.ndim - 1), v_t, v_c)

        h = t - r

        def u_only(z_in, h_in):
            local_batch = dict(batch_with_noise)
            local_batch["m_now"] = z_in
            return self.model(
                local_batch,
                h=h_in,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
                zero_audio_mask=audio_drop_mask,
                return_v=False,
            )

        u, du_dt = jvp(
            u_only,
            (z_t, h),
            (v_c_for_jvp, torch.ones_like(h)),
        )
        _, v = self.model(
            batch_with_noise,
            h=h,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            zero_audio_mask=audio_drop_mask,
            return_v=True,
        )

        v_g = v_g.detach()
        V = u + append_dims(h, u.ndim - 1) * du_dt.detach()

        loss_u_raw = torch.sum((V - v_g) ** 2, dim=(1, 2))
        loss_v_raw = torch.sum((v - v_g) ** 2, dim=(1, 2))
        loss_u = self._adaptive_weight(loss_u_raw).mean()
        loss_v = self._adaptive_weight(loss_v_raw).mean()
        loss = loss_u + loss_v

        metrics = {
            "loss": loss,
            "loss_u": ((V - v_g) ** 2).mean().detach(),
            "loss_v": ((v - v_g) ** 2).mean().detach(),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_u", metrics["loss_u"], prog_bar=True)
        self.log("loss_v", metrics["loss_v"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            loss, metrics = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_u", metrics["loss_u"], prog_bar=True)
        self.log("val_loss_v", metrics["loss_v"], prog_bar=True)

    def _init_preview(self):
        if not self.opt.preview_ref_path or not self.opt.preview_aud_path:
            return False
        if self.preview_renderer is not None:
            return True
        if not os.path.exists(self.opt.renderer_path):
            print(f"[WARNING] Preview disabled: renderer checkpoint not found at {self.opt.renderer_path}")
            return False

        self.preview_renderer = IMTRenderer(self.opt).to(self.device).eval()
        renderer_ckpt = torch.load(self.opt.renderer_path, map_location="cpu")["state_dict"]
        ae_state_dict = {k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")}
        self.preview_renderer.load_state_dict(ae_state_dict, strict=False)
        self.preview_processor = DataProcessor(self.opt)
        return True

    @torch.no_grad()
    def _load_preview_data(self):
        if self.preview_data is not None:
            return self.preview_data
        if not self._init_preview():
            return None

        data = self.preview_processor.preprocess(
            self.opt.preview_ref_path,
            self.opt.preview_aud_path,
            crop=self.opt.preview_crop,
        )
        data["s"] = data["s"].to(self.device)
        data["a"] = data["a"].to(self.device)

        if self.opt.preview_pose_path and os.path.exists(self.opt.preview_pose_path):
            pose, cam = load_smirk_params(torch.load(self.opt.preview_pose_path, map_location="cpu"))
            data["pose"] = pose.to(self.device)
            data["cam"] = cam.to(self.device)
        else:
            data["pose"], data["cam"] = None, None

        if self.opt.preview_gaze_path and os.path.exists(self.opt.preview_gaze_path):
            gaze = torch.from_numpy(np.load(self.opt.preview_gaze_path)).float().to(self.device)
            data["gaze"] = gaze
        else:
            data["gaze"] = None

        self.preview_data = data
        return data

    @torch.no_grad()
    def _log_preview(self):
        if not self.trainer.is_global_zero:
            return
        if self.global_step == 0:
            return
        need_image = (
            self.opt.preview_every_n_val > 0
            and self.global_step != self.last_preview_step
            and self.global_step % self.opt.preview_every_n_val == 0
        )
        need_video = (
            self.opt.preview_video_every_n_val > 0
            and self.global_step != self.last_preview_video_step
            and self.global_step % self.opt.preview_video_every_n_val == 0
        )
        if not need_image and not need_video:
            return

        data = self._load_preview_data()
        if data is None:
            return

        self.preview_start_time = time.perf_counter()
        self.model.eval()
        f_r, g_r = self.preview_renderer.dense_feature_encoder(data["s"])
        t_r = self.preview_renderer.latent_token_encoder(data["s"])
        ta_r = self.preview_renderer.adapt(t_r, g_r)
        m_r = self.preview_renderer.latent_token_decoder(ta_r)

        sample_input = {
            "a": data["a"],
            "ref_x": t_r,
            "pose": data["pose"],
            "cam": data["cam"],
            "gaze": data["gaze"],
        }
        sample = self.model.sample(
            sample_input,
            a_cfg_scale=self.opt.preview_a_cfg_scale,
            nfe=self.opt.nfe,
            seed=self.opt.seed,
        )

        frames = []
        for t in range(sample.shape[1]):
            ta_c = self.preview_renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.preview_renderer.latent_token_decoder(ta_c)
            frames.append(self.preview_renderer.decode(m_c, m_r, f_r))
        frames = torch.stack(frames, dim=1)[0]

        num_frames = frames.shape[0]
        indices = sorted({0, num_frames // 2, num_frames - 1})
        preview = torch.cat([frames[idx] for idx in indices], dim=2)
        preview = ((preview.clamp(-1, 1) + 1) / 2).cpu()

        if need_image and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image("preview/generated_triplet", preview, self.global_step)
            self.last_preview_step = self.global_step

        if need_video:
            self._save_preview_video(frames)
            self.last_preview_video_step = self.global_step

        if self.preview_start_time is not None:
            preview_time = time.perf_counter() - self.preview_start_time
            self.log("perf/preview_sec", preview_time, prog_bar=False, on_step=False, on_epoch=False)
            print(
                f"[TIMING] preview step={self.global_step} "
                f"image={'yes' if need_image else 'no'} "
                f"video={'yes' if need_video else 'no'} "
                f"time={preview_time:.2f}s"
            )

        self.model.train()

    def _save_preview_video(self, frames):
        preview_dir = Path(self.opt.exp_path) / self.opt.exp_name / "preview_videos"
        preview_dir.mkdir(parents=True, exist_ok=True)
        out_path = preview_dir / f"step={self.global_step:06d}.mp4"

        frames_uint8 = (
            ((frames.clamp(-1, 1) + 1) * 127.5)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, frame in enumerate(frames_uint8):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(tmpdir, f"{idx:06d}.png"), frame_bgr)

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(self.opt.fps),
                "-i",
                os.path.join(tmpdir, "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"[INFO] Saved preview video to {out_path}")

    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "ema_state_dict" in ckpt:
            print("[INFO] Found EMA weights in checkpoint. Loading EMA weights for better stability.")
            state_dict = ckpt["ema_state_dict"]
        else:
            print("[INFO] EMA weights not found. Loading standard state_dict.")
            state_dict = ckpt.get("state_dict", ckpt)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model_state_dict = self.model.state_dict()
        loadable_params = {}
        unmatched_keys = []
        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                loadable_params[key] = value
            else:
                unmatched_keys.append(key)

        missing_keys, _ = self.model.load_state_dict(loadable_params, strict=False)
        self.ema.register()

        print(f"[INFO] Loaded {len(loadable_params)} params from checkpoint.")
        if missing_keys:
            print(f"[WARNING] Missing keys: {missing_keys}")
        if unmatched_keys:
            print(f"[WARNING] {len(unmatched_keys)} keys skipped.")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.opt.iter,
            eta_min=max(self.opt.lr * 0.1, 1e-6),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = super().initialize(parser)
        default_cpu_workers = max(4, min(16, os.cpu_count() or 8))
        parser.add_argument("--dataset_path", default=None, type=str)
        parser.add_argument("--dataset_pat", dest="dataset_path", default=None, type=str, help=argparse.SUPPRESS)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--iter", default=5000000, type=int)
        parser.add_argument("--exp_path", type=str, default="./exps")
        parser.add_argument("--exp_name", type=str, default="debug")
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer.ckpt")
        parser.add_argument("--train_num_workers", type=int, default=default_cpu_workers)
        parser.add_argument("--val_num_workers", type=int, default=max(2, default_cpu_workers // 2))
        parser.add_argument("--prefetch_factor", type=int, default=4)
        parser.add_argument("--precision", type=str, default="bf16-mixed")
        parser.add_argument("--preview_ref_path", type=str, default=None)
        parser.add_argument("--preview_aud_path", type=str, default=None)
        parser.add_argument("--preview_pose_path", type=str, default=None)
        parser.add_argument("--preview_gaze_path", type=str, default=None)
        parser.add_argument("--preview_crop", action="store_true")
        parser.add_argument("--preview_every_n_val", type=int, default=5000)
        parser.add_argument("--preview_video_every_n_val", type=int, default=15000)
        parser.add_argument("--preview_a_cfg_scale", type=float, default=1.0)
        parser.add_argument("--val_holdout", type=int, default=100)
        return parser


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage):
        if self.opt.val_holdout <= 0:
            self.train_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=None)
            self.val_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=1)
        else:
            self.train_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=-self.opt.val_holdout)
            self.val_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=-self.opt.val_holdout, end=None)

    def _loader_kwargs(self, num_workers, shuffle, drop_last):
        kwargs = {
            "num_workers": num_workers,
            "batch_size": self.opt.batch_size if shuffle else min(8, self.opt.batch_size),
            "shuffle": shuffle,
            "drop_last": drop_last,
            "pin_memory": True,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.opt.prefetch_factor
        return kwargs

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            **self._loader_kwargs(
                num_workers=self.opt.train_num_workers,
                shuffle=True,
                drop_last=True,
            ),
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            **self._loader_kwargs(
                num_workers=self.opt.val_num_workers,
                shuffle=False,
                drop_last=False,
            ),
        )


if __name__ == "__main__":
    opt = TrainOptions().parse()
    system = System(opt)
    dm = DataModule(opt)

    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, "checkpoints"),
        filename="{step:06d}",
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True,
    )

    if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
        system.load_ckpt(opt.resume_ckpt)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        max_steps=opt.iter,
        val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        precision=opt.precision,
    )

    trainer.fit(system, dm)
