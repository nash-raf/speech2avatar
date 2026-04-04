import os
import time
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch import pi
from torch.nn import Module
from torch.utils import data
from torch import nn, optim
from einops import rearrange, repeat
from generator.dataset import AudioMotionSmirkGazeDataset
from FM import FMGenerator
from options.base_options import BaseOptions
from pytorch_lightning.loggers import TensorBoardLogger

# ==========================================
# 1. New EMA Class Helper
# ==========================================
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
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def cosmap(t):
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

class L1loss(Module):
    def forward(self, pred, target, **kwargs):
        return F.l1_loss(pred, target)

class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = FMGenerator(opt)
        self.opt = opt
        self.loss_fn = L1loss()

        # Freeze non-audio projections for Mimi experiment stability
        if getattr(opt, 'freeze_non_audio_proj', False):
            for proj in [self.model.gaze_projection, self.model.pose_projection, self.model.cam_projection]:
                for p in proj.parameters():
                    p.requires_grad = False
            print("[INFO] Froze gaze/pose/cam projections.")

        self.ema = EMA(self.model, decay=0.9999)

    def forward(self, x):
        return self.model(x)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    def on_validation_epoch_start(self):
        self.ema.apply_shadow()

    def on_validation_epoch_end(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.shadow = checkpoint["ema_state_dict"]

    def training_step(self, batch, batch_idx):
        m_now = batch["m_now"]

        noise = torch.randn_like(m_now)
        times = torch.rand(m_now.size(0), device=self.device)
        t = append_dims(times, m_now.ndim - 1)
        noised_motion = t * m_now + (1 - t) * noise
        gt_flow = m_now - noise

        batch["m_now"] = noised_motion

        pred_flow_anchor = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow_anchor, gt_flow)
        velocity_loss = self.loss_fn(pred_flow_anchor[:, 1:] - pred_flow_anchor[:, :-1], 
                                     gt_flow[:, 1:] - gt_flow[:, :-1])

        train_loss = fm_loss + velocity_loss

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("fm_loss", fm_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        m_now = batch["m_now"]
        noise = torch.randn_like(m_now); times = torch.rand(m_now.size(0), device=self.device); t = append_dims(times, m_now.ndim - 1)
        noised_motion = t * m_now + (1 - t) * noise; gt_flow = m_now - noise
        batch["m_now"] = noised_motion
        pred_flow_anchor = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow_anchor, gt_flow)
        velocity_loss = self.loss_fn(pred_flow_anchor[:, 1:] - pred_flow_anchor[:, :-1], 
                                     gt_flow[:, 1:] - gt_flow[:, :-1])

        val_loss = fm_loss + velocity_loss

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_fm_loss", fm_loss, prog_bar=True)
    
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

        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                loadable_params[k] = v
            else:
                unmatched_keys.append(k)

        missing_keys, unexpected_keys = self.model.load_state_dict(loadable_params, strict=False)

        self.ema.register()

        print(f"[INFO] Loaded {len(loadable_params)} params from checkpoint.")
        if missing_keys:
            print(f"[WARNING] Missing keys: {missing_keys}")
        if unmatched_keys:
            print(f"[WARNING] {len(unmatched_keys)} keys skipped.")

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.opt.iter, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": scheduler}
    
class VideoGenCallback(pl.Callback):
    """Generates a validation video every N steps using EMA weights."""

    def __init__(self, opt):
        self.opt = opt
        self._ready = False
        self.renderer = None
        self.f_r = self.t_r = self.g_r = None
        self.a_data = None

    def _setup(self, device):
        import cv2
        import torchvision.transforms as T
        from PIL import Image
        from renderer.models import IMTRenderer

        # Load renderer
        self.renderer = IMTRenderer(self.opt).to(device).eval()
        ckpt = torch.load(self.opt.renderer_path, map_location='cpu')['state_dict']
        ae_sd = {k[len('gen.'):]: v for k, v in ckpt.items() if k.startswith('gen.')}
        self.renderer.load_state_dict(ae_sd, strict=False)

        # Encode reference image (done once, cached)
        img = cv2.cvtColor(cv2.imread(self.opt.val_ref_path), cv2.COLOR_BGR2RGB)
        img_t = T.Compose([T.Resize((512, 512)), T.ToTensor()])(
            Image.fromarray(img)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            f_r, g_r = self.renderer.dense_feature_encoder(img_t)
            t_r = self.renderer.latent_token_encoder(img_t)
        self.f_r, self.g_r, self.t_r = f_r, g_r, t_r

        # Load audio features (done once, cached)
        feat_path = getattr(self.opt, 'val_audio_feat_path', None)
        if feat_path and os.path.exists(feat_path):
            a_feat = torch.from_numpy(np.load(feat_path)).float()
            if a_feat.ndim == 2:
                a_feat = a_feat.unsqueeze(0)
            self.a_data = {'a_feat': a_feat.to(device)}
        else:
            # Fallback: raw waveform via Wav2Vec
            import librosa
            from transformers import Wav2Vec2FeatureExtractor
            preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.opt.wav2vec_model_path, local_files_only=True
            )
            speech, sr = librosa.load(self.opt.val_aud_path, sr=self.opt.sampling_rate)
            raw = preprocessor(speech, sampling_rate=sr, return_tensors='pt').input_values[0]
            self.a_data = {'a': raw.unsqueeze(0).to(device)}

        self._ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.opt.val_video_freq != 0:
            return
        if trainer.global_rank != 0:
            return
        if not getattr(self.opt, 'renderer_path', None):
            return

        device = next(pl_module.model.parameters()).device
        if not self._ready:
            try:
                self._setup(device)
            except Exception as e:
                print(f"[VideoGenCallback] Setup failed: {e}")
                return

        pl_module.ema.apply_shadow()
        try:
            self._generate(pl_module.model, trainer, step)
        except Exception as e:
            print(f"[VideoGenCallback] Generation failed at step {step}: {e}")
        finally:
            pl_module.ema.restore()

    @torch.no_grad()
    def _generate(self, fm, trainer, step):
        device = next(fm.parameters()).device
        data = {k: v for k, v in self.a_data.items()}
        data['ref_x'] = self.t_r

        sample = fm.sample(
            data,
            a_cfg_scale=getattr(self.opt, 'a_cfg_scale', 1.0),
            nfe=getattr(self.opt, 'nfe', 10),
        )

        # Decode through renderer
        T = sample.shape[1]
        ta_r = self.renderer.adapt(self.t_r, self.g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        frames = []
        for i in range(T):
            ta_c = self.renderer.adapt(sample[:, i], self.g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            frames.append(self.renderer.decode(m_c, m_r, self.f_r))

        vid = torch.stack(frames, dim=0).squeeze(1)       # [T, C, H, W]
        vid = vid.permute(0, 2, 3, 1).clamp(-1, 1).cpu()  # [T, H, W, C]
        vid = (vid * 255).byte()

        # Save to disk
        out_dir = os.path.join(self.opt.exp_path, self.opt.exp_name, 'val_videos')
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, f'step_{step:07d}.mp4')

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        torchvision.io.write_video(tmp_path, vid, fps=self.opt.fps)
        cmd = (
            f"ffmpeg -i {tmp_path} -i {self.opt.val_aud_path} "
            f"-c:v copy -c:a aac {video_path} -y -loglevel quiet"
        )
        subprocess.call(cmd, shell=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Log to TensorBoard as video
        if trainer.logger:
            vid_tb = vid.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0  # [1, T, C, H, W]
            trainer.logger.experiment.add_video(
                'val/generated', vid_tb, global_step=step, fps=int(self.opt.fps)
            )

        print(f"[VideoGenCallback] Saved: {video_path}")


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--dataset_path", default=None, type=str)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--iter', default=5000000, type=int)
        parser.add_argument("--exp_path", type=str, default='./exps')
        parser.add_argument("--exp_name", type=str, default='debug')
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--freeze_non_audio_proj", action='store_true',
                            help='Freeze gaze/pose/cam projections (recommended for Mimi experiments)')

        # Validation video generation
        parser.add_argument("--renderer_path", type=str, default=None,
                            help='Renderer checkpoint for validation video generation')
        parser.add_argument("--val_ref_path", type=str, default="./assets/source_5.png")
        parser.add_argument("--val_aud_path", type=str, default="./assets/audio_3.wav")
        parser.add_argument("--val_audio_feat_path", type=str, default=None,
                            help='Precomputed audio features .npy for val video (bypasses Wav2Vec)')
        parser.add_argument("--val_video_freq", type=int, default=4000,
                            help='Generate validation video every N steps')

        return parser

class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage):
        self.train_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=-100)
        self.val_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=-100, end=-1)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, num_workers=8, batch_size=self.opt.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, num_workers=0, batch_size=8, shuffle=False)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    system = System(opt)
    dm = DataModule(opt)

    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True
    )
    if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
        system.load_ckpt(opt.resume_ckpt)

    callbacks = [checkpoint_callback, VideoGenCallback(opt)]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=opt.iter,
        val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    trainer.fit(system, dm)


