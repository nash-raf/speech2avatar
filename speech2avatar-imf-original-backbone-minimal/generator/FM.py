import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from generator.wav2vec2 import Wav2VecModel
from generator.FMT import FlowMatchingTransformer


class FMGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fps = opt.fps
        self.rank = opt.rank

        # Load motion normalization stats (computed offline from training data)
        self._load_motion_stats(opt)

        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames

        self.audio_input_dim = 768 if opt.only_last_features else 12 * 768

        self.audio_encoder = AudioEncoder(opt)
        self.fmt = FlowMatchingTransformer(opt)

        self.audio_projection = self._make_projection(self.audio_input_dim, opt.dim_c)
        self.gaze_projection = self._make_projection(2, opt.dim_c)
        self.pose_projection = self._make_projection(3, opt.dim_c)
        self.cam_projection = self._make_projection(3, opt.dim_c)

        if getattr(opt, "freeze_frontend", False):
            self._freeze_frontend()
        self._print_model_stats()

    def _load_motion_stats(self, opt):
        """Load per-dimension motion normalization stats for unnormalizing at inference."""
        stats_path = None
        if hasattr(opt, "dataset_path") and opt.dataset_path:
            stats_path = Path(opt.dataset_path) / "motion_stats.pt"
        if stats_path and stats_path.exists():
            stats = torch.load(stats_path, map_location="cpu")
            self.register_buffer("motion_mean", stats["mean"])
            self.register_buffer("motion_std", stats["std"])
            print(f"[Info] Loaded motion stats for unnormalization: mean_range=[{stats['mean'].min():.2f}, {stats['mean'].max():.2f}]")
        else:
            self.motion_mean = None
            self.motion_std = None
            print("[Warning] No motion_stats.pt found — sampling will output raw (unnormalized) latents")

    def _unnormalize_motion(self, m):
        """Convert normalized motion latents back to renderer space."""
        if self.motion_mean is not None:
            return m * self.motion_std.to(m.device) + self.motion_mean.to(m.device)
        return m

    def _normalize_motion(self, m):
        """Normalize motion latents to zero-mean, unit-variance (for ref_x at inference)."""
        if self.motion_mean is not None:
            return (m - self.motion_mean.to(m.device)) / self.motion_std.to(m.device)
        return m

    def _make_projection(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_frontend(self):
        # Keep the original input/conditioning front-end fixed and only adapt the motion backbone + heads.
        self._freeze_module(self.audio_projection)
        self._freeze_module(self.gaze_projection)
        self._freeze_module(self.pose_projection)
        self._freeze_module(self.cam_projection)

    def _print_model_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n[Model Stats] Parameters: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")

    def _get_batch_item(self, batch, name):
        if name in batch:
            return batch[name]
        now_key = f"{name}_now"
        if now_key in batch:
            return batch[now_key]
        raise KeyError(f"Missing batch key: {name}")

    def _get_optional_batch_item(self, batch, name, like_tensor, dim):
        if name in batch:
            return batch[name]
        now_key = f"{name}_now"
        if now_key in batch:
            return batch[now_key]
        return torch.zeros(*like_tensor.shape[:-1], dim, device=like_tensor.device, dtype=like_tensor.dtype)

    def _project_training_batch(self, batch, zero_audio_mask=None):
        x = batch["m_now"]
        prev_x = batch["m_prev"]
        a = batch["a_now"]
        prev_a = batch["a_prev"]
        ref_x = batch["m_ref"]

        gaze = self._get_batch_item(batch, "gaze")
        prev_gaze = batch["gaze_prev"]
        pose = self._get_batch_item(batch, "pose")
        prev_pose = batch["pose_prev"]
        cam = self._get_optional_batch_item(batch, "cam", pose, 3)
        prev_cam = batch.get(
            "cam_prev",
            torch.zeros(*prev_pose.shape[:-1], 3, device=prev_pose.device, dtype=prev_pose.dtype),
        )

        bsz = x.size(0)
        if not self.opt.only_last_features:
            a = a.view(bsz, self.num_frames_for_clip, -1)
            prev_a = prev_a.view(bsz, self.num_prev_frames, -1)

        a = self.audio_projection(a)
        prev_a = self.audio_projection(prev_a)
        if zero_audio_mask is not None and zero_audio_mask.any():
            a = a.clone()
            prev_a = prev_a.clone()
            a[zero_audio_mask] = 0
            prev_a[zero_audio_mask] = 0

        gaze = self.gaze_projection(gaze)
        prev_gaze = self.gaze_projection(prev_gaze)
        pose = self.pose_projection(pose)
        prev_pose = self.pose_projection(prev_pose)
        cam = self.cam_projection(cam)
        prev_cam = self.cam_projection(prev_cam)

        return {
            "x": x,
            "prev_x": prev_x,
            "a": a,
            "prev_a": prev_a,
            "ref_x": ref_x,
            "gaze": gaze,
            "prev_gaze": prev_gaze,
            "pose": pose,
            "prev_pose": prev_pose,
            "cam": cam,
            "prev_cam": prev_cam,
        }

    def _predict_projected(self, projected, h, omega, t_min, t_max, return_v=True):
        return self.fmt(
            x=projected["x"],
            a=projected["a"],
            prev_x=projected["prev_x"],
            prev_a=projected["prev_a"],
            ref_x=projected["ref_x"],
            gaze=projected["gaze"],
            prev_gaze=projected["prev_gaze"],
            pose=projected["pose"],
            prev_pose=projected["prev_pose"],
            cam=projected["cam"],
            prev_cam=projected["prev_cam"],
            h=h,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            return_v=return_v,
        )

    def forward(self, batch, h, omega, t_min, t_max, zero_audio_mask=None, return_v=True):
        projected = self._project_training_batch(batch, zero_audio_mask=zero_audio_mask)
        out = self._predict_projected(projected, h, omega, t_min, t_max, return_v=return_v)
        if return_v:
            u, v = out
            return u[:, self.num_prev_frames:], v[:, self.num_prev_frames:]
        return out[:, self.num_prev_frames:]

    def _align_sequence(self, tensor, target_len):
        if tensor is None:
            return None
        tensor = tensor.to(self.rank)
        curr_len = tensor.shape[0]
        if curr_len > target_len:
            return tensor[:target_len]
        if curr_len < target_len:
            pad_len = target_len - curr_len
            padding = torch.zeros(pad_len, tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        return tensor

    @torch.no_grad()
    def sample(self, data, a_cfg_scale=1.0, nfe=1, seed=None):
        a, ref_x = data["a"], data["ref_x"]
        gaze_raw = data.get("gaze")
        pose_raw = data.get("pose")
        cam_raw = data.get("cam")

        device = self.rank
        a = a.to(device)
        ref_x = ref_x.to(device)
        if ref_x.ndim == 1:
            ref_x = ref_x.unsqueeze(0)

        batch_size = a.shape[0]
        total_frames = math.ceil(a.shape[-1] * self.fps / self.opt.sampling_rate)

        a = self.audio_encoder.inference(a, seq_len=total_frames)
        a = self.audio_projection(a)

        gaze = self._align_sequence(gaze_raw, total_frames)
        pose = self._align_sequence(pose_raw, total_frames)
        cam = self._align_sequence(cam_raw, total_frames)

        if gaze is None:
            gaze = torch.zeros(total_frames, 2, device=device)
        if pose is None:
            pose = torch.zeros(total_frames, 3, device=device)
        if cam is None:
            cam = torch.zeros(total_frames, 3, device=device)

        gaze = self.gaze_projection(gaze).unsqueeze(0)
        pose = self.pose_projection(pose).unsqueeze(0)
        cam = self.cam_projection(cam).unsqueeze(0)

        samples = []
        num_chunks = int(math.ceil(total_frames / self.num_frames_for_clip))
        t_steps = torch.linspace(1.0, 0.0, nfe + 1, device=device)
        prev_a_ctx = None
        prev_gaze_ctx = None
        prev_pose_ctx = None
        prev_cam_ctx = None

        def pad_last_chunk(tensor):
            if tensor.shape[1] == self.num_frames_for_clip:
                return tensor
            pad_len = self.num_frames_for_clip - tensor.shape[1]
            tail = tensor[:, -1:, :].expand(-1, pad_len, -1)
            return torch.cat([tensor, tail], dim=1)

        for chunk_idx in range(num_chunks):
            if self.opt.fix_noise_seed:
                current_seed = self.opt.seed if seed is None else seed
                generator = torch.Generator(device=device)
                generator.manual_seed(current_seed + chunk_idx)
                z_t = torch.randn(
                    batch_size,
                    self.num_frames_for_clip,
                    self.opt.dim_motion,
                    device=device,
                    generator=generator,
                )
            else:
                z_t = torch.randn(batch_size, self.num_frames_for_clip, self.opt.dim_motion, device=device)

            start_idx = chunk_idx * self.num_frames_for_clip
            end_idx = (chunk_idx + 1) * self.num_frames_for_clip

            a_t = pad_last_chunk(a[:, start_idx:end_idx])
            gaze_t = pad_last_chunk(gaze[:, start_idx:end_idx])
            pose_t = pad_last_chunk(pose[:, start_idx:end_idx])
            cam_t = pad_last_chunk(cam[:, start_idx:end_idx])

            if chunk_idx == 0:
                prev_x_t = torch.zeros(batch_size, self.num_prev_frames, self.opt.dim_motion, device=device)
                prev_a_t = torch.zeros(batch_size, self.num_prev_frames, a.shape[-1], device=device)
                prev_gaze_t = torch.zeros(batch_size, self.num_prev_frames, gaze.shape[-1], device=device)
                prev_pose_t = torch.zeros(batch_size, self.num_prev_frames, pose.shape[-1], device=device)
                prev_cam_t = torch.zeros(batch_size, self.num_prev_frames, cam.shape[-1], device=device)
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames:]
                prev_a_t = prev_a_ctx
                prev_gaze_t = prev_gaze_ctx
                prev_pose_t = prev_pose_ctx
                prev_cam_t = prev_cam_ctx

            projected = {
                "x": z_t,
                "prev_x": prev_x_t,
                "a": a_t,
                "prev_a": prev_a_t,
                "ref_x": ref_x,
                "gaze": gaze_t,
                "prev_gaze": prev_gaze_t,
                "pose": pose_t,
                "prev_pose": prev_pose_t,
                "cam": cam_t,
                "prev_cam": prev_cam_t,
            }

            omega = torch.full((batch_size,), max(float(a_cfg_scale), 1.0), device=device)
            t_min = torch.zeros(batch_size, device=device)
            t_max = torch.ones(batch_size, device=device)

            for step_idx in range(nfe):
                t = torch.full((batch_size,), t_steps[step_idx], device=device)
                r = torch.full((batch_size,), t_steps[step_idx + 1], device=device)
                h = t - r
                projected["x"] = z_t
                u = self._predict_projected(projected, h, omega, t_min, t_max, return_v=False)
                z_t = z_t - h[:, None, None] * u[:, self.num_prev_frames:]

            sample_t = z_t
            samples.append(sample_t)
            prev_a_ctx = a_t[:, -self.num_prev_frames:]
            prev_gaze_ctx = gaze_t[:, -self.num_prev_frames:]
            prev_pose_ctx = pose_t[:, -self.num_prev_frames:]
            prev_cam_ctx = cam_t[:, -self.num_prev_frames:]

        sample = torch.cat(samples, dim=1)[:, :total_frames]
        # Unnormalize from training space back to renderer space
        return self._unnormalize_motion(sample)


class AudioEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate

        self.num_frames_for_clip = int(opt.wav2vec_sec * self.fps)
        self.num_prev_frames = int(opt.num_prev_frames)

        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.wav2vec2.feature_extractor._freeze_parameters()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _pad_audio(self, a, target_frames):
        target_samples = int(target_frames * self.sampling_rate / self.fps)
        if a.shape[1] % target_samples != 0:
            diff = target_samples - (a.shape[1] % target_samples)
            a = F.pad(a, (0, diff), mode="replicate")
        return a

    def get_wav2vec2_feature(self, a, seq_len):
        out = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
        if self.only_last_features:
            return out.last_hidden_state
        feat = torch.stack(out.hidden_states[1:], dim=1)
        feat = feat.permute(0, 2, 1, 3)
        return feat.reshape(feat.shape[0], feat.shape[1], -1)

    def forward(self, a, prev_a=None):
        total_frames = self.num_frames_for_clip
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
            total_frames += self.num_prev_frames
        a = self._pad_audio(a, total_frames)
        return self.get_wav2vec2_feature(a, seq_len=total_frames)

    @torch.no_grad()
    def inference(self, a, seq_len):
        a = self._pad_audio(a, seq_len)
        return self.get_wav2vec2_feature(a, seq_len=seq_len)
