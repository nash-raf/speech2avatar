import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from generator.FMT import FlowMatchingTransformer
from generator.wav2vec2 import Wav2VecModel


class AudioBridge768(nn.Module):
    """Mimi in_dim -> 768 via a shallow MLP.

    Intentionally no output LayerNorm: the downstream audio_projection was
    trained on raw wav2vec last_hidden_state, which is not unit-normalised
    per frame. An output LN here would force the bridge into an OOD
    distribution for the frozen audio_projection.
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_ln(x)
        return self.fc2(self.act(self.fc1(x)))


class AudioTemporalConv32(nn.Module):
    """Mimi in_dim -> 32 with local temporal context."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 32,
        channels: int = 256,
        kernel: int = 5,
    ):
        super().__init__()
        pad = kernel // 2
        self.in_ln = nn.LayerNorm(in_dim)
        self.fc_in = nn.Linear(in_dim, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, groups=8)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, groups=8)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(channels, out_dim)
        self.out_ln = nn.LayerNorm(out_dim)
        self.out_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_ln(x)
        x = self.fc_in(x)
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.fc_out(x)
        return self.out_act(self.out_ln(x))


class FMGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fps = opt.fps
        self.rank = opt.rank

        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames

        self.audio_input_dim = getattr(
            opt, "audio_feat_dim", 768 if opt.only_last_features else 12 * 768
        )

        self.audio_encoder = AudioEncoder(opt)
        self.fmt = FlowMatchingTransformer(opt)

        mode = getattr(opt, "audio_adapter_mode", "none")
        if mode == "none":
            self.audio_adapter = nn.Identity()
            self.audio_projection = self._make_projection(self.audio_input_dim, opt.dim_c)
        elif mode == "bridge_to_768":
            self.audio_adapter = AudioBridge768(
                self.audio_input_dim, 768, hidden_dim=opt.adapter_hidden_dim
            )
            self.audio_projection = self._make_projection(768, opt.dim_c)
        elif mode == "temporal_conv_to_32":
            self.audio_adapter = AudioTemporalConv32(
                self.audio_input_dim,
                opt.dim_c,
                channels=opt.adapter_conv_channels,
                kernel=opt.adapter_conv_kernel,
            )
            self.audio_projection = nn.Identity()
        else:
            raise ValueError(f"Unknown audio_adapter_mode: {mode}")

        self.gaze_projection = self._make_projection(2, opt.dim_c)
        self.pose_projection = self._make_projection(3, opt.dim_c)
        self.cam_projection = self._make_projection(3, opt.dim_c)

        self.odeint_kwargs = {
            "atol": opt.ode_atol,
            "rtol": opt.ode_rtol,
            "method": opt.torchdiffeq_ode_method,
        }

        self._print_model_stats()

    def _device(self) -> torch.device:
        if isinstance(self.rank, torch.device):
            return self.rank
        if isinstance(self.rank, int):
            return torch.device(f"cuda:{self.rank}")
        return torch.device(self.rank)

    def _make_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def _print_model_stats(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[Model Stats] Parameters: {total:,} | Trainable: {trainable:,}")

    def _project_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_projection(self.audio_adapter(audio))

    def _align_sequence(self, tensor, target_len: int, device: torch.device) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        tensor = tensor.to(device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        curr_len = tensor.shape[1]

        if curr_len > target_len:
            return tensor[:, :target_len]
        if curr_len < target_len:
            pad_len = target_len - curr_len
            if curr_len == 0:
                padding = torch.zeros(
                    tensor.shape[0], pad_len, tensor.shape[-1], device=device, dtype=tensor.dtype
                )
            else:
                padding = tensor[:, -1:, :].expand(-1, pad_len, -1)
            return torch.cat([tensor, padding], dim=1)
        return tensor

    def _project_condition_or_zero(
        self, tensor: torch.Tensor | None, projection: nn.Module, batch_size: int, target_len: int
    ) -> torch.Tensor:
        if tensor is None:
            out_dim = self.opt.dim_c
            return torch.zeros(batch_size, target_len, out_dim, device=self._device())
        projected = projection(tensor)
        if projected.shape[0] == 1 and batch_size > 1:
            projected = projected.expand(batch_size, -1, -1)
        return projected

    def forward(self, batch, t):
        x, prev_x = batch["m_now"], batch["m_prev"]
        a, prev_a = batch["a_now"], batch["a_prev"]
        m_ref = batch["m_ref"]

        gaze, prev_gaze = batch["gaze_now"], batch["gaze_prev"]
        pose, prev_pose = batch["pose_now"], batch["pose_prev"]
        cam, prev_cam = batch["cam_now"], batch["cam_prev"]

        bs = x.size(0)

        if not self.opt.only_last_features:
            a = a.view(bs, self.num_frames_for_clip, -1)
            prev_a = prev_a.view(bs, self.num_prev_frames, -1)

        a = self._project_audio(a)
        prev_a = self._project_audio(prev_a)
        gaze = self.gaze_projection(gaze)
        prev_gaze = self.gaze_projection(prev_gaze)
        pose = self.pose_projection(pose)
        prev_pose = self.pose_projection(prev_pose)
        cam = self.cam_projection(cam)
        prev_cam = self.cam_projection(prev_cam)

        pred = self.fmt(
            t,
            x,
            a,
            prev_x,
            prev_a,
            m_ref,
            gaze=gaze,
            prev_gaze=prev_gaze,
            pose=pose,
            prev_pose=prev_pose,
            cam=cam,
            prev_cam=prev_cam,
            train=self.training,
        )

        return pred[:, self.num_prev_frames :, ...]

    @torch.no_grad()
    def sample(
        self,
        data,
        a_cfg_scale: float = 1.0,
        nfe: int = 10,
        seed=None,
        stream_state=None,
        return_state: bool = False,
    ):
        ref_x = data["ref_x"].to(self._device())
        gaze_raw = data.get("gaze")
        pose_raw = data.get("pose")
        cam_raw = data.get("cam")

        device = self._device()
        time_steps = torch.linspace(0, 1, nfe, device=device)

        if "a_feat" in data:
            a_feat = data["a_feat"].to(device)
            if a_feat.ndim == 2:
                a_feat = a_feat.unsqueeze(0)
            B = a_feat.shape[0]
            T = a_feat.shape[1]
            a = self._project_audio(a_feat)
        else:
            a = data["a"].to(device)
            B = a.shape[0]
            T = math.ceil(a.shape[-1] * self.fps / self.opt.sampling_rate)
            a = self.audio_encoder.inference(a, seq_len=T)
            a = self._project_audio(a)

        gaze = self._align_sequence(gaze_raw, T, device)
        pose = self._align_sequence(pose_raw, T, device)
        cam = self._align_sequence(cam_raw, T, device)

        gaze = self._project_condition_or_zero(gaze, self.gaze_projection, B, T)
        pose = self._project_condition_or_zero(pose, self.pose_projection, B, T)
        cam = self._project_condition_or_zero(cam, self.cam_projection, B, T)

        sample = []
        num_chunks = int(math.ceil(T / self.num_frames_for_clip))
        prev_sample_t = None if stream_state is None else stream_state.get("prev_sample")
        prev_a_ctx = None if stream_state is None else stream_state.get("prev_a")
        prev_gaze_ctx = None if stream_state is None else stream_state.get("prev_gaze")
        prev_pose_ctx = None if stream_state is None else stream_state.get("prev_pose")
        prev_cam_ctx = None if stream_state is None else stream_state.get("prev_cam")
        prev_x0_ctx = None if stream_state is None else stream_state.get("prev_x0")
        debug_chunk_index = data.get("debug_chunk_index")

        def repeat_first_step(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            return tensor[:, :1, :].expand(-1, self.num_prev_frames, -1).contiguous()

        for chunk_idx in range(num_chunks):
            if self.opt.fix_noise_seed:
                current_seed = self.opt.seed if seed is None else seed
                generator = torch.Generator(device=device)
                generator.manual_seed(current_seed + chunk_idx)
                x0 = torch.randn(
                    B,
                    self.num_frames_for_clip,
                    self.opt.dim_w,
                    device=device,
                    generator=generator,
                )
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=device)

            carry_frames = 0
            if prev_x0_ctx is not None and prev_x0_ctx.numel() > 0:
                carry_frames = min(
                    prev_x0_ctx.shape[1],
                    x0.shape[1],
                    max(self.num_prev_frames, self.num_frames_for_clip // 2),
                )
                if carry_frames > 0:
                    x0[:, :carry_frames, :] = prev_x0_ctx[:, -carry_frames:, :].to(
                        device=x0.device, dtype=x0.dtype
                    )
            if getattr(self.opt, "debug_session", False) and debug_chunk_index is not None:
                print(
                    "[DBG/x0] "
                    f"idx={int(debug_chunk_index) + chunk_idx} "
                    f"carry_frames={carry_frames} "
                    f"head_norm={x0[:, :1, :].norm(dim=-1).mean().item():.4f}"
                )

            if prev_sample_t is None:
                # First live chunk should start from the reference motion latent,
                # not an all-zero latent that never appears during training.
                prev_x_t = repeat_first_step(ref_x.to(device))
                prev_a_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_c, device=device)
                prev_gaze_t = repeat_first_step(gaze)
                prev_pose_t = repeat_first_step(pose)
                prev_cam_t = repeat_first_step(cam)
            else:
                prev_x_t = prev_sample_t[:, -self.num_prev_frames :]
                prev_a_t = prev_a_ctx[:, -self.num_prev_frames :]
                prev_gaze_t = prev_gaze_ctx[:, -self.num_prev_frames :]
                prev_pose_t = prev_pose_ctx[:, -self.num_prev_frames :]
                prev_cam_t = prev_cam_ctx[:, -self.num_prev_frames :]

            start_idx = chunk_idx * self.num_frames_for_clip
            end_idx = (chunk_idx + 1) * self.num_frames_for_clip

            a_t = a[:, start_idx:end_idx]
            gaze_t = gaze[:, start_idx:end_idx]
            pose_t = pose[:, start_idx:end_idx]
            cam_t = cam[:, start_idx:end_idx]

            current_chunk_len = a_t.shape[1]
            if current_chunk_len < self.num_frames_for_clip:
                pad_len = self.num_frames_for_clip - current_chunk_len

                def pad_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    last = tensor[:, -1:, :].expand(-1, pad_len, -1)
                    return torch.cat([tensor, last], dim=1)

                a_t = pad_tensor(a_t)
                gaze_t = pad_tensor(gaze_t)
                pose_t = pad_tensor(pose_t)
                cam_t = pad_tensor(cam_t)

            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfg(
                    t=tt.unsqueeze(0),
                    x=zt,
                    a=a_t,
                    prev_x=prev_x_t,
                    prev_a=prev_a_t,
                    ref_x=ref_x,
                    gaze=gaze_t,
                    prev_gaze=prev_gaze_t,
                    pose=pose_t,
                    prev_pose=prev_pose_t,
                    cam=cam_t,
                    prev_cam=prev_cam_t,
                    a_cfg_scale=a_cfg_scale,
                )
                return out[:, self.num_prev_frames :]

            trajectory_t = odeint(sample_chunk, x0, time_steps, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)
            prev_sample_t = sample_t
            prev_a_ctx = a_t
            prev_gaze_ctx = gaze_t
            prev_pose_ctx = pose_t
            prev_cam_ctx = cam_t
            prev_x0_ctx = x0.detach()

        final_sample = torch.cat(sample, dim=1)[:, :T]
        if not return_state:
            return final_sample

        next_state = {
            "prev_sample": prev_sample_t[:, -self.num_prev_frames :].detach(),
            "prev_a": prev_a_ctx[:, -self.num_prev_frames :].detach(),
            "prev_gaze": prev_gaze_ctx[:, -self.num_prev_frames :].detach(),
            "prev_pose": prev_pose_ctx[:, -self.num_prev_frames :].detach(),
            "prev_cam": prev_cam_ctx[:, -self.num_prev_frames :].detach(),
            "prev_x0": prev_x0_ctx.detach(),
        }
        return final_sample, next_state


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

    def _pad_audio(self, a: torch.Tensor, target_frames: int) -> torch.Tensor:
        target_samples = int(target_frames * self.sampling_rate / self.fps)
        if a.shape[1] % target_samples != 0:
            diff = target_samples - (a.shape[1] % target_samples)
            a = F.pad(a, (0, diff), mode="replicate")
        return a

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        out = self.wav2vec2(a, seq_len=seq_len, output_hidden_states=not self.only_last_features)
        if self.only_last_features:
            return out.last_hidden_state
        feat = torch.stack(out.hidden_states[1:], dim=1)
        feat = feat.permute(0, 2, 1, 3)
        return feat.reshape(feat.shape[0], feat.shape[1], -1)

    def forward(self, a: torch.Tensor, prev_a: torch.Tensor | None = None) -> torch.Tensor:
        total_frames = self.num_frames_for_clip
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
            total_frames += self.num_prev_frames
        a = self._pad_audio(a, total_frames)
        return self.get_wav2vec2_feature(a, seq_len=total_frames)

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len: int) -> torch.Tensor:
        a = self._pad_audio(a, seq_len)
        return self.get_wav2vec2_feature(a, seq_len=seq_len)
