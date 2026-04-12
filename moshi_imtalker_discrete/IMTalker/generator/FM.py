import math
import torch
import torch.nn as nn
from torchdiffeq import odeint

from generator.wav2vec2 import MoshiTokenEncoder
from generator.FMT import FlowMatchingTransformer


class FMGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fps = opt.fps
        self.rank = opt.rank
        
        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames
        self.audio_input_dim = opt.audio_feat_dim
        
        self.audio_encoder = MoshiTokenEncoder(opt)
        self.fmt = FlowMatchingTransformer(opt)
        
        self.audio_projection = self._make_projection(self.audio_input_dim, opt.dim_c)
        self.gaze_projection = self._make_projection(2, opt.dim_c)
        self.pose_projection = self._make_projection(3, opt.dim_c)
        self.cam_projection = self._make_projection(3, opt.dim_c)

        self.odeint_kwargs = {
            'atol': opt.ode_atol,
            'rtol': opt.ode_rtol,
            'method': opt.torchdiffeq_ode_method
        }

        if getattr(opt, "freeze_non_audio", False):
            self._freeze_non_audio()
        
        self._print_model_stats()

    def _make_projection(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def _print_model_stats(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[Model Stats] Parameters: {total:,} | Trainable: {trainable:,}")

    def _freeze_non_audio(self):
        for param in self.parameters():
            param.requires_grad = False

        for module in (self.audio_encoder, self.audio_projection):
            for param in module.parameters():
                param.requires_grad = True

    def _get_batch_tensor(self, batch, key, fallback_key=None):
        if key in batch:
            return batch[key]
        if fallback_key is not None and fallback_key in batch:
            return batch[fallback_key]
        raise KeyError(f"Missing required batch key: {key}")

    def forward(self, batch, t):
        x, prev_x = batch["m_now"], batch["m_prev"]
        m_ref = batch["m_ref"]

        a = self.audio_encoder(
            batch["a_now"],
            seq_len=self.num_frames_for_clip,
            token_len=batch.get("a_now_len"),
        )
        prev_a = self.audio_encoder(
            batch["a_prev"],
            seq_len=self.num_prev_frames,
            token_len=batch.get("a_prev_len"),
        )

        gaze = self._get_batch_tensor(batch, "gaze", fallback_key="gaze_now")
        prev_gaze = batch["gaze_prev"]
        pose = self._get_batch_tensor(batch, "pose", fallback_key="pose_now")
        prev_pose = batch["pose_prev"]
        cam = self._get_batch_tensor(batch, "cam", fallback_key="cam_now")
        prev_cam = batch["cam_prev"]

        # Joint per-sample aux dropout (training-only).
        # With prob p, zero all of {gaze, pose, cam} (and their prev counterparts)
        # simultaneously for that sample, so the audio adapter learns to handle
        # the "aux fully absent" regime as an in-distribution mode.
        p = self.opt.aux_joint_dropout_prob
        if self.training and p > 0.0:
            B = gaze.shape[0]
            drop_mask = (
                torch.rand(B, device=gaze.device) < p
            )  # shape [B], True = drop this sample's aux
            if drop_mask.any():
                # Broadcast [B] mask to [B, 1, 1] for (B, T, C) tensors
                keep = (~drop_mask).to(gaze.dtype).view(B, 1, 1)
                gaze = gaze * keep
                prev_gaze = prev_gaze * keep
                pose = pose * keep
                prev_pose = prev_pose * keep
                cam = cam * keep
                prev_cam = prev_cam * keep

        a = self.audio_projection(a)
        prev_a = self.audio_projection(prev_a)
        gaze = self.gaze_projection(gaze)
        prev_gaze = self.gaze_projection(prev_gaze)
        pose = self.pose_projection(pose)
        prev_pose = self.pose_projection(prev_pose)
        cam = self.cam_projection(cam)
        prev_cam = self.cam_projection(prev_cam)

        pred = self.fmt(
            t, x, a, prev_x, prev_a, m_ref,
            gaze=gaze, prev_gaze=prev_gaze,
            pose=pose, prev_pose=prev_pose,
            cam=cam, prev_cam=prev_cam
        )

        return pred[:, self.num_prev_frames:, ...]

    def _align_sequence(self, tensor, target_len):
        """Helper to crop or pad sequences to target length."""
        if tensor is None:
            return None
            
        tensor = tensor.to(self.rank)
        curr_len = tensor.shape[0]
        
        if curr_len > target_len:
            return tensor[:target_len]
        elif curr_len < target_len:
            pad_len = target_len - curr_len
            padding = torch.zeros(pad_len, tensor.shape[1], device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        return tensor

    def _ensure_prev_context(self, tensor, batch_size, feat_dim, device):
        if tensor is None:
            return torch.zeros(batch_size, self.num_prev_frames, feat_dim, device=device)
        if tensor.shape[1] > self.num_prev_frames:
            return tensor[:, -self.num_prev_frames:]
        if tensor.shape[1] < self.num_prev_frames:
            pad_len = self.num_prev_frames - tensor.shape[1]
            padding = torch.zeros(
                batch_size, pad_len, feat_dim, device=device, dtype=tensor.dtype
            )
            return torch.cat([padding, tensor], dim=1)
        return tensor

    @torch.no_grad()
    def sample(self, data, a_cfg_scale=1.0, nfe=10, seed=None, stream_state=None, return_state=False):
        ref_x = data['ref_x']
        gaze_raw = data.get('gaze')
        pose_raw = data.get('pose')
        cam_raw = data.get('cam')

        device = self.rank
        time_steps = torch.linspace(0, 1, nfe, device=device)

        a_feat = data.get('a_feat')
        if a_feat is not None:
            # Live path: caller already produced video-rate audio features
            # (e.g. by running MoshiTokenEncoder externally on streaming
            # codebook-0 token IDs). Skip audio_encoder; just project.
            a = a_feat.to(device)
            if a.dim() == 2:
                a = a.unsqueeze(0)
            B = a.shape[0]
            T = a.shape[1]
        else:
            a = data['a'].to(device)
            B = a.shape[0]
            T = math.ceil(a.shape[1] * self.fps / self.opt.audio_token_rate)
            a = self.audio_encoder.inference(a, seq_len=T)
        a = self.audio_projection(a)

        gaze = self._align_sequence(gaze_raw, T)
        pose = self._align_sequence(pose_raw, T)
        cam = self._align_sequence(cam_raw, T)

        if gaze is None:
            gaze = torch.zeros(B, T, self.opt.dim_c, device=device)
        else:
            gaze = self.gaze_projection(gaze).unsqueeze(0)

        if pose is None:
            pose = torch.zeros(B, T, self.opt.dim_c, device=device)
        else:
            pose = self.pose_projection(pose).unsqueeze(0)

        if cam is None:
            cam = torch.zeros(B, T, self.opt.dim_c, device=device)
        else:
            cam = self.cam_projection(cam).unsqueeze(0)

        sample = []
        num_chunks = int(math.ceil(T / self.num_frames_for_clip))
        next_stream_state = None

        for t in range(num_chunks):
            if self.opt.fix_noise_seed:
                current_seed = self.opt.seed if seed is None else seed
                g = torch.Generator(device)
                g.manual_seed(current_seed)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=device, generator=g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=device)

            if t == 0:
                prev_x_t = self._ensure_prev_context(
                    stream_state.get("prev_x") if stream_state else None,
                    B,
                    self.opt.dim_w,
                    device,
                )
                prev_a_t = self._ensure_prev_context(
                    stream_state.get("prev_a") if stream_state else None,
                    B,
                    a.shape[-1],
                    device,
                )
                prev_gaze_t = self._ensure_prev_context(
                    stream_state.get("prev_gaze") if stream_state else None,
                    B,
                    gaze.shape[-1],
                    device,
                )
                prev_pose_t = self._ensure_prev_context(
                    stream_state.get("prev_pose") if stream_state else None,
                    B,
                    pose.shape[-1],
                    device,
                )
                prev_cam_t = self._ensure_prev_context(
                    stream_state.get("prev_cam") if stream_state else None,
                    B,
                    cam.shape[-1],
                    device,
                )
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames:]
                prev_a_t = a_t[:, -self.num_prev_frames:]
                prev_gaze_t = gaze_t[:, -self.num_prev_frames:]
                prev_pose_t = pose_t[:, -self.num_prev_frames:]
                prev_cam_t = cam_t[:, -self.num_prev_frames:]

            start_idx = t * self.num_frames_for_clip
            end_idx = (t + 1) * self.num_frames_for_clip
            
            a_t = a[:, start_idx:end_idx]
            gaze_t = gaze[:, start_idx:end_idx]
            pose_t = pose[:, start_idx:end_idx]
            cam_t = cam[:, start_idx:end_idx]

            current_chunk_len = a_t.shape[1]
            if current_chunk_len < self.num_frames_for_clip:
                pad_len = self.num_frames_for_clip - current_chunk_len
                
                def pad_tensor(tensor):
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
                    gaze=gaze_t, prev_gaze=prev_gaze_t,
                    pose=pose_t, prev_pose=prev_pose_t,
                    cam=cam_t, prev_cam=prev_cam_t,
                    a_cfg_scale=a_cfg_scale,
                )
                return out[:, self.num_prev_frames:]

            trajectory_t = odeint(sample_chunk, x0, time_steps, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)
            next_stream_state = {
                "prev_x": sample_t[:, -self.num_prev_frames:].detach(),
                "prev_a": a_t[:, -self.num_prev_frames:].detach(),
                "prev_gaze": gaze_t[:, -self.num_prev_frames:].detach(),
                "prev_pose": pose_t[:, -self.num_prev_frames:].detach(),
                "prev_cam": cam_t[:, -self.num_prev_frames:].detach(),
            }

        sample = torch.cat(sample, dim=1)[:, :T]
        if return_state:
            return sample, next_stream_state
        return sample

    def make_initial_stream_state(self, ref_x, dim_c=None):
        """Zero state for the first streaming chunk."""
        device = self.rank
        B = ref_x.shape[0]
        d = self.opt.dim_c if dim_c is None else dim_c
        return {
            "prev_x":    torch.zeros(B, self.num_prev_frames, self.opt.dim_w, device=device),
            "prev_a":    torch.zeros(B, self.num_prev_frames, d,              device=device),
            "prev_gaze": torch.zeros(B, self.num_prev_frames, d,              device=device),
            "prev_pose": torch.zeros(B, self.num_prev_frames, d,              device=device),
            "prev_cam":  torch.zeros(B, self.num_prev_frames, d,              device=device),
        }
