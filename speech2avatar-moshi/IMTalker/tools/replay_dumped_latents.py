from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torchvision

from launch_live import LaunchOptions
from live_pipeline import LiveMoshiIMTalkerSession


class ReplayOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--latents_path", required=True, type=str)
        parser.add_argument("--out_path", required=True, type=str)
        parser.add_argument(
            "--audio_path",
            default=None,
            type=str,
            help="Optional audio file to mux back into the rendered mp4.",
        )
        return parser


def _normalize_a_feat(payload, feat_dim: int) -> torch.Tensor:
    latents = payload["latents"] if isinstance(payload, dict) else payload
    if isinstance(latents, list):
        latents = torch.cat([chunk.detach().cpu() for chunk in latents], dim=-1)
    latents = torch.as_tensor(latents).detach().cpu().float()

    # Expected FM.sample(a_feat=...) layout is [T, C] or [1, T, C].
    if latents.ndim == 3:
        if latents.shape[0] == 1 and latents.shape[1] == feat_dim:
            latents = latents[0].transpose(0, 1).contiguous()
        elif latents.shape[0] == 1 and latents.shape[2] == feat_dim:
            latents = latents[0].contiguous()
        else:
            raise RuntimeError(f"Unsupported latent shape {tuple(latents.shape)}")
    elif latents.ndim == 2:
        if latents.shape[1] == feat_dim:
            latents = latents.contiguous()
        elif latents.shape[0] == feat_dim:
            latents = latents.transpose(0, 1).contiguous()
        else:
            raise RuntimeError(f"Unsupported latent shape {tuple(latents.shape)}")
    else:
        raise RuntimeError(f"Unsupported latent rank {latents.ndim} for shape {tuple(latents.shape)}")

    if latents.shape[1] != feat_dim:
        raise RuntimeError(
            f"Expected feature dim {feat_dim}, got shape {tuple(latents.shape)}"
        )
    return latents


def _save_video(vid_tensor: torch.Tensor, out_path: Path, audio_path: str | None, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)

    vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    vid = (vid * 255).round().to(torch.uint8)
    torchvision.io.write_video(str(temp_path), vid, fps=fps)

    try:
        if audio_path:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_path),
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_path.unlink(missing_ok=True)
        else:
            os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    opt = ReplayOptions().parse()
    opt.rank = opt.device

    session = LiveMoshiIMTalkerSession(
        opt,
        generator_path=opt.generator_path,
        renderer_path=opt.renderer_path,
        ref_path=opt.ref_path,
        crop=opt.crop,
        nfe=opt.nfe,
        a_cfg_scale=opt.a_cfg_scale,
        moshi_repo=opt.moshi_repo,
        mimi_hf_repo=opt.hf_repo,
    )

    payload = torch.load(opt.latents_path, map_location="cpu")
    a_feat = _normalize_a_feat(payload, int(opt.audio_feat_dim))

    with torch.no_grad():
        sample = session.fm.sample(
            {"ref_x": session.ref_x, "a_feat": a_feat.to(session.device)},
            a_cfg_scale=opt.a_cfg_scale,
            nfe=opt.nfe,
            seed=opt.seed,
        )
        frames = session._decode_sample_to_frames(sample)

    out_path = Path(opt.out_path)
    _save_video(frames, out_path, opt.audio_path, float(opt.fps))

    print(
        f"[replay_dumped_latents] saved={out_path} "
        f"latent_T={a_feat.shape[0]} feat_dim={a_feat.shape[1]} "
        f"audio={'yes' if opt.audio_path else 'no'}"
    )


if __name__ == "__main__":
    main()
