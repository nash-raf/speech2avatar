from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

from launch_live import LaunchOptions
from live_pipeline import LiveMoshiIMTalkerSession


class CompareBridgeOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--aligned_npy", required=True, type=str)
        parser.add_argument("--moshi_latents_pt", required=True, type=str)
        parser.add_argument("--meta_json", default=None, type=str)
        parser.add_argument("--wav_path", default=None, type=str)
        parser.add_argument("--bridge_output_dir", required=True, type=str)
        parser.add_argument("--tag", default="bridge_ab", type=str)
        parser.add_argument("--ridge_lambda", default=1.0, type=float)
        return parser


def _load_meta(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"meta_json not found: {p}")
    return json.loads(p.read_text())


def _normalize_aligned_npy(path: str, feat_dim: int) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected [T, C] conditioning, got shape {tuple(arr.shape)}")
    if arr.shape[1] != feat_dim:
        raise RuntimeError(
            f"Expected conditioning feature dim {feat_dim}, got shape {tuple(arr.shape)}"
        )
    return torch.from_numpy(arr.astype(np.float32, copy=False)).contiguous()


def _load_moshi_raw_latents(path: str, feat_dim: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    latents = payload["latents"] if isinstance(payload, dict) else payload
    if isinstance(latents, list):
        latents = torch.cat([chunk.detach().cpu() for chunk in latents], dim=-1)
    latents = torch.as_tensor(latents).detach().cpu().float()

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
        raise RuntimeError(f"Unsupported latent rank {latents.ndim}")

    if latents.shape[1] != feat_dim:
        raise RuntimeError(f"Expected feature dim {feat_dim}, got {tuple(latents.shape)}")
    return latents


def _latent_stats(seq: torch.Tensor) -> dict[str, float]:
    seq = seq.detach().cpu()
    deltas = seq[1:] - seq[:-1] if seq.shape[0] > 1 else None
    return {
        "T": int(seq.shape[0]),
        "C": int(seq.shape[1]),
        "mean_norm": float(seq.norm(dim=-1).mean().item()),
        "std_over_time": float(seq.std(dim=0).mean().item()),
        "delta_mean": float(deltas.norm(dim=-1).mean().item() if deltas is not None else 0.0),
        "global_mean": float(seq.mean().item()),
        "global_std": float(seq.std().item()),
    }


def _save_video(vid_tensor: torch.Tensor, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)
    vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    vid = (vid * 255).round().to(torch.uint8)
    torchvision.io.write_video(str(temp_path), vid, fps=fps)
    temp_path.replace(out_path)


def _mux_audio(video_path: Path, audio_path: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _render_variant(
    session: LiveMoshiIMTalkerSession,
    conditioning: torch.Tensor,
    out_dir: Path,
    tag: str,
    wav_path: str | None,
    fps: float,
    a_cfg_scale: float,
    nfe: int,
    seed: int,
) -> dict[str, object]:
    with torch.no_grad():
        sample = session.fm.sample(
            {"ref_x": session.ref_x, "a_feat": conditioning.to(session.device)},
            a_cfg_scale=a_cfg_scale,
            nfe=nfe,
            seed=seed,
        )
        frames = session._decode_sample_to_frames(sample)

    silent_path = out_dir / f"{tag}_silent.mp4"
    _save_video(frames, silent_path, fps=fps)
    muxed_path = None
    if wav_path:
        muxed_path = out_dir / f"{tag}_muxed.mp4"
        _mux_audio(silent_path, wav_path, muxed_path)

    return {
        "conditioning_stats": _latent_stats(conditioning),
        "fm_sample_stats": _latent_stats(sample.squeeze(0)),
        "sample_shape": list(sample.shape),
        "rendered_frames": int(frames.shape[0]),
        "silent_mp4": str(silent_path),
        "muxed_mp4": str(muxed_path) if muxed_path is not None else None,
    }


def _match_distribution(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_s = source.mean(dim=0, keepdim=True)
    std_s = source.std(dim=0, keepdim=True).clamp_min(1e-6)
    mu_t = target.mean(dim=0, keepdim=True)
    std_t = target.std(dim=0, keepdim=True)
    return mu_t + (source - mu_s) * (std_t / std_s)


def _fit_ridge_affine(source: torch.Tensor, target: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    x = source.float()
    y = target.float()
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)  # [T, C+1]
    xtx = x_aug.T @ x_aug
    reg = torch.eye(xtx.shape[0], dtype=xtx.dtype) * ridge_lambda
    reg[-1, -1] = 0.0  # do not regularize bias
    xty = x_aug.T @ y
    w = torch.linalg.solve(xtx + reg, xty)  # [C+1, C]
    return x_aug @ w


def main() -> None:
    opt = CompareBridgeOptions().parse()
    opt.rank = opt.device

    meta = _load_meta(opt.meta_json)
    aligned_current = _normalize_aligned_npy(opt.aligned_npy, feat_dim=int(opt.audio_feat_dim))
    raw_direct = _load_moshi_raw_latents(opt.moshi_latents_pt, feat_dim=int(opt.audio_feat_dim))

    target_frames = aligned_current.shape[0]
    if meta is not None:
        expected_frames = int(meta.get("aligned_concat_shape", [target_frames, aligned_current.shape[1]])[0])
        if expected_frames != target_frames:
            print(
                f"[compare_bridge_ab] warning: aligned_npy has {target_frames} frames "
                f"but meta_json reports {expected_frames}"
            )

    out_dir = Path(opt.bridge_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    aligned_direct = session._align_raw_mimi_to_target_frames(raw_direct, target_frames)
    aligned_direct_normmatch = _match_distribution(aligned_direct, aligned_current)
    aligned_direct_ridgefit = _fit_ridge_affine(
        aligned_direct, aligned_current, ridge_lambda=float(opt.ridge_lambda)
    )

    summary: dict[str, object] = {
        "target_frames": target_frames,
        "current_aligned_shape": list(aligned_current.shape),
        "direct_raw_shape": list(raw_direct.shape),
        "direct_aligned_shape": list(aligned_direct.shape),
        "ridge_lambda": float(opt.ridge_lambda),
        "variants": {},
    }

    variants = {
        "current_pcm_mimi_reencode": aligned_current,
        "direct_moshi_latent": aligned_direct,
        "direct_moshi_latent_normmatch": aligned_direct_normmatch,
        "direct_moshi_latent_ridgefit": aligned_direct_ridgefit,
    }

    for name, conditioning in variants.items():
        result = _render_variant(
            session=session,
            conditioning=conditioning,
            out_dir=out_dir,
            tag=f"{opt.tag}_{name}",
            wav_path=opt.wav_path,
            fps=float(opt.fps),
            a_cfg_scale=float(opt.a_cfg_scale),
            nfe=int(opt.nfe),
            seed=int(opt.seed),
        )
        result["feature_mse_vs_current"] = float(
            torch.mean((conditioning.detach().cpu() - aligned_current.detach().cpu()) ** 2).item()
        )
        summary["variants"][name] = result
        print(
            f"[compare_bridge_ab] {name} | "
            f"cond_delta={result['conditioning_stats']['delta_mean']:.4f} | "
            f"fm_delta={result['fm_sample_stats']['delta_mean']:.4f} | "
            f"fm_std={result['fm_sample_stats']['std_over_time']:.4f} | "
            f"feat_mse={result['feature_mse_vs_current']:.6f}"
        )

    summary_path = out_dir / f"{opt.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[compare_bridge_ab] summary={summary_path}")


if __name__ == "__main__":
    main()
