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


class LatentSurgeryOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--aligned_npy", required=True, type=str)
        parser.add_argument("--original_latents_pt", required=True, type=str)
        parser.add_argument("--wav_path", default=None, type=str)
        parser.add_argument("--meta_json", default=None, type=str)
        parser.add_argument("--surgery_output_dir", required=True, type=str)
        parser.add_argument("--tag", default="latent_surgery", type=str)
        parser.add_argument("--gain_values", default="1.0,1.2,1.4,1.6", type=str)
        parser.add_argument("--blend_values", default="0.25,0.5", type=str)
        return parser


def _parse_float_list(raw: str) -> list[float]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise RuntimeError("Expected at least one float value")
    return vals


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


def _load_latents_pt(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("sample", "latents", "motion_latents"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise RuntimeError(f"Expected tensor latents in {path}, got {type(obj)}")
    if obj.ndim != 3:
        raise RuntimeError(f"Expected latent tensor [B, T, D], got {tuple(obj.shape)}")
    if obj.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1, got {tuple(obj.shape)}")
    return obj.float().contiguous()


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


def _latent_stats(sample: torch.Tensor) -> dict[str, float]:
    sample_cpu = sample.detach().cpu()
    per_frame_norm = sample_cpu.norm(dim=-1)
    delta = sample_cpu[:, 1:] - sample_cpu[:, :-1] if sample_cpu.shape[1] > 1 else None
    return {
        "sample_mean_norm": float(per_frame_norm.mean().item()),
        "sample_std_over_time": float(sample_cpu.std(dim=1).mean().item()),
        "frame_to_frame_delta_mean": float(
            delta.norm(dim=-1).mean().item() if delta is not None else 0.0
        ),
    }


def _apply_global_gain(sample: torch.Tensor, gain: float) -> torch.Tensor:
    center = sample.mean(dim=1, keepdim=True)
    return center + gain * (sample - center)


def _apply_variance_match(fm_sample: torch.Tensor, orig_sample: torch.Tensor) -> torch.Tensor:
    mu_f = fm_sample.mean(dim=1, keepdim=True)
    std_f = fm_sample.std(dim=1, keepdim=True).clamp_min(1e-6)
    std_o = orig_sample.std(dim=1, keepdim=True)
    return mu_f + (fm_sample - mu_f) * (std_o / std_f)


def _apply_blend(fm_sample: torch.Tensor, orig_sample: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * fm_sample + alpha * orig_sample


def _render_variant(
    session: LiveMoshiIMTalkerSession,
    sample: torch.Tensor,
    out_dir: Path,
    tag: str,
    fps: float,
    wav_path: str | None,
) -> dict[str, object]:
    frames = session._decode_sample_to_frames(sample)
    silent_path = out_dir / f"{tag}_silent.mp4"
    _save_video(frames, silent_path, fps=fps)

    muxed_path = None
    if wav_path:
        muxed_path = out_dir / f"{tag}_muxed.mp4"
        _mux_audio(silent_path, wav_path, muxed_path)

    stats = _latent_stats(sample)
    return {
        "sample_shape": list(sample.shape),
        "rendered_frames": int(frames.shape[0]),
        "silent_mp4": str(silent_path),
        "muxed_mp4": str(muxed_path) if muxed_path is not None else None,
        **stats,
    }


def main() -> None:
    opt = LatentSurgeryOptions().parse()
    opt.rank = opt.device

    meta = _load_meta(opt.meta_json)
    aligned = _normalize_aligned_npy(opt.aligned_npy, feat_dim=int(opt.audio_feat_dim))
    orig_latents = _load_latents_pt(opt.original_latents_pt)
    gain_values = _parse_float_list(opt.gain_values)
    blend_values = _parse_float_list(opt.blend_values)

    expected_frames = None
    if meta is not None:
        expected_frames = int(meta.get("aligned_concat_shape", [aligned.shape[0], aligned.shape[1]])[0])
        if expected_frames != aligned.shape[0]:
            print(
                f"[latent_surgery] warning: aligned_npy has {aligned.shape[0]} frames "
                f"but meta_json reports {expected_frames}"
            )

    out_dir = Path(opt.surgery_output_dir)
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

    with torch.no_grad():
        fm_sample = session.fm.sample(
            {"ref_x": session.ref_x, "a_feat": aligned.to(session.device)},
            a_cfg_scale=opt.a_cfg_scale,
            nfe=opt.nfe,
            seed=opt.seed,
        ).detach()

    if fm_sample.shape != orig_latents.shape:
        raise RuntimeError(
            f"FM sample shape {tuple(fm_sample.shape)} does not match original latent shape {tuple(orig_latents.shape)}"
        )

    orig_latents = orig_latents.to(session.device)

    fm_stats = _latent_stats(fm_sample)
    orig_stats = _latent_stats(orig_latents)
    delta_match_gain = orig_stats["frame_to_frame_delta_mean"] / max(
        fm_stats["frame_to_frame_delta_mean"], 1e-6
    )

    summary: dict[str, object] = {
        "aligned_shape": list(aligned.shape),
        "meta_expected_frames": expected_frames,
        "original_latent_shape": list(orig_latents.shape),
        "fm_baseline": fm_stats,
        "original_reference": orig_stats,
        "delta_match_gain": delta_match_gain,
        "runs": [],
    }

    fm_latents_path = out_dir / f"{opt.tag}_fm_baseline_latents.pt"
    torch.save(fm_sample.detach().cpu(), fm_latents_path)

    baseline = _render_variant(
        session=session,
        sample=fm_sample,
        out_dir=out_dir,
        tag=f"{opt.tag}_fm_baseline",
        fps=float(opt.fps),
        wav_path=opt.wav_path,
    )
    summary["runs"].append({"kind": "fm_baseline", **baseline})
    print(
        f"[latent_surgery] fm_baseline mean_norm={baseline['sample_mean_norm']:.4f} "
        f"std_t={baseline['sample_std_over_time']:.4f} "
        f"delta_mean={baseline['frame_to_frame_delta_mean']:.4f}"
    )

    for gain in gain_values:
        edited = _apply_global_gain(fm_sample, gain)
        result = _render_variant(
            session=session,
            sample=edited,
            out_dir=out_dir,
            tag=f"{opt.tag}_global_gain{gain:.1f}",
            fps=float(opt.fps),
            wav_path=opt.wav_path,
        )
        summary["runs"].append({"kind": "global_gain", "gain": gain, **result})
        print(
            f"[latent_surgery] global_gain={gain:.1f} mean_norm={result['sample_mean_norm']:.4f} "
            f"std_t={result['sample_std_over_time']:.4f} "
            f"delta_mean={result['frame_to_frame_delta_mean']:.4f}"
        )

    variance_match = _apply_variance_match(fm_sample, orig_latents)
    result = _render_variant(
        session=session,
        sample=variance_match,
        out_dir=out_dir,
        tag=f"{opt.tag}_variance_match",
        fps=float(opt.fps),
        wav_path=opt.wav_path,
    )
    summary["runs"].append({"kind": "variance_match_to_original", **result})
    print(
        f"[latent_surgery] variance_match mean_norm={result['sample_mean_norm']:.4f} "
        f"std_t={result['sample_std_over_time']:.4f} "
        f"delta_mean={result['frame_to_frame_delta_mean']:.4f}"
    )

    delta_match = _apply_global_gain(fm_sample, delta_match_gain)
    result = _render_variant(
        session=session,
        sample=delta_match,
        out_dir=out_dir,
        tag=f"{opt.tag}_delta_match",
        fps=float(opt.fps),
        wav_path=opt.wav_path,
    )
    summary["runs"].append(
        {"kind": "delta_match_to_original", "gain": delta_match_gain, **result}
    )
    print(
        f"[latent_surgery] delta_match gain={delta_match_gain:.4f} "
        f"mean_norm={result['sample_mean_norm']:.4f} "
        f"std_t={result['sample_std_over_time']:.4f} "
        f"delta_mean={result['frame_to_frame_delta_mean']:.4f}"
    )

    for alpha in blend_values:
        edited = _apply_blend(fm_sample, orig_latents, alpha)
        result = _render_variant(
            session=session,
            sample=edited,
            out_dir=out_dir,
            tag=f"{opt.tag}_blend{alpha:.2f}",
            fps=float(opt.fps),
            wav_path=opt.wav_path,
        )
        summary["runs"].append({"kind": "blend_to_original", "alpha": alpha, **result})
        print(
            f"[latent_surgery] blend alpha={alpha:.2f} mean_norm={result['sample_mean_norm']:.4f} "
            f"std_t={result['sample_std_over_time']:.4f} "
            f"delta_mean={result['frame_to_frame_delta_mean']:.4f}"
        )

    summary_path = out_dir / f"{opt.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[latent_surgery] summary={summary_path}")
    print(
        "[latent_surgery] path=aligned_npy -> FM.sample(a_feat=...) -> latent surgery -> renderer "
        "(no wav2vec, no Mimi re-encode)"
    )


if __name__ == "__main__":
    main()
