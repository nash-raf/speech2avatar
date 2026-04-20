from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

from launch_live import LaunchOptions
from live_pipeline import LiveMoshiIMTalkerSession


class ReplaySweepOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--aligned_npy", required=True, type=str)
        parser.add_argument("--wav_path", default=None, type=str)
        parser.add_argument("--meta_json", default=None, type=str)
        parser.add_argument("--sweep_output_dir", required=True, type=str)
        parser.add_argument(
            "--cfg_values",
            default="3.0,4.0,5.0",
            type=str,
            help="Comma-separated a_cfg_scale values to replay.",
        )
        parser.add_argument(
            "--gain_values",
            default="1.0,1.1,1.2",
            type=str,
            help="Comma-separated diagnostic latent gains.",
        )
        parser.add_argument(
            "--cfg_for_gain",
            default=None,
            type=float,
            help="CFG value whose sample should be reused for the gain sweep. Defaults to first cfg_values entry.",
        )
        parser.add_argument(
            "--tag",
            default="replay",
            type=str,
            help="Filename prefix for outputs.",
        )
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


def _parse_float_list(raw: str) -> list[float]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise RuntimeError("Expected at least one float value")
    return vals


def _save_video(vid_tensor: torch.Tensor, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)

    vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    vid = (vid * 255).round().to(torch.uint8)
    torchvision.io.write_video(str(temp_path), vid, fps=fps)
    os.replace(temp_path, out_path)


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


def _apply_latent_gain(sample: torch.Tensor, gain: float) -> torch.Tensor:
    center = sample.mean(dim=1, keepdim=True)
    return center + gain * (sample - center)


def main() -> None:
    opt = ReplaySweepOptions().parse()
    opt.rank = opt.device

    meta = _load_meta(opt.meta_json)
    aligned = _normalize_aligned_npy(opt.aligned_npy, feat_dim=int(opt.audio_feat_dim))
    cfg_values = _parse_float_list(opt.cfg_values)
    gain_values = _parse_float_list(opt.gain_values)
    gain_cfg = float(opt.cfg_for_gain) if opt.cfg_for_gain is not None else float(cfg_values[0])

    expected_frames = None
    if meta is not None:
        expected_frames = int(meta.get("aligned_concat_shape", [aligned.shape[0], aligned.shape[1]])[0])
        if expected_frames != aligned.shape[0]:
            print(
                f"[replay_sweep] warning: aligned_npy has {aligned.shape[0]} frames "
                f"but meta_json reports {expected_frames}"
            )

    out_dir = Path(opt.sweep_output_dir)
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

    summary: dict[str, object] = {
        "aligned_shape": list(aligned.shape),
        "meta_expected_frames": expected_frames,
        "cfg_values": cfg_values,
        "gain_values": gain_values,
        "gain_cfg": gain_cfg,
        "runs": [],
    }

    cached_samples: dict[float, torch.Tensor] = {}

    with torch.no_grad():
        for cfg in cfg_values:
            sample = session.fm.sample(
                {"ref_x": session.ref_x, "a_feat": aligned.to(session.device)},
                a_cfg_scale=cfg,
                nfe=opt.nfe,
                seed=opt.seed,
            )
            cached_samples[cfg] = sample.detach().clone()
            frames = session._decode_sample_to_frames(sample)

            silent_name = f"{opt.tag}_cfg{cfg:.1f}_silent.mp4"
            silent_path = out_dir / silent_name
            _save_video(frames, silent_path, fps=float(opt.fps))

            muxed_path = None
            if opt.wav_path:
                muxed_name = f"{opt.tag}_cfg{cfg:.1f}_muxed.mp4"
                muxed_path = out_dir / muxed_name
                _mux_audio(silent_path, opt.wav_path, muxed_path)

            stats = _latent_stats(sample)
            summary["runs"].append(
                {
                    "kind": "cfg",
                    "cfg": cfg,
                    "gain": 1.0,
                    "sample_shape": list(sample.shape),
                    "rendered_frames": int(frames.shape[0]),
                    "silent_mp4": str(silent_path),
                    "muxed_mp4": str(muxed_path) if muxed_path is not None else None,
                    **stats,
                }
            )
            print(
                f"[replay_sweep] cfg={cfg:.1f} rendered_frames={frames.shape[0]} "
                f"mean_norm={stats['sample_mean_norm']:.4f} "
                f"std_t={stats['sample_std_over_time']:.4f} "
                f"delta_mean={stats['frame_to_frame_delta_mean']:.4f}"
            )

        if gain_cfg not in cached_samples:
            raise RuntimeError(
                f"cfg_for_gain={gain_cfg} not found in cfg_values={cfg_values}"
            )

        base_sample = cached_samples[gain_cfg]
        for gain in gain_values:
            gained = _apply_latent_gain(base_sample, gain)
            frames = session._decode_sample_to_frames(gained)

            silent_name = f"{opt.tag}_cfg{gain_cfg:.1f}_gain{gain:.1f}_silent.mp4"
            silent_path = out_dir / silent_name
            _save_video(frames, silent_path, fps=float(opt.fps))

            muxed_path = None
            if opt.wav_path:
                muxed_name = f"{opt.tag}_cfg{gain_cfg:.1f}_gain{gain:.1f}_muxed.mp4"
                muxed_path = out_dir / muxed_name
                _mux_audio(silent_path, opt.wav_path, muxed_path)

            stats = _latent_stats(gained)
            summary["runs"].append(
                {
                    "kind": "gain",
                    "cfg": gain_cfg,
                    "gain": gain,
                    "sample_shape": list(gained.shape),
                    "rendered_frames": int(frames.shape[0]),
                    "silent_mp4": str(silent_path),
                    "muxed_mp4": str(muxed_path) if muxed_path is not None else None,
                    **stats,
                }
            )
            print(
                f"[replay_sweep] cfg={gain_cfg:.1f} gain={gain:.1f} rendered_frames={frames.shape[0]} "
                f"mean_norm={stats['sample_mean_norm']:.4f} "
                f"std_t={stats['sample_std_over_time']:.4f} "
                f"delta_mean={stats['frame_to_frame_delta_mean']:.4f}"
            )

    summary_path = out_dir / f"{opt.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[replay_sweep] summary={summary_path}")
    print(
        "[replay_sweep] path=aligned_npy -> FM.sample(a_feat=...) -> optional latent gain -> renderer "
        "(no wav2vec, no Mimi re-encode)"
    )


if __name__ == "__main__":
    main()
