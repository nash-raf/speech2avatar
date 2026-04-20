from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

from launch_live import LaunchOptions
from live_pipeline import LiveMoshiIMTalkerSession


class ReplayLiveConditioningOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--aligned_npy", required=True, type=str)
        parser.add_argument("--wav_path", default=None, type=str)
        parser.add_argument("--meta_json", default=None, type=str)
        parser.add_argument("--pose_pt", default=None, type=str)
        parser.add_argument("--gaze_npy", default=None, type=str)
        parser.add_argument("--output_mp4", required=True, type=str)
        parser.add_argument("--output_muxed_mp4", default=None, type=str)
        return parser


def _load_meta(path: str | None) -> dict | None:
    if not path:
        return None
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_json not found: {meta_path}")
    return json.loads(meta_path.read_text())


def _normalize_aligned_npy(path: str, feat_dim: int) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected aligned conditioning to have rank 2 [T, C], got shape {tuple(arr.shape)}"
        )
    if arr.shape[1] != feat_dim:
        raise RuntimeError(
            f"Expected aligned conditioning feature dim {feat_dim}, got shape {tuple(arr.shape)}"
        )
    return torch.from_numpy(arr.astype(np.float32, copy=False)).contiguous()


def _save_video(vid_tensor: torch.Tensor, out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    vid = (vid * 255).round().to(torch.uint8)
    torchvision.io.write_video(str(out_path), vid, fps=fps)


def _load_smirk_params(smirk_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pose = smirk_data["pose_params"].float()
    cam = smirk_data["cam"].float()
    return pose, cam


def _load_pose_cam(path: str | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path:
        return None, None
    pose_path = Path(path)
    if not pose_path.exists():
        raise FileNotFoundError(f"pose_pt not found: {pose_path}")
    return _load_smirk_params(torch.load(pose_path, map_location="cpu"))


def _load_gaze(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    gaze_path = Path(path)
    if not gaze_path.exists():
        raise FileNotFoundError(f"gaze_npy not found: {gaze_path}")
    return torch.tensor(np.load(gaze_path), dtype=torch.float32)


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


def main() -> None:
    opt = ReplayLiveConditioningOptions().parse()
    opt.rank = opt.device

    meta = _load_meta(opt.meta_json)
    aligned = _normalize_aligned_npy(opt.aligned_npy, feat_dim=int(opt.audio_feat_dim))

    expected_frames = None
    if meta is not None:
        expected_frames = int(meta.get("aligned_concat_shape", [aligned.shape[0], aligned.shape[1]])[0])
        if expected_frames != aligned.shape[0]:
            print(
                f"[replay_live_conditioning] warning: aligned_npy has {aligned.shape[0]} frames "
                f"but meta_json reports {expected_frames}"
            )

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
    pose, cam = _load_pose_cam(opt.pose_pt)
    gaze = _load_gaze(opt.gaze_npy)

    with torch.no_grad():
        sample = session.fm.sample(
            {
                "ref_x": session.ref_x,
                "a_feat": aligned.to(session.device),
                "pose": None if pose is None else pose.to(session.device),
                "cam": None if cam is None else cam.to(session.device),
                "gaze": None if gaze is None else gaze.to(session.device),
            },
            a_cfg_scale=opt.a_cfg_scale,
            nfe=opt.nfe,
            seed=opt.seed,
        )
        frames = session._decode_sample_to_frames(sample)

    output_mp4 = Path(opt.output_mp4)
    _save_video(frames, output_mp4, fps=float(opt.fps))

    if opt.output_muxed_mp4:
        if not opt.wav_path:
            raise RuntimeError("--output_muxed_mp4 requires --wav_path")
        _mux_audio(output_mp4, opt.wav_path, Path(opt.output_muxed_mp4))

    print(
        "[replay_live_conditioning] "
        f"aligned_shape={tuple(aligned.shape)} "
        f"meta_expected_frames={expected_frames if expected_frames is not None else 'n/a'} "
        f"pose={'yes' if pose is not None else 'no'} "
        f"cam={'yes' if cam is not None else 'no'} "
        f"gaze={'yes' if gaze is not None else 'no'} "
        f"sample_shape={tuple(sample.shape)} "
        f"rendered_frames={frames.shape[0]} "
        f"fps={opt.fps}"
    )
    print(f"[replay_live_conditioning] silent_mp4={output_mp4}")
    if opt.output_muxed_mp4:
        print(f"[replay_live_conditioning] muxed_mp4={opt.output_muxed_mp4}")
    print(
        "[replay_live_conditioning] path=aligned_npy -> FM.sample(a_feat=...) -> renderer "
        "(no wav2vec, no Mimi re-encode, no Moshi tokens)"
    )


if __name__ == "__main__":
    main()
