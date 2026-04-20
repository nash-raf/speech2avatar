#!/usr/bin/env python3
"""Teacher-student diagnostics for IMTalker distillation runs.

Outputs:
- per-block hidden-state divergence plot
- velocity PSD comparison plot
- per-dim velocity histogram overlay plot + variance ratio table
- L_distill by t-bin table/plot
- CFG direction cosine similarity table/plot
- optional module-swap render triplet + merged comparison
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from generator.dataset import AudioMotionSmirkGazeDataset
from generator.FM import FMGenerator
from generator.options.base_options import BaseOptions
from live_pipeline import LiveMoshiIMTalkerSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose teacher-student IMTalker runs.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--student_ckpt", required=True, type=str)
    parser.add_argument("--teacher_ckpt", required=True, type=str)
    parser.add_argument("--renderer_path", required=True, type=str)
    parser.add_argument("--wav2vec_model_path", required=True, type=str)
    parser.add_argument("--teacher_audio_subdir", default="audio_wav2vec", type=str)
    parser.add_argument("--student_audio_subdir", default="audio_rt_aligned", type=str)
    parser.add_argument("--student_audio_feat_dim", default=512, type=int)
    parser.add_argument("--teacher_audio_feat_dim", default=768, type=int)
    parser.add_argument("--split", default="test.txt", type=str)
    parser.add_argument("--num_batches", default=20, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    parser.add_argument("--report_root", default="/workspace/diagnose", type=str)
    parser.add_argument("--title", default=None, type=str)
    parser.add_argument("--cfg_scale", default=2.0, type=float)
    parser.add_argument("--num_t_bins", default=5, type=int)
    parser.add_argument("--swap_ref_path", default=None, type=str)
    parser.add_argument("--swap_wav_path", default=None, type=str)
    parser.add_argument("--swap_crop", action="store_true")
    parser.add_argument("--swap_nfe", default=5, type=int)
    parser.add_argument("--moshi_repo", default="/workspace/moshi", type=str)
    parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16", type=str)
    parser.add_argument("--skip_swap_render", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_base_opt(**overrides) -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    opt = BaseOptions().initialize(parser).parse_args([])
    for key, value in overrides.items():
        setattr(opt, key, value)
    return opt


def _select_best_state_dict(checkpoint: dict, model_state_dict: dict) -> dict:
    raw_state = checkpoint.get("state_dict", checkpoint)
    if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]

    if not isinstance(raw_state, dict):
        raise RuntimeError(f"Unsupported checkpoint structure: {type(raw_state)!r}")

    candidates = [raw_state]
    for prefix in ("model.", "student.", "teacher."):
        stripped = {k[len(prefix):]: v for k, v in raw_state.items() if k.startswith(prefix)}
        if stripped:
            candidates.append(stripped)

    best_state = None
    best_match_count = -1
    for candidate in candidates:
        match_count = sum(
            1
            for k, v in candidate.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        )
        if match_count > best_match_count:
            best_match_count = match_count
            best_state = candidate

    if best_state is None or best_match_count <= 0:
        raise RuntimeError("Could not find a compatible generator state_dict in checkpoint.")
    return best_state


def _merge_ema_shadow(checkpoint: dict, full_state: dict, model_state_dict: dict, use_ema: bool) -> tuple[dict, int]:
    if not use_ema:
        return full_state, 0
    ema_state = checkpoint.get("ema_state_dict")
    if not isinstance(ema_state, dict):
        return full_state, 0

    merged = dict(full_state)
    applied = 0
    for key, value in ema_state.items():
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            merged[key] = value
            applied += 1
    return merged, applied


def load_generator_checkpoint(model: FMGenerator, ckpt_path: str, use_ema: bool) -> dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_state = model.state_dict()
    state_dict = _select_best_state_dict(checkpoint, model_state)
    state_dict, ema_applied = _merge_ema_shadow(checkpoint, state_dict, model_state, use_ema)
    loadable = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint": ckpt_path,
        "loaded_params": len(loadable),
        "use_ema": bool(use_ema),
        "ema_params_merged": ema_applied,
        "c_embedder_fingerprint": module_state_fingerprint(model.fmt.c_embedder),
    }


def module_state_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def move_batch_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def make_model_batch(batch: dict, m_now: torch.Tensor, a_now: torch.Tensor, a_prev: torch.Tensor) -> dict:
    return {
        "m_now": m_now,
        "a_now": a_now,
        "gaze_now": batch["gaze_now"],
        "pose_now": batch["pose_now"],
        "cam_now": batch["cam_now"],
        "m_prev": batch["m_prev"],
        "a_prev": a_prev,
        "gaze_prev": batch["gaze_prev"],
        "pose_prev": batch["pose_prev"],
        "cam_prev": batch["cam_prev"],
        "m_ref": batch["m_ref"],
    }


def shared_noised_motion(m_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(m_now)
    times = torch.empty(m_now.size(0), device=m_now.device).uniform_(0.1, 0.95)
    t = times.view(times.shape[0], *([1] * (m_now.ndim - 1)))
    noised_motion = t * m_now + (1 - t) * noise
    gt_flow = m_now - noise
    return noised_motion, gt_flow, times


def clone_model_batch(batch: dict) -> dict:
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def zero_audio_in_place(batch: dict) -> dict:
    batch["a_now"] = torch.zeros_like(batch["a_now"])
    batch["a_prev"] = torch.zeros_like(batch["a_prev"])
    return batch


def forward_model_no_dropout(model: FMGenerator, batch: dict, times: torch.Tensor) -> torch.Tensor:
    x, prev_x = batch["m_now"], batch["m_prev"]
    a, prev_a = batch["a_now"], batch["a_prev"]
    m_ref = batch["m_ref"]
    gaze = batch.get("gaze", batch.get("gaze_now"))
    prev_gaze = batch["gaze_prev"]
    pose = batch.get("pose", batch.get("pose_now"))
    prev_pose = batch["pose_prev"]
    cam = batch.get("cam", batch.get("cam_now"))
    prev_cam = batch["cam_prev"]

    bs = x.size(0)
    if not model.opt.only_last_features:
        a = a.view(bs, model.num_frames_for_clip, -1)
        prev_a = prev_a.view(bs, model.num_prev_frames, -1)

    a = model.audio_projection(a)
    prev_a = model.audio_projection(prev_a)
    gaze = model.gaze_projection(gaze)
    prev_gaze = model.gaze_projection(prev_gaze)
    pose = model.pose_projection(pose)
    prev_pose = model.pose_projection(prev_pose)
    cam = model.cam_projection(cam)
    prev_cam = model.cam_projection(prev_cam)

    pred = model.fmt(
        times,
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
        train=False,
    )
    return pred[:, model.num_prev_frames :, ...]


def capture_block_outputs(model: FMGenerator, batch: dict, times: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    captured: list[torch.Tensor] = []
    handles = []

    def _hook(_, __, output):
        captured.append(output.detach())

    for block in model.fmt.blocks:
        handles.append(block.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            output = forward_model_no_dropout(model, batch, times)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def per_sample_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(start_dim=1).mean(dim=1)


def compute_psd_summary(v: torch.Tensor) -> torch.Tensor:
    motion = torch.cumsum(v, dim=1)
    motion = motion - motion.mean(dim=1, keepdim=True)
    fft = torch.fft.rfft(motion, dim=1)
    power = (fft.abs() ** 2).mean(dim=(0, 2))
    return power.detach().cpu()


def tensor_to_numpy_rows(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().reshape(-1, x.shape[-1]).numpy()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_block_divergence(values: list[float], out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(values))
    plt.plot(xs, values, marker="o")
    plt.xticks(xs, [f"block_{i}" for i in xs], rotation=45)
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_psd(freqs: np.ndarray, teacher_psd: np.ndarray, student_psd: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, teacher_psd, label="teacher", linewidth=2)
    plt.plot(freqs, student_psd, label="student", linewidth=2)
    plt.xlabel("Temporal frequency bin")
    plt.ylabel("Average power")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_tbin_loss(bin_centers: list[str], means: list[float], counts: list[int], out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(bin_centers))
    plt.bar(xs, means)
    for idx, count in enumerate(counts):
        plt.text(idx, means[idx], str(count), ha="center", va="bottom", fontsize=9)
    plt.xticks(xs, bin_centers, rotation=45)
    plt.ylabel("Mean L_distill")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cfg_cosine(values: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=20, alpha=0.85)
    plt.xlabel("cosine(student_cfg_dir, teacher_cfg_dir)")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_velocity_histograms(student_vel: np.ndarray, teacher_vel: np.ndarray, out_path: Path) -> list[dict]:
    dim = student_vel.shape[1]
    cols = 4
    rows = int(math.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    axes = np.asarray(axes).reshape(-1)
    ratio_rows = []

    for d in range(dim):
        ax = axes[d]
        s = student_vel[:, d]
        t = teacher_vel[:, d]
        bins = np.linspace(
            min(s.min(), t.min()),
            max(s.max(), t.max()),
            50,
        )
        ax.hist(t, bins=bins, alpha=0.45, density=True, label="teacher")
        ax.hist(s, bins=bins, alpha=0.45, density=True, label="student")
        ax.set_title(f"dim {d}")
        if d == 0:
            ax.legend(fontsize=8)
        t_var = float(np.var(t))
        s_var = float(np.var(s))
        ratio_rows.append(
            {
                "dim": d,
                "teacher_mean": float(np.mean(t)),
                "student_mean": float(np.mean(s)),
                "teacher_std": float(np.std(t)),
                "student_std": float(np.std(s)),
                "variance_ratio_student_over_teacher": float(s_var / (t_var + 1e-8)),
            }
        )

    for ax in axes[dim:]:
        ax.axis("off")

    fig.suptitle("Per-dim velocity distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return ratio_rows


def _load_audio_native(path: Path, target_sr: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32, copy=False)


def _load_wav_mono(path: Path, target_sr: int, device: str) -> torch.Tensor:
    audio = _load_audio_native(path, target_sr)
    return torch.from_numpy(audio).to(device=device, dtype=torch.float32)


def make_session_opt(args: argparse.Namespace) -> SimpleNamespace:
    opt = build_base_opt(
        audio_feat_dim=args.student_audio_feat_dim,
        rank=args.device,
        device=args.device,
        renderer_path=args.renderer_path,
        wav2vec_model_path=args.wav2vec_model_path,
        nfe=args.swap_nfe,
        a_cfg_scale=1.0,
        moshi_repo=args.moshi_repo,
        hf_repo=args.mimi_hf_repo,
        use_ema=args.use_ema,
    )
    opt.debug_session = False
    return opt


def encode_wav_to_aligned_latents(session: LiveMoshiIMTalkerSession, wav_path: Path) -> torch.Tensor:
    wav = _load_wav_mono(wav_path, int(session.mimi.sample_rate), str(session.device))
    raw_chunks: list[torch.Tensor] = []
    aligned_chunks: list[torch.Tensor] = []
    cursor = 0
    session.reset_reply()

    while cursor < wav.numel():
        end = min(cursor + session.chunk_samples, wav.numel())
        original_num_samples = int(end - cursor)
        chunk = wav[cursor:end].clone()
        if chunk.numel() < session.chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, session.chunk_samples - chunk.numel()))

        latents = session._encode_reply_pcm_to_latents(
            chunk,
            original_num_samples=original_num_samples,
        )
        target_frames = session._target_frames_for_pcm(original_num_samples)
        max_overlap_frames = max(session.render_window_frames - target_frames, 0)
        overlap_frames = min(session.overlap_frames, max_overlap_frames)

        latents_for_align = latents
        if (
            session.overlap_context_latent_count > 0
            and session._overlap_latents is not None
            and session._overlap_latents.shape[0] >= session.overlap_context_latent_count
        ):
            latents_for_align = torch.cat([session._overlap_latents, latents], dim=0)
        elif overlap_frames > 0:
            overlap_frames = 0

        aligned_chunk = session._align_raw_mimi_to_target_frames(
            latents_for_align,
            target_frames + overlap_frames,
        )

        if session.overlap_context_latent_count > 0:
            session._overlap_latents = latents[-session.overlap_context_latent_count :].clone()
        else:
            session._overlap_latents = None

        emitted_aligned = aligned_chunk[overlap_frames:].contiguous() if overlap_frames > 0 else aligned_chunk
        raw_chunks.append(latents)
        aligned_chunks.append(emitted_aligned)
        cursor += session.chunk_samples

    aligned_full = torch.cat(aligned_chunks, dim=0).to(session.device)
    return aligned_full.unsqueeze(0)


def build_hybrid_model(student_model: FMGenerator, teacher_model: FMGenerator, device: str) -> FMGenerator:
    hybrid = copy.deepcopy(student_model)
    hybrid.fmt = copy.deepcopy(teacher_model.fmt)
    hybrid.gaze_projection = copy.deepcopy(teacher_model.gaze_projection)
    hybrid.pose_projection = copy.deepcopy(teacher_model.pose_projection)
    hybrid.cam_projection = copy.deepcopy(teacher_model.cam_projection)
    hybrid.to(device).eval()
    return hybrid


def save_video_tensor(frames: torch.Tensor, out_path: Path, fps: float, audio_path: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)

    vid = frames.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    vid = (vid * 255).round().to(torch.uint8)
    torchvision.io.write_video(str(temp_path), vid, fps=fps)

    if audio_path is not None and audio_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.unlink(missing_ok=True)
    else:
        temp_path.replace(out_path)


def merge_videos(videos: list[Path], labels: list[str], out_path: Path) -> None:
    cmd = [
        "python3",
        str(Path(__file__).resolve().parent / "compare_videos.py"),
        *[str(v) for v in videos],
        "--labels",
        *labels,
        "--output",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def render_module_swap_triplet(
    args: argparse.Namespace,
    report_dir: Path,
    student_model: FMGenerator,
    teacher_model: FMGenerator,
) -> dict | None:
    if args.skip_swap_render or not args.swap_ref_path or not args.swap_wav_path:
        return None

    session_opt = make_session_opt(args)
    session = LiveMoshiIMTalkerSession(
        session_opt,
        generator_path=args.student_ckpt,
        renderer_path=args.renderer_path,
        ref_path=args.swap_ref_path,
        crop=args.swap_crop,
        nfe=args.swap_nfe,
        a_cfg_scale=1.0,
        moshi_repo=args.moshi_repo,
        mimi_hf_repo=args.mimi_hf_repo,
    )

    hybrid_model = build_hybrid_model(student_model, teacher_model, args.device)
    wav_path = Path(args.swap_wav_path)
    audio_24k = _load_wav_mono(wav_path, int(session.mimi.sample_rate), str(session.device))
    audio_16k = _load_wav_mono(wav_path, 16000, str(session.device)).unsqueeze(0)
    aligned = encode_wav_to_aligned_latents(session, wav_path)

    cond = {
        "ref_x": session.ref_x,
        "a_feat": aligned,
        "gaze": None,
        "pose": None,
        "cam": None,
    }

    with torch.no_grad():
        student_sample = session.fm.sample(cond, a_cfg_scale=1.0, nfe=args.swap_nfe, seed=args.seed)
        student_frames = session._decode_sample_to_frames(student_sample)

        hybrid_sample = hybrid_model.sample(cond, a_cfg_scale=1.0, nfe=args.swap_nfe, seed=args.seed)
        hybrid_frames = session._decode_sample_to_frames(hybrid_sample)

        teacher_sample = teacher_model.sample(
            {
                "ref_x": session.ref_x,
                "a": audio_16k,
                "gaze": None,
                "pose": None,
                "cam": None,
            },
            a_cfg_scale=args.cfg_scale,
            nfe=max(args.swap_nfe, 10),
            seed=args.seed,
        )
        teacher_frames = session._decode_sample_to_frames(teacher_sample)

    video_dir = report_dir / "videos"
    teacher_path = video_dir / "teacher_full.mp4"
    student_path = video_dir / "student_full.mp4"
    hybrid_path = video_dir / "swap_studentAudioProj_teacherDeep.mp4"
    merged_path = video_dir / "swap_triplet_merged.mp4"

    save_video_tensor(teacher_frames, teacher_path, fps=float(session.opt.fps), audio_path=wav_path)
    save_video_tensor(student_frames, student_path, fps=float(session.opt.fps), audio_path=wav_path)
    save_video_tensor(hybrid_frames, hybrid_path, fps=float(session.opt.fps), audio_path=wav_path)
    merge_videos(
        [teacher_path, student_path, hybrid_path],
        ["teacher_full", "student_full", "swap_audioProj->teacher"],
        merged_path,
    )

    return {
        "teacher_full": str(teacher_path),
        "student_full": str(student_path),
        "swap_studentAudioProj_teacherDeep": str(hybrid_path),
        "merged": str(merged_path),
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_root) / ts
    plots_dir = report_dir / "plots"
    tables_dir = report_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    diag_title = args.title or f"teacher_vs_student_diagnostics_{ts}"

    data_opt = build_base_opt(
        dataset_path=args.dataset_path,
        audio_subdir=args.student_audio_subdir,
        teacher_audio_subdir=args.teacher_audio_subdir,
    )
    split_path = str(Path(args.dataset_path) / args.split)
    dataset = AudioMotionSmirkGazeDataset(opt=data_opt, split_path=split_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    student_opt = build_base_opt(
        audio_feat_dim=args.student_audio_feat_dim,
        wav2vec_model_path=args.wav2vec_model_path,
        rank=args.device,
    )
    teacher_opt = build_base_opt(
        audio_feat_dim=args.teacher_audio_feat_dim,
        wav2vec_model_path=args.wav2vec_model_path,
        rank=args.device,
    )

    student = FMGenerator(student_opt).to(args.device).eval()
    teacher = FMGenerator(teacher_opt).to(args.device).eval()
    student_load_info = load_generator_checkpoint(student, args.student_ckpt, args.use_ema)
    teacher_load_info = load_generator_checkpoint(teacher, args.teacher_ckpt, args.use_ema)
    for param in student.parameters():
        param.requires_grad = False
    for param in teacher.parameters():
        param.requires_grad = False

    block_mse_sums = None
    psd_teacher_sum = None
    psd_student_sum = None
    psd_count = 0
    teacher_vel_rows = []
    student_vel_rows = []
    cfg_cosines = []
    tbin_values = defaultdict(list)
    tbin_edges = np.linspace(0.1, 0.95, args.num_t_bins + 1)

    num_processed = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        batch = move_batch_to_device(batch, args.device)

        m_now = batch["m_now"]
        noised_motion, gt_flow, times = shared_noised_motion(m_now)

        student_batch = make_model_batch(
            batch,
            m_now=noised_motion,
            a_now=batch["a_now"],
            a_prev=batch["a_prev"],
        )
        teacher_batch = make_model_batch(
            batch,
            m_now=noised_motion,
            a_now=batch["teacher_a_now"],
            a_prev=batch["teacher_a_prev"],
        )

        v_student, student_hidden = capture_block_outputs(student, student_batch, times)
        v_teacher, teacher_hidden = capture_block_outputs(teacher, teacher_batch, times)

        block_mses = [F.mse_loss(s, t).item() for s, t in zip(student_hidden, teacher_hidden)]
        if block_mse_sums is None:
            block_mse_sums = np.zeros(len(block_mses), dtype=np.float64)
        block_mse_sums += np.array(block_mses, dtype=np.float64)

        teacher_vel_rows.append(tensor_to_numpy_rows(v_teacher))
        student_vel_rows.append(tensor_to_numpy_rows(v_student))

        psd_teacher = compute_psd_summary(v_teacher)
        psd_student = compute_psd_summary(v_student)
        if psd_teacher_sum is None:
            psd_teacher_sum = torch.zeros_like(psd_teacher)
            psd_student_sum = torch.zeros_like(psd_student)
        psd_teacher_sum += psd_teacher
        psd_student_sum += psd_student
        psd_count += 1

        sample_distill = per_sample_mse(v_student, v_teacher).detach().cpu().numpy()
        time_np = times.detach().cpu().numpy()
        bin_ids = np.digitize(time_np, tbin_edges[1:-1], right=False)
        for sample_loss, sample_time, bin_id in zip(sample_distill, time_np, bin_ids):
            tbin_values[int(bin_id)].append(float(sample_loss))

        student_uncond = forward_model_no_dropout(
            student, clone_model_batch(zero_audio_in_place(student_batch)), times
        )
        teacher_uncond = forward_model_no_dropout(
            teacher, clone_model_batch(zero_audio_in_place(teacher_batch)), times
        )
        teacher_cfg_dir = (v_teacher - teacher_uncond).flatten(start_dim=1)
        student_cfg_dir = (v_student - student_uncond).flatten(start_dim=1)
        cosine = F.cosine_similarity(student_cfg_dir, teacher_cfg_dir, dim=1)
        cfg_cosines.extend(cosine.detach().cpu().tolist())

        num_processed += 1

    if num_processed == 0:
        raise RuntimeError("No batches processed for diagnostics.")

    block_mse_mean = (block_mse_sums / num_processed).tolist()
    teacher_vel = np.concatenate(teacher_vel_rows, axis=0)
    student_vel = np.concatenate(student_vel_rows, axis=0)
    teacher_psd = (psd_teacher_sum / psd_count).numpy()
    student_psd = (psd_student_sum / psd_count).numpy()
    freqs = np.arange(len(teacher_psd))

    block_plot = plots_dir / "per_block_divergence.png"
    plot_block_divergence(block_mse_mean, block_plot, f"{diag_title}: per-block divergence")

    psd_plot = plots_dir / "velocity_psd.png"
    plot_psd(freqs, teacher_psd, student_psd, psd_plot, f"{diag_title}: motion-latent PSD")

    hist_plot = plots_dir / "per_dim_velocity_histograms.png"
    variance_rows = plot_velocity_histograms(student_vel, teacher_vel, hist_plot)
    variance_csv = tables_dir / "velocity_dim_variance_ratios.csv"
    save_csv(variance_csv, variance_rows)

    tbin_rows = []
    tbin_labels = []
    tbin_means = []
    tbin_counts = []
    for idx in range(args.num_t_bins):
        lo, hi = tbin_edges[idx], tbin_edges[idx + 1]
        vals = tbin_values.get(idx, [])
        label = f"[{lo:.2f},{hi:.2f})" if idx < args.num_t_bins - 1 else f"[{lo:.2f},{hi:.2f}]"
        mean_val = float(np.mean(vals)) if vals else float("nan")
        tbin_rows.append({"bin": label, "count": len(vals), "mean_distill_loss": mean_val})
        tbin_labels.append(label)
        tbin_means.append(0.0 if math.isnan(mean_val) else mean_val)
        tbin_counts.append(len(vals))
    tbin_csv = tables_dir / "distill_by_t_bin.csv"
    save_csv(tbin_csv, tbin_rows)
    tbin_plot = plots_dir / "distill_by_t_bin.png"
    plot_tbin_loss(tbin_labels, tbin_means, tbin_counts, tbin_plot, f"{diag_title}: L_distill by t-bin")

    cfg_cosines_np = np.asarray(cfg_cosines, dtype=np.float32)
    cfg_rows = [
        {
            "num_samples": int(cfg_cosines_np.size),
            "cosine_mean": float(cfg_cosines_np.mean()),
            "cosine_std": float(cfg_cosines_np.std()),
            "cosine_p05": float(np.percentile(cfg_cosines_np, 5)),
            "cosine_p50": float(np.percentile(cfg_cosines_np, 50)),
            "cosine_p95": float(np.percentile(cfg_cosines_np, 95)),
        }
    ]
    cfg_csv = tables_dir / "cfg_direction_cosine.csv"
    save_csv(cfg_csv, cfg_rows)
    cfg_plot = plots_dir / "cfg_direction_cosine_hist.png"
    plot_cfg_cosine(cfg_cosines_np, cfg_plot, f"{diag_title}: CFG direction cosine")

    swap_outputs = render_module_swap_triplet(args, report_dir, student, teacher)

    report = {
        "title": diag_title,
        "report_dir": str(report_dir),
        "student_load_info": student_load_info,
        "teacher_load_info": teacher_load_info,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_batches": num_processed,
        "batch_size": args.batch_size,
        "device": args.device,
        "use_ema": args.use_ema,
        "plots": {
            "per_block_divergence": str(block_plot),
            "velocity_psd": str(psd_plot),
            "per_dim_velocity_histograms": str(hist_plot),
            "distill_by_t_bin": str(tbin_plot),
            "cfg_direction_cosine_hist": str(cfg_plot),
        },
        "tables": {
            "velocity_dim_variance_ratios": str(variance_csv),
            "distill_by_t_bin": str(tbin_csv),
            "cfg_direction_cosine": str(cfg_csv),
        },
        "metrics": {
            "per_block_divergence_mse": block_mse_mean,
            "psd_summary": {
                "teacher_mean_power": teacher_psd.tolist(),
                "student_mean_power": student_psd.tolist(),
            },
            "cfg_direction_cosine": cfg_rows[0],
            "distill_by_t_bin": tbin_rows,
            "velocity_variance_ratios": variance_rows,
        },
        "module_swap_render": swap_outputs,
    }
    report_json = report_dir / "report.json"
    save_json(report_json, report)

    summary_lines = [
        f"report_dir={report_dir}",
        f"per_block_divergence_plot={block_plot}",
        f"velocity_psd_plot={psd_plot}",
        f"velocity_histograms_plot={hist_plot}",
        f"distill_tbin_csv={tbin_csv}",
        f"cfg_cosine_csv={cfg_csv}",
    ]
    if swap_outputs:
        summary_lines.append(f"swap_render_merged={swap_outputs['merged']}")
    print("\n".join(summary_lines))
    print(f"Saved report JSON to {report_json}")


if __name__ == "__main__":
    main()
