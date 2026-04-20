from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from generator.FM import FMGenerator


def make_opt(audio_feat_dim: int, device: str, wav2vec_model_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        seed=42,
        fix_noise_seed=False,
        input_size=256,
        input_nc=3,
        fps=25.0,
        sampling_rate=16000,
        audio_feat_dim=audio_feat_dim,
        audio_marcing=2,
        wav2vec_sec=2.0,
        wav2vec_model_path=wav2vec_model_path,
        attention_window=5,
        only_last_features=True,
        average_emotion=False,
        audio_dropout_prob=0.0,
        ref_dropout_prob=0.0,
        emotion_dropout_prob=0.0,
        style_dim=512,
        dim_a=512,
        dim_h=512,
        dim_e=7,
        dim_motion=32,
        dim_c=32,
        dim_w=32,
        fmt_depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        no_learned_pe=False,
        num_prev_frames=10,
        max_grad_norm=1.0,
        ode_atol=1e-5,
        ode_rtol=1e-5,
        torchdiffeq_ode_method="euler",
        a_cfg_scale=3.0,
        swin_res_threshold=128,
        window_size=8,
        rank=device,
        debug_session=False,
    )


def load_fm_weights(model: FMGenerator, ckpt_path: str) -> tuple[int, int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    else:
        state_dict = ckpt.get("state_dict", ckpt)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value

    model_state = model.state_dict()
    loadable = {
        key: value
        for key, value in cleaned.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = len(cleaned) - len(loadable)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return len(loadable), skipped, len(missing) + len(unexpected)


def stems_from_split(root: Path, split: str, limit: int) -> list[str]:
    stems = [
        Path(line.strip()).stem
        for line in (root / split).read_text().splitlines()
        if line.strip()
    ]
    return stems[:limit]


def build_batch(root: Path, stems: list[str], audio_subdir: str, device: str) -> dict[str, torch.Tensor]:
    num_prev = 10
    num_now = 50
    required = num_prev + num_now
    batch = []
    for stem in stems:
        motion = torch.load(root / "motion" / f"{stem}.pt", map_location="cpu").float()
        audio = torch.from_numpy(np.load(root / audio_subdir / f"{stem}.npy").astype(np.float32))
        gaze = torch.from_numpy(np.load(root / "gaze" / f"{stem}.npy").astype(np.float32))
        smirk = torch.load(root / "smirk" / f"{stem}.pt", map_location="cpu")
        pose = smirk["pose_params"].float()
        cam = smirk["cam"].float()

        min_len = min(
            motion.shape[0], audio.shape[0], gaze.shape[0], pose.shape[0], cam.shape[0]
        )
        if min_len < required:
            raise RuntimeError(
                f"{stem} too short: motion={motion.shape} audio={audio.shape} "
                f"gaze={gaze.shape} pose={pose.shape} cam={cam.shape}"
            )
        motion = motion[:required]
        audio = audio[:required]
        gaze = gaze[:required]
        pose = pose[:required]
        cam = cam[:required]
        batch.append((stem, motion, audio, gaze, pose, cam))

    def stack_motion(start: int, end: int) -> torch.Tensor:
        return torch.stack([motion[start:end] for _, motion, *_ in batch], dim=0).to(device)

    def stack_audio(start: int, end: int) -> torch.Tensor:
        return torch.stack([audio[start:end] for _, _, audio, *_ in batch], dim=0).to(device)

    def stack_gaze(start: int, end: int) -> torch.Tensor:
        return torch.stack([gaze[start:end] for _, _, _, gaze, _, _ in batch], dim=0).to(device)

    def stack_pose(start: int, end: int) -> torch.Tensor:
        return torch.stack([pose[start:end] for _, _, _, _, pose, _ in batch], dim=0).to(device)

    def stack_cam(start: int, end: int) -> torch.Tensor:
        return torch.stack([cam[start:end] for _, _, _, _, _, cam in batch], dim=0).to(device)

    motion_full = torch.stack([motion for _, motion, *_ in batch], dim=0).to(device)
    return {
        "stems": [stem for stem, *_ in batch],
        "m_now_clean": stack_motion(num_prev, required),
        "m_prev": stack_motion(0, num_prev),
        "a_now": stack_audio(num_prev, required),
        "a_prev": stack_audio(0, num_prev),
        "gaze_now": stack_gaze(num_prev, required),
        "gaze_prev": stack_gaze(0, num_prev),
        "pose_now": stack_pose(num_prev, required),
        "pose_prev": stack_pose(0, num_prev),
        "cam_now": stack_cam(num_prev, required),
        "cam_prev": stack_cam(0, num_prev),
        "m_ref": motion_full[:, 0],
    }


def set_trainable(student: FMGenerator) -> list[str]:
    for param in student.parameters():
        param.requires_grad = False
    trainable_prefixes = ["audio_projection.", "fmt.blocks.0.", "fmt.blocks.1."]
    names = []
    for name, param in student.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad = True
            names.append(name)
    return names


def grad_norm(module: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, param in module.named_parameters():
        if name.startswith(prefix) and param.grad is not None:
            total += float(param.grad.detach().float().norm().item() ** 2)
    return total ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/hdtf_preprocess")
    parser.add_argument("--teacher_ckpt", default="/workspace/IMTalker/checkpoints/generator.ckpt")
    parser.add_argument("--student_ckpt", default="/workspace/IMTalker/ckpts_mimi/cont_12000_adamw_64_batch.ckpt")
    parser.add_argument("--wav2vec_model_path", default="/workspace/IMTalker/checkpoints/wav2vec2-base-960h")
    parser.add_argument("--split", default="test.txt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(1234)
    root = Path(args.root)
    stems = stems_from_split(root, args.split, args.batch_size)
    print(f"[smoke] stems={stems}")

    teacher = FMGenerator(make_opt(768, args.device, args.wav2vec_model_path)).to(args.device).eval()
    student = FMGenerator(make_opt(512, args.device, args.wav2vec_model_path)).to(args.device).train()

    print(f"[smoke] load teacher={load_fm_weights(teacher, args.teacher_ckpt)}")
    print(f"[smoke] load student={load_fm_weights(student, args.student_ckpt)}")
    for p in teacher.parameters():
        p.requires_grad = False

    trainable_names = set_trainable(student)
    print(f"[smoke] trainable_param_tensors={len(trainable_names)}")
    print(f"[smoke] trainable_first={trainable_names[:5]}")

    teacher_batch = build_batch(root, stems, "audio_wav2vec", args.device)
    student_batch = build_batch(root, stems, "audio_rt_aligned", args.device)
    print(
        "[smoke] shapes "
        f"m_now={tuple(student_batch['m_now_clean'].shape)} "
        f"student_audio={tuple(student_batch['a_now'].shape)} "
        f"teacher_audio={tuple(teacher_batch['a_now'].shape)} "
        f"gaze={tuple(student_batch['gaze_now'].shape)} "
        f"pose={tuple(student_batch['pose_now'].shape)} "
        f"cam={tuple(student_batch['cam_now'].shape)}"
    )

    m_now = student_batch["m_now_clean"]
    eps = torch.randn_like(m_now)
    times = torch.empty(m_now.shape[0], device=args.device).uniform_(0.1, 0.95)
    t_view = times.view(-1, 1, 1)
    z_t = t_view * m_now + (1.0 - t_view) * eps
    gt_flow = m_now - eps

    def prepare(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {k: v for k, v in base.items() if k != "stems"}
        out["m_now"] = z_t
        return out

    with torch.no_grad():
        v_teacher = teacher(prepare(teacher_batch), t=times)
    v_student = student(prepare(student_batch), t=times)

    l_distill = F.mse_loss(v_student, v_teacher)
    l_fm = F.mse_loss(v_student, gt_flow)
    loss = l_distill + 0.3 * l_fm
    loss.backward()

    optim = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-5)
    optim.step()
    optim.zero_grad(set_to_none=True)

    print(
        "[smoke] losses "
        f"L_distill={l_distill.item():.6f} "
        f"L_fm={l_fm.item():.6f} "
        f"L_total={loss.item():.6f}"
    )
    print(
        "[smoke] output_stats "
        f"teacher_delta={(v_teacher[:, 1:] - v_teacher[:, :-1]).norm(dim=-1).mean().item():.6f} "
        f"student_delta={(v_student[:, 1:] - v_student[:, :-1]).norm(dim=-1).mean().item():.6f} "
        f"teacher_std={v_teacher.std(dim=1).mean().item():.6f} "
        f"student_std={v_student.std(dim=1).mean().item():.6f}"
    )
    # Gradients were consumed by optimizer.step above; rerun a cheap backward for norm reporting.
    v_student_2 = student(prepare(student_batch), t=times)
    loss_2 = F.mse_loss(v_student_2, v_teacher) + 0.3 * F.mse_loss(v_student_2, gt_flow)
    loss_2.backward()
    print(
        "[smoke] grad_norms "
        f"audio_projection={grad_norm(student, 'audio_projection.'):.6f} "
        f"fmt.blocks.0={grad_norm(student, 'fmt.blocks.0.'):.6f} "
        f"fmt.blocks.1={grad_norm(student, 'fmt.blocks.1.'):.6f} "
        f"fmt.blocks.2={grad_norm(student, 'fmt.blocks.2.'):.6f}"
    )
    print("[smoke] TEACHER_STUDENT_SMOKE_OK")


if __name__ == "__main__":
    main()
