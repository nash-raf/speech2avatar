import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from generator.FM import AudioBridge768


class PairedAudioFeatureDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        wav2vec_subdir: str,
        mimi_subdir: str,
        stems: list[str],
        clip_frames: int,
        random_window: bool = True,
    ):
        self.root = Path(dataset_path)
        self.wav2vec_dir = self.root / wav2vec_subdir
        self.mimi_dir = self.root / mimi_subdir
        self.stems = stems
        self.clip_frames = clip_frames
        self.random_window = random_window

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[index]
        wav2vec = np.load(self.wav2vec_dir / f"{stem}.npy", mmap_mode="r")
        mimi = np.load(self.mimi_dir / f"{stem}.npy", mmap_mode="r")

        max_len = min(len(wav2vec), len(mimi))
        if max_len < self.clip_frames:
            raise RuntimeError(f"paired clip {stem} shorter than clip_frames={self.clip_frames}")

        if self.random_window:
            start = random.randint(0, max_len - self.clip_frames)
        else:
            start = max(0, (max_len - self.clip_frames) // 2)
        end = start + self.clip_frames

        wav2vec_clip = torch.from_numpy(wav2vec[start:end].copy()).float()
        mimi_clip = torch.from_numpy(mimi[start:end].copy()).float()
        return mimi_clip, wav2vec_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised Mimi->wav2vec bridge pretrain.")
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--exp_path", default="./exps", type=str)
    parser.add_argument("--exp_name", default="mimi_bridge_768_control", type=str)
    parser.add_argument("--wav2vec_subdir", default="audio", type=str)
    parser.add_argument("--mimi_subdir", default="audio_mimi", type=str)
    parser.add_argument("--steps", default=3000, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip_frames", default=100, type=int)
    parser.add_argument("--adapter_hidden_dim", default=1024, type=int)
    parser.add_argument("--val_interval", default=300, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_paired_stems(dataset_path: str, wav2vec_subdir: str, mimi_subdir: str, clip_frames: int) -> list[str]:
    root = Path(dataset_path)
    wav2vec_dir = root / wav2vec_subdir
    mimi_dir = root / mimi_subdir
    wav2vec_stems = {p.stem for p in wav2vec_dir.glob("*.npy")}
    mimi_stems = {p.stem for p in mimi_dir.glob("*.npy")}
    stems = sorted(wav2vec_stems & mimi_stems)
    valid_stems = []
    for stem in stems:
        wav2vec = np.load(wav2vec_dir / f"{stem}.npy", mmap_mode="r")
        mimi = np.load(mimi_dir / f"{stem}.npy", mmap_mode="r")
        if min(len(wav2vec), len(mimi)) >= clip_frames:
            valid_stems.append(stem)
    if not valid_stems:
        raise RuntimeError("No paired wav2vec/Mimi feature stems found.")
    return valid_stems


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, dataset, batch_size: int, device: torch.device) -> float:
    """Full-val-set L1. Deterministic (shuffle=False, single-window dataset)."""
    if len(dataset) == 0:
        return float("nan")
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
    )
    total_loss = 0.0
    total_elems = 0
    for mimi_clip, wav2vec_clip in loader:
        mimi_clip = mimi_clip.to(device)
        wav2vec_clip = wav2vec_clip.to(device)
        pred = model(mimi_clip)
        # accumulate with correct weighting across variable-sized last batch
        n = wav2vec_clip.numel()
        total_loss += F.l1_loss(pred, wav2vec_clip, reduction="sum").item()
        total_elems += n
    return total_loss / max(1, total_elems)


@torch.no_grad()
def wav2vec_target_stats(
    dataset_path: str,
    wav2vec_subdir: str,
    stems: list[str],
    clip_frames: int,
    max_clips: int = 512,
) -> dict[str, float]:
    """Compute summary statistics of the raw wav2vec target distribution.

    Used to interpret L1 losses against the scale of the target:
      norm_l1 = val_l1 / target_std.
    """
    root = Path(dataset_path) / wav2vec_subdir
    vals = []
    sample_stems = stems[: min(len(stems), max_clips)]
    for stem in sample_stems:
        arr = np.load(root / f"{stem}.npy", mmap_mode="r")
        if len(arr) < clip_frames:
            continue
        start = max(0, (len(arr) - clip_frames) // 2)
        clip = np.asarray(arr[start : start + clip_frames], dtype=np.float32)
        vals.append(clip)
    if not vals:
        return {"target_mean_abs": float("nan"), "target_std": float("nan")}
    stacked = np.concatenate(vals, axis=0)
    return {
        "target_mean_abs": float(np.mean(np.abs(stacked))),
        "target_std": float(np.std(stacked)),
        "target_mean": float(np.mean(stacked)),
        "target_min": float(np.min(stacked)),
        "target_max": float(np.max(stacked)),
        "num_clips_sampled": int(len(vals)),
    }


@torch.no_grad()
def distribution_summary(model, dataset, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False, drop_last=False)
    pred_chunks = []
    tgt_chunks = []
    for mimi_clip, wav2vec_clip in loader:
        pred_chunks.append(model(mimi_clip.to(device)).cpu())
        tgt_chunks.append(wav2vec_clip.cpu())
    pred_all = torch.cat(pred_chunks, dim=0).reshape(-1, pred_chunks[0].shape[-1])
    tgt_all = torch.cat(tgt_chunks, dim=0).reshape(-1, tgt_chunks[0].shape[-1])

    pred_mean = pred_all.mean(dim=0)
    tgt_mean = tgt_all.mean(dim=0)
    pred_std = pred_all.std(dim=0)
    tgt_std = tgt_all.std(dim=0)
    std_ratio = pred_std / (tgt_std + 1e-8)

    return {
        "mean_abs_mean_delta": float((pred_mean - tgt_mean).abs().mean().item()),
        "mean_abs_std_ratio_minus_one": float((std_ratio - 1.0).abs().mean().item()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    exp_dir = Path(args.exp_path) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_paired_stems(
        args.dataset_path,
        args.wav2vec_subdir,
        args.mimi_subdir,
        args.clip_frames,
    )
    split_at = max(0, len(stems) - 100)
    train_stems = stems[:split_at] if split_at > 0 else stems
    val_stems = stems[split_at:] if split_at > 0 else stems

    target_stats = wav2vec_target_stats(
        args.dataset_path,
        args.wav2vec_subdir,
        train_stems,
        args.clip_frames,
    )
    print("[bridge] wav2vec target stats (sampled): " + json.dumps(target_stats, indent=2))
    target_std = target_stats.get("target_std") or float("nan")

    train_dataset = PairedAudioFeatureDataset(
        args.dataset_path,
        args.wav2vec_subdir,
        args.mimi_subdir,
        train_stems,
        args.clip_frames,
        random_window=True,
    )
    val_dataset = PairedAudioFeatureDataset(
        args.dataset_path,
        args.wav2vec_subdir,
        args.mimi_subdir,
        val_stems,
        args.clip_frames,
        random_window=False,
    )
    val_summary_dataset = PairedAudioFeatureDataset(
        args.dataset_path,
        args.wav2vec_subdir,
        args.mimi_subdir,
        val_stems,
        args.clip_frames,
        random_window=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_iter = cycle_loader(train_loader)

    model = AudioBridge768(512, 768, hidden_dim=args.adapter_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    recent_losses = deque(maxlen=args.log_interval)
    history = []
    best_val = float("inf")

    model.train()
    for step in range(1, args.steps + 1):
        mimi_clip, wav2vec_clip = next(train_iter)
        mimi_clip = mimi_clip.to(device)
        wav2vec_clip = wav2vec_clip.to(device)

        pred = model(mimi_clip)
        loss = F.l1_loss(pred, wav2vec_clip)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        recent_losses.append(loss.item())

        should_validate = step == 1 or step % args.val_interval == 0 or step == args.steps
        val_l1 = None
        if should_validate:
            model.eval()
            val_l1 = evaluate(model, val_dataset, args.batch_size, device)
            model.train()
            best_val = min(best_val, val_l1)

        if step % args.log_interval == 0 or should_validate:
            train_l1 = float(np.mean(recent_losses))
            msg = f"[bridge] step={step}/{args.steps} train_l1={train_l1:.6f}"
            if val_l1 is not None:
                msg += f" val_l1={val_l1:.6f} best_val={best_val:.6f}"
            print(msg)
            history.append(
                {
                    "step": step,
                    "train_l1": train_l1,
                    "val_l1": val_l1,
                    "best_val_l1": best_val,
                }
            )

    model.eval()
    summary = distribution_summary(model, val_summary_dataset, device)
    final_val = evaluate(model, val_dataset, args.batch_size, device)

    ckpt_path = exp_dir / "bridge_pretrained.pt"
    torch.save(model.state_dict(), ckpt_path)

    # normalise by target std so val L1 is comparable across feature scales
    if target_std and not np.isnan(target_std) and target_std > 0:
        final_val_norm = final_val / target_std
        best_val_norm = best_val / target_std
    else:
        final_val_norm = float("nan")
        best_val_norm = float("nan")

    report = {
        "train_stems": len(train_stems),
        "val_stems": len(val_stems),
        "steps": args.steps,
        "final_val_l1": final_val,
        "best_val_l1": best_val,
        "final_val_l1_normalised_by_target_std": final_val_norm,
        "best_val_l1_normalised_by_target_std": best_val_norm,
        "target_stats": target_stats,
        **summary,
        "history": history,
        "bridge_ckpt": str(ckpt_path.resolve()),
    }
    report_path = exp_dir / "bridge_pretrain_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved bridge checkpoint: {ckpt_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
