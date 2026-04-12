import argparse
import math
import sys
import traceback
from pathlib import Path

def default_moshi_repo():
    return Path(__file__).resolve().parents[2] / "moshi"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke check Mimi discrete encode shape, dtype, and streaming token count."
    )
    parser.add_argument(
        "--moshi_repo",
        type=Path,
        default=default_moshi_repo(),
        help="Path to the sibling moshi repo root.",
    )
    parser.add_argument(
        "--hf_repo",
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face repo to load Mimi from.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for Mimi (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--secs",
        type=float,
        default=2.0,
        help="Length of fake zero PCM to encode.",
    )
    return parser.parse_args()


def ensure_moshi_importable(moshi_repo):
    try:
        from moshi.models import loaders
        return loaders
    except ImportError:
        moshi_pkg = Path(moshi_repo) / "moshi"
        if str(moshi_pkg) not in sys.path:
            sys.path.insert(0, str(moshi_pkg))
        from moshi.models import loaders
        return loaders


def is_integer_dtype(dtype):
    dtype_name = str(dtype)
    return dtype_name.startswith("torch.int") or dtype_name.startswith("torch.uint")


def build_zero_pcm(sample_rate, frame_size, secs, device, torch):
    n_samples = int(math.ceil(secs * sample_rate))
    if n_samples <= 0:
        raise ValueError("--secs must produce at least one sample.")
    n_padded = int(math.ceil(n_samples / frame_size) * frame_size)
    wav = torch.zeros(1, 1, n_padded, device=device)
    return wav, n_samples, n_padded


def main():
    args = parse_args()
    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = ensure_moshi_importable(args.moshi_repo)

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    mimi = checkpoint_info.get_mimi(device=args.device)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    wav, n_samples, n_padded = build_zero_pcm(
        sample_rate=mimi.sample_rate,
        frame_size=frame_size,
        secs=args.secs,
        device=args.device,
        torch=torch,
    )

    print(f"sample_rate={mimi.sample_rate}")
    print(f"frame_rate={mimi.frame_rate}")
    print(f"frame_size={frame_size}")
    print(f"n_samples={n_samples}")
    print(f"n_padded={n_padded}")

    with torch.no_grad():
        codes = mimi.encode(wav)
    print(f"one_shot_codes.shape={tuple(codes.shape)}")
    print(f"one_shot_codes.dtype={codes.dtype}")

    assert codes.dim() == 3, f"expected 3D codes, got {codes.dim()}D"
    assert is_integer_dtype(codes.dtype), f"expected integer dtype, got {codes.dtype}"
    assert codes.shape[1] >= 1, f"expected at least one codebook, got {codes.shape[1]}"

    B, n_q, T = codes.shape
    print(f"B={B} n_q={n_q} T={T}")

    cb0 = codes[:, 0]
    print(f"cb0.shape={tuple(cb0.shape)}")
    print(f"cb0.dtype={cb0.dtype}")
    print(f"cb0_first8={cb0[0, :8].tolist()}")

    assert cb0.dim() == 2, f"expected cb0 to be 2D, got {cb0.dim()}D"
    assert is_integer_dtype(cb0.dtype), f"expected cb0 integer dtype, got {cb0.dtype}"

    split = (n_padded // 2 // frame_size) * frame_size
    if split <= 0 or split >= n_padded:
        raise ValueError(
            f"Need at least two frame-aligned chunks for streaming check, got n_padded={n_padded}."
        )

    mimi.streaming_forever(1)
    mimi.reset_streaming()

    chunk1 = wav[:, :, :split]
    chunk2 = wav[:, :, split:]
    with torch.no_grad():
        chunk1_codes = mimi.encode(chunk1)
        chunk2_codes = mimi.encode(chunk2)

    print(f"chunk1_codes.shape={tuple(chunk1_codes.shape)}")
    print(f"chunk2_codes.shape={tuple(chunk2_codes.shape)}")

    streaming_total = chunk1_codes.shape[-1] + chunk2_codes.shape[-1]
    print(f"streaming_total={streaming_total}")
    print(f"one_shot_T={T}")
    print(f"observed_token_rate={T / (n_padded / mimi.sample_rate):.6f}")

    assert streaming_total == T, (
        f"streaming token count mismatch: streaming_total={streaming_total}, one_shot_T={T}"
    )

    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
