#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
S2A_DIR="${S2A_DIR:-${ROOT_DIR}/speech2avatar}"
MEAD_DIR="${MEAD_DIR:-${ROOT_DIR}/MEAD}"
MEAD_ZIP="${MEAD_ZIP:-${ROOT_DIR}/MEAD.zip}"
MEAD_URL="${MEAD_URL:-https://huggingface.co/datasets/NoahMartinezXiang/MEAD/resolve/main/MEAD.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/generator_dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${S2A_DIR}/checkpoints}"
TOOLS_DIR="${TOOLS_DIR:-${ROOT_DIR}/external_tools}"
SMIRK_DIR="${SMIRK_DIR:-${TOOLS_DIR}/smirk}"
L2CS_DIR="${L2CS_DIR:-${TOOLS_DIR}/L2CS-Net}"
OPENFACE3_DIR="${OPENFACE3_DIR:-${TOOLS_DIR}/OpenFace-3.0}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/model_cache}"
GPU_ID="${GPU_ID:-0}"
WORKERS="${WORKERS:-1}"
MOTION_BATCH="${MOTION_BATCH:-32}"
FACE_BATCH="${FACE_BATCH:-64}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-0}"
CPP_OPENFACE_BIN="${CPP_OPENFACE_BIN:-}"

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

if [[ ! -d preprocess ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" -m venv preprocess
fi

source preprocess/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  huggingface_hub \
  transformers \
  librosa \
  pandas \
  timm \
  tqdm \
  gdown \
  opencv-python-headless \
  scikit-image \
  av \
  einops \
  mediapipe \
  face_alignment

mkdir -p "${TOOLS_DIR}" "${MODEL_DIR}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"

if [[ ! -d "${SMIRK_DIR}" ]]; then
  git clone --depth 1 https://github.com/georgeretsi/smirk "${SMIRK_DIR}"
fi
if [[ ! -d "${L2CS_DIR}" ]]; then
  git clone --depth 1 https://github.com/Ahmednull/L2CS-Net "${L2CS_DIR}"
fi
if [[ ! -d "${OPENFACE3_DIR}" ]]; then
  git clone --depth 1 https://github.com/CMU-MultiComp-Lab/OpenFace-3.0 "${OPENFACE3_DIR}"
fi

python -m pip install -r "${SMIRK_DIR}/requirements.txt" || true
python -m pip install -e "${L2CS_DIR}"
python -m pip install -r "${OPENFACE3_DIR}/requirements.txt" || true

if [[ ! -f "${MEAD_ZIP}" ]]; then
  wget -c -O "${MEAD_ZIP}" "${MEAD_URL}"
fi

if ! find "${MEAD_DIR}" -type f -name '*.mp4' | head -n 1 >/dev/null 2>&1; then
  mkdir -p "${MEAD_DIR}"
  unzip -q "${MEAD_ZIP}" -d "${MEAD_DIR}"
fi

download_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ ! -s "${dest}" ]]; then
    mkdir -p "$(dirname "${dest}")"
    wget -c -O "${dest}" "${url}"
  fi
}

download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/renderer.ckpt" "${CHECKPOINT_DIR}/renderer.ckpt"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/generator.ckpt" "${CHECKPOINT_DIR}/generator.ckpt"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/config.json"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/pytorch_model.bin" "${CHECKPOINT_DIR}/wav2vec2-base-960h/pytorch_model.bin"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/preprocessor_config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/preprocessor_config.json"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/feature_extractor_config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/feature_extractor_config.json"

if [[ ! -s "${MODEL_DIR}/SMIRK_em1.pt" ]]; then
  gdown --fuzzy "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing" -O "${MODEL_DIR}/SMIRK_em1.pt"
fi

if [[ ! -s "${MODEL_DIR}/L2CSNet_gaze360.pkl" ]]; then
  gdown --folder "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing" -O "${MODEL_DIR}/l2cs_drive"
  if [[ -f "${MODEL_DIR}/l2cs_drive/L2CSNet_gaze360.pkl" ]]; then
    cp "${MODEL_DIR}/l2cs_drive/L2CSNet_gaze360.pkl" "${MODEL_DIR}/L2CSNet_gaze360.pkl"
  fi
fi

export ROOT_DIR S2A_DIR MEAD_DIR OUTPUT_DIR CHECKPOINT_DIR SMIRK_DIR L2CS_DIR OPENFACE3_DIR MODEL_DIR
export GPU_ID WORKERS MOTION_BATCH FACE_BATCH OVERWRITE LIMIT CPP_OPENFACE_BIN

python - <<'PY'
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

ROOT_DIR = Path(os.environ["ROOT_DIR"])
S2A_DIR = Path(os.environ["S2A_DIR"])
MEAD_DIR = Path(os.environ["MEAD_DIR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
CHECKPOINT_DIR = Path(os.environ["CHECKPOINT_DIR"])
SMIRK_DIR = Path(os.environ["SMIRK_DIR"])
L2CS_DIR = Path(os.environ["L2CS_DIR"])
OPENFACE3_DIR = Path(os.environ["OPENFACE3_DIR"])
MODEL_DIR = Path(os.environ["MODEL_DIR"])
GPU_ID = int(os.environ["GPU_ID"])
WORKERS = int(os.environ["WORKERS"])
MOTION_BATCH = int(os.environ["MOTION_BATCH"])
FACE_BATCH = int(os.environ["FACE_BATCH"])
OVERWRITE = os.environ["OVERWRITE"] == "1"
LIMIT = int(os.environ["LIMIT"])
CPP_OPENFACE_BIN = os.environ.get("CPP_OPENFACE_BIN", "").strip()

sys.path.insert(0, str(S2A_DIR))
sys.path.insert(0, str(SMIRK_DIR))
sys.path.insert(0, str(L2CS_DIR))
sys.path.insert(0, str(OPENFACE3_DIR))

from renderer.models import IMTRenderer
from generator.options.base_options import BaseOptions
from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe
from l2cs.utils import getArch, prep_input_numpy
from model.MLT import MLT

DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

AU_ORDER = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
]
OPENFACE3_TO_17 = {
    0: "AU01",
    1: "AU02",
    2: "AU04",
    3: "AU06",
    4: "AU09",
    5: "AU12",
    6: "AU25",
    7: "AU26",
}


def ensure_dirs():
    for name in ("motion", "audio", "smirk", "gaze", "au"):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def output_paths(stem: str):
    return {
        "motion": OUTPUT_DIR / "motion" / f"{stem}.pt",
        "audio": OUTPUT_DIR / "audio" / f"{stem}.npy",
        "smirk": OUTPUT_DIR / "smirk" / f"{stem}.pt",
        "gaze": OUTPUT_DIR / "gaze" / f"{stem}.npy",
        "au": OUTPUT_DIR / "au" / f"{stem}.npy",
    }


def all_exist(stem: str):
    return all(path.exists() and path.stat().st_size > 0 for path in output_paths(stem).values())


def iter_videos():
    videos = []
    for split in ("train", "val", "test"):
        split_root = MEAD_DIR / split
        if split_root.exists():
            videos.extend(sorted(split_root.rglob("*.mp4")))
    if LIMIT > 0:
        videos = videos[:LIMIT]
    return videos


def make_renderer():
    parser = BaseOptions().initialize(__import__("argparse").ArgumentParser())
    opt = parser.parse_args([])
    model = IMTRenderer(opt).to(DEVICE).eval()
    ckpt = torch.load(CHECKPOINT_DIR / "renderer.ckpt", map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    renderer_state = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
    model.load_state_dict(renderer_state, strict=False)
    return model


def make_audio():
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(CHECKPOINT_DIR / "wav2vec2-base-960h"), local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(str(CHECKPOINT_DIR / "wav2vec2-base-960h"), local_files_only=True)
    model = model.to(DEVICE).eval()
    return processor, model


def make_smirk():
    model = SmirkEncoder().to(DEVICE).eval()
    checkpoint = torch.load(MODEL_DIR / "SMIRK_em1.pt", map_location="cpu")
    checkpoint_encoder = {k.replace("smirk_encoder.", ""): v for k, v in checkpoint.items() if "smirk_encoder" in k}
    model.load_state_dict(checkpoint_encoder, strict=False)
    return model


def make_l2cs():
    model = getArch("ResNet50", 90)
    model.load_state_dict(torch.load(MODEL_DIR / "L2CSNet_gaze360.pkl", map_location="cpu"))
    model = model.to(DEVICE).eval()
    softmax = torch.nn.Softmax(dim=1)
    idx_tensor = torch.arange(90, dtype=torch.float32, device=DEVICE)
    return model, softmax, idx_tensor


def make_openface3():
    model = MLT().to(DEVICE).eval()
    weights_path = OPENFACE3_DIR / "weights" / "stage2_epoch_7_loss_1.1606_acc_0.5589.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    return model


def crop_face(frame_bgr):
    kpts = run_mediapipe(frame_bgr)
    if kpts is None:
        resized = cv2.resize(frame_bgr, (224, 224))
        return resized

    landmarks = kpts[:, :2]
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * 1.4)
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst_pts = np.array([[0, 0], [0, 223], [223, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    cropped = warp(frame_bgr, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    return cropped


def to_motion_batch(frames_rgb):
    arr = np.stack([cv2.resize(frame, (512, 512)) for frame in frames_rgb], axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)


def to_face_batch(frames_bgr):
    crops = np.stack([crop_face(frame) for frame in frames_bgr], axis=0)
    rgb = crops[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def extract_audio(video_path: Path, processor, model, target_len: int):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(wav_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        speech_array, sampling_rate = librosa.load(str(wav_path), sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.inference_mode():
            features = model(inputs.input_values.to(DEVICE)).last_hidden_state.float()
        features = F.interpolate(features.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return features.transpose(1, 2)[0].cpu().numpy().astype(np.float32)
    finally:
        if wav_path.exists():
            wav_path.unlink()


def extract_motion(frames_rgb, renderer):
    latents = []
    with torch.inference_mode():
        for start in range(0, len(frames_rgb), MOTION_BATCH):
            batch = to_motion_batch(frames_rgb[start:start + MOTION_BATCH]).to(DEVICE, dtype=DTYPE)
            latents.append(renderer.latent_token_encoder(batch).cpu())
    return torch.cat(latents, dim=0)


def extract_smirk(frames_bgr, smirk_model):
    pose_all, cam_all = [], []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), FACE_BATCH):
            batch = to_face_batch(frames_bgr[start:start + FACE_BATCH]).to(DEVICE, dtype=DTYPE)
            outputs = smirk_model(batch)
            pose_all.append(outputs["pose_params"].cpu())
            cam_all.append(outputs["cam"].cpu())
    return torch.cat(pose_all, dim=0), torch.cat(cam_all, dim=0)


def extract_l2cs(frames_bgr, l2cs_model, softmax, idx_tensor):
    preds = []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), FACE_BATCH):
            crops = np.stack([crop_face(frame)[:, :, ::-1] for frame in frames_bgr[start:start + FACE_BATCH]], axis=0)
            batch = prep_input_numpy(crops, DEVICE)
            gaze_pitch, gaze_yaw = l2cs_model(batch)
            pitch_pred = torch.sum(softmax(gaze_pitch) * idx_tensor, dim=1) * 4 - 180
            yaw_pred = torch.sum(softmax(gaze_yaw) * idx_tensor, dim=1) * 4 - 180
            pitch_pred = pitch_pred.cpu().numpy() * np.pi / 180.0
            yaw_pred = yaw_pred.cpu().numpy() * np.pi / 180.0
            preds.append(np.stack([pitch_pred, yaw_pred], axis=1).astype(np.float32))
    return np.concatenate(preds, axis=0)


def extract_openface3(frames_bgr, openface_model):
    aus = []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), FACE_BATCH):
            batch = to_face_batch(frames_bgr[start:start + FACE_BATCH]).to(DEVICE, dtype=DTYPE)
            _, _, au_output = openface_model(batch)
            au_output = torch.sigmoid(au_output).cpu().numpy().astype(np.float32)
            for row in au_output:
                expanded = np.zeros((len(AU_ORDER),), dtype=np.float32)
                for idx, au_name in OPENFACE3_TO_17.items():
                    expanded[AU_ORDER.index(au_name)] = float(np.clip(row[idx], 0.0, 1.0))
                aus.append(expanded)
    return np.stack(aus, axis=0)


def extract_cpp_openface(video_path: Path):
    if not CPP_OPENFACE_BIN or not Path(CPP_OPENFACE_BIN).exists():
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                CPP_OPENFACE_BIN,
                "-f", str(video_path),
                "-out_dir", tmpdir,
                "-2Dfp", "-aus", "-gaze",
                "-q",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        csv_path = Path(tmpdir) / f"{video_path.stem}.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        cols = [f"{au}_r" for au in AU_ORDER]
        if not all(col in df.columns for col in cols):
            return None
        values = df[cols].to_numpy(dtype=np.float32)
        values = np.clip(values / 5.0, 0.0, 1.0)
        return values


def read_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr, frames_rgb = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames_bgr, frames_rgb


def save_video_features(stem: str, motion, audio, pose, cam, gaze, au):
    paths = output_paths(stem)
    target_len = min(len(motion), len(audio), len(pose), len(cam), len(gaze), len(au))
    torch.save(motion[:target_len].float(), paths["motion"])
    np.save(paths["audio"], audio[:target_len].astype(np.float32))
    torch.save({"pose_params": pose[:target_len].float(), "cam": cam[:target_len].float()}, paths["smirk"])
    np.save(paths["gaze"], gaze[:target_len].astype(np.float32))
    np.save(paths["au"], au[:target_len].astype(np.float32))
    return target_len


def main():
    ensure_dirs()
    videos = iter_videos()
    print(f"[INFO] videos={len(videos)} device={DEVICE}")
    if not CPP_OPENFACE_BIN:
        print("[WARN] CPP_OPENFACE_BIN not set. Falling back to OpenFace-3.0 AU head, which only covers 8 AU channels directly.")

    renderer = make_renderer()
    audio_processor, audio_model = make_audio()
    smirk_model = make_smirk()
    l2cs_model, softmax, idx_tensor = make_l2cs()
    openface3_model = make_openface3()

    processed = 0
    skipped = 0
    failures = []

    for video_path in tqdm(videos, desc="RunPod exact prep"):
        stem = video_path.stem
        if not OVERWRITE and all_exist(stem):
            skipped += 1
            continue

        try:
            frames_bgr, frames_rgb = read_frames(video_path)
            if not frames_bgr:
                raise RuntimeError("no frames decoded")

            motion = extract_motion(frames_rgb, renderer)
            audio = extract_audio(video_path, audio_processor, audio_model, len(frames_bgr))
            pose, cam = extract_smirk(frames_bgr, smirk_model)
            gaze = extract_l2cs(frames_bgr, l2cs_model, softmax, idx_tensor)

            au = extract_cpp_openface(video_path)
            if au is None:
                au = extract_openface3(frames_bgr, openface3_model)

            target_len = save_video_features(stem, motion, audio, pose, cam, gaze, au)
            processed += 1
            if processed % 100 == 0:
                print(f"[INFO] processed={processed} last={stem} T={target_len}")
        except Exception as exc:
            failures.append((stem, str(exc)))
            print(f"[WARN] {stem}: {exc}")

    print("[SUMMARY]")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={len(failures)}")

    if failures:
        failure_log = OUTPUT_DIR / "runpod_exact_failures.tsv"
        with failure_log.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["stem", "reason"])
            writer.writerows(failures)
        print(f"failure_log={failure_log}")


if __name__ == "__main__":
    main()
PY
