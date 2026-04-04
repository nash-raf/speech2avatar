#!/usr/bin/env python
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["motion_audio", "smirk", "gaze", "au"], required=True)
    parser.add_argument("--root_dir", default=os.environ.get("ROOT_DIR", "/workspace"))
    parser.add_argument("--s2a_dir", default=os.environ.get("S2A_DIR"))
    parser.add_argument("--mead_dir", default=os.environ.get("MEAD_DIR"))
    parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--checkpoint_dir", default=os.environ.get("CHECKPOINT_DIR"))
    parser.add_argument("--smirk_dir", default=os.environ.get("SMIRK_DIR"))
    parser.add_argument("--l2cs_dir", default=os.environ.get("L2CS_DIR"))
    parser.add_argument("--openface3_dir", default=os.environ.get("OPENFACE3_DIR"))
    parser.add_argument("--model_dir", default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--gpu_id", type=int, default=int(os.environ.get("GPU_ID", "0")))
    parser.add_argument("--motion_batch", type=int, default=int(os.environ.get("MOTION_BATCH", "32")))
    parser.add_argument("--face_batch", type=int, default=int(os.environ.get("FACE_BATCH", "64")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    parser.add_argument("--cpp_openface_bin", default=os.environ.get("CPP_OPENFACE_BIN", ""))
    return parser.parse_args()


ARGS = parse_args()
ROOT_DIR = Path(ARGS.root_dir)
S2A_DIR = Path(ARGS.s2a_dir or ROOT_DIR / "speech2avatar")
MEAD_DIR = Path(ARGS.mead_dir or ROOT_DIR / "MEAD")
OUTPUT_DIR = Path(ARGS.output_dir or ROOT_DIR / "generator_dataset")
CHECKPOINT_DIR = Path(ARGS.checkpoint_dir or S2A_DIR / "checkpoints")
SMIRK_DIR = Path(ARGS.smirk_dir or ROOT_DIR / "external_tools" / "smirk")
L2CS_DIR = Path(ARGS.l2cs_dir or ROOT_DIR / "external_tools" / "L2CS-Net")
OPENFACE3_DIR = Path(ARGS.openface3_dir or ROOT_DIR / "external_tools" / "OpenFace-3.0")
MODEL_DIR = Path(ARGS.model_dir or ROOT_DIR / "model_cache")

sys.path.insert(0, str(S2A_DIR))
sys.path.insert(0, str(SMIRK_DIR))
sys.path.insert(0, str(L2CS_DIR))
sys.path.insert(0, str(OPENFACE3_DIR))

from renderer.models import IMTRenderer
from generator.options.base_options import BaseOptions

DEVICE = torch.device(f"cuda:{ARGS.gpu_id}" if torch.cuda.is_available() else "cpu")
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
    stage_to_dirs = {
        "motion_audio": ("motion", "audio"),
        "smirk": ("smirk",),
        "gaze": ("gaze",),
        "au": ("au",),
    }
    for name in stage_to_dirs[ARGS.stage]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def iter_videos():
    videos = []
    for split in ("train", "val", "test"):
        split_root = MEAD_DIR / split
        if split_root.exists():
            videos.extend(sorted(split_root.rglob("*.mp4")))
    if ARGS.limit > 0:
        videos = videos[:ARGS.limit]
    return videos


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


def crop_face(frame_bgr):
    from skimage.transform import estimate_transform, warp
    from utils.mediapipe_utils import run_mediapipe

    kpts = run_mediapipe(frame_bgr)
    if kpts is None:
        return cv2.resize(frame_bgr, (224, 224))

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
    return warp(frame_bgr, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)


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


def make_renderer():
    parser = BaseOptions().initialize(argparse.ArgumentParser())
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
    return processor, model.to(DEVICE).eval()


def make_smirk():
    from src.smirk_encoder import SmirkEncoder

    model = SmirkEncoder().to(DEVICE).eval()
    checkpoint = torch.load(MODEL_DIR / "SMIRK_em1.pt", map_location="cpu")
    checkpoint_encoder = {k.replace("smirk_encoder.", ""): v for k, v in checkpoint.items() if "smirk_encoder" in k}
    model.load_state_dict(checkpoint_encoder, strict=False)
    return model


def make_l2cs():
    from l2cs.utils import getArch

    model = getArch("ResNet50", 90)
    model.load_state_dict(torch.load(MODEL_DIR / "L2CSNet_gaze360.pkl", map_location="cpu"))
    model = model.to(DEVICE).eval()
    softmax = torch.nn.Softmax(dim=1)
    idx_tensor = torch.arange(90, dtype=torch.float32, device=DEVICE)
    return model, softmax, idx_tensor


def make_openface3():
    from model.MLT import MLT

    model = MLT().to(DEVICE).eval()
    weights_path = OPENFACE3_DIR / "weights" / "stage2_epoch_7_loss_1.1606_acc_0.5589.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    return model


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
        for start in range(0, len(frames_rgb), ARGS.motion_batch):
            batch = to_motion_batch(frames_rgb[start:start + ARGS.motion_batch]).to(DEVICE, dtype=DTYPE)
            latents.append(renderer.latent_token_encoder(batch).cpu())
    return torch.cat(latents, dim=0)


def extract_smirk(frames_bgr, smirk_model):
    pose_all, cam_all = [], []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), ARGS.face_batch):
            batch = to_face_batch(frames_bgr[start:start + ARGS.face_batch]).to(DEVICE, dtype=DTYPE)
            outputs = smirk_model(batch)
            pose_all.append(outputs["pose_params"].cpu())
            cam_all.append(outputs["cam"].cpu())
    return torch.cat(pose_all, dim=0), torch.cat(cam_all, dim=0)


def extract_l2cs(frames_bgr, l2cs_model, softmax, idx_tensor):
    from l2cs.utils import prep_input_numpy

    preds = []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), ARGS.face_batch):
            crops = np.stack([crop_face(frame)[:, :, ::-1] for frame in frames_bgr[start:start + ARGS.face_batch]], axis=0)
            batch = prep_input_numpy(crops, DEVICE)
            gaze_pitch, gaze_yaw = l2cs_model(batch)
            pitch_pred = torch.sum(softmax(gaze_pitch) * idx_tensor, dim=1) * 4 - 180
            yaw_pred = torch.sum(softmax(gaze_yaw) * idx_tensor, dim=1) * 4 - 180
            pitch_pred = pitch_pred.cpu().numpy() * np.pi / 180.0
            yaw_pred = yaw_pred.cpu().numpy() * np.pi / 180.0
            preds.append(np.stack([pitch_pred, yaw_pred], axis=1).astype(np.float32))
    return np.concatenate(preds, axis=0)


def extract_cpp_openface(video_path: Path):
    import pandas as pd

    cpp_bin = ARGS.cpp_openface_bin.strip()
    if not cpp_bin or not Path(cpp_bin).exists():
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [cpp_bin, "-f", str(video_path), "-out_dir", tmpdir, "-2Dfp", "-aus", "-gaze", "-q"],
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
        return np.clip(df[cols].to_numpy(dtype=np.float32) / 5.0, 0.0, 1.0)


def extract_openface3(frames_bgr, openface_model):
    aus = []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), ARGS.face_batch):
            batch = to_face_batch(frames_bgr[start:start + ARGS.face_batch]).to(DEVICE, dtype=DTYPE)
            _, _, au_output = openface_model(batch)
            au_output = torch.sigmoid(au_output).cpu().numpy().astype(np.float32)
            for row in au_output:
                expanded = np.zeros((len(AU_ORDER),), dtype=np.float32)
                for idx, au_name in OPENFACE3_TO_17.items():
                    expanded[AU_ORDER.index(au_name)] = float(np.clip(row[idx], 0.0, 1.0))
                aus.append(expanded)
    return np.stack(aus, axis=0)


def save_failure_log(failures, stem_name):
    if not failures:
        return
    failure_log = OUTPUT_DIR / f"{stem_name}_failures.tsv"
    with failure_log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["stem", "reason"])
        writer.writerows(failures)


def main():
    ensure_dirs()
    videos = iter_videos()
    print(f"[INFO] stage={ARGS.stage} videos={len(videos)} device={DEVICE}")

    stage_models = {}
    if ARGS.stage == "motion_audio":
        stage_models["renderer"] = make_renderer()
        stage_models["audio"] = make_audio()
    elif ARGS.stage == "smirk":
        stage_models["smirk"] = make_smirk()
    elif ARGS.stage == "gaze":
        stage_models["l2cs"] = make_l2cs()
    elif ARGS.stage == "au":
        stage_models["openface3"] = make_openface3()
        if not ARGS.cpp_openface_bin:
            print("[WARN] CPP_OPENFACE_BIN not set. Falling back to OpenFace-3.0 AU head.")

    processed = 0
    skipped = 0
    failures = []

    for video_path in tqdm(videos, desc=f"RunPod {ARGS.stage}"):
        stem = video_path.stem
        try:
            if ARGS.stage == "motion_audio":
                motion_path = OUTPUT_DIR / "motion" / f"{stem}.pt"
                audio_path = OUTPUT_DIR / "audio" / f"{stem}.npy"
                if not ARGS.overwrite and motion_path.exists() and audio_path.exists():
                    skipped += 1
                    continue
                frames_bgr, frames_rgb = read_frames(video_path)
                if not frames_bgr:
                    raise RuntimeError("no frames decoded")
                motion = extract_motion(frames_rgb, stage_models["renderer"])
                processor, audio_model = stage_models["audio"]
                audio = extract_audio(video_path, processor, audio_model, len(frames_bgr))
                target_len = min(len(motion), len(audio))
                torch.save(motion[:target_len].float(), motion_path)
                np.save(audio_path, audio[:target_len].astype(np.float32))

            elif ARGS.stage == "smirk":
                smirk_path = OUTPUT_DIR / "smirk" / f"{stem}.pt"
                if not ARGS.overwrite and smirk_path.exists():
                    skipped += 1
                    continue
                frames_bgr, _ = read_frames(video_path)
                if not frames_bgr:
                    raise RuntimeError("no frames decoded")
                pose, cam = extract_smirk(frames_bgr, stage_models["smirk"])
                torch.save({"pose_params": pose.float(), "cam": cam.float()}, smirk_path)

            elif ARGS.stage == "gaze":
                gaze_path = OUTPUT_DIR / "gaze" / f"{stem}.npy"
                if not ARGS.overwrite and gaze_path.exists():
                    skipped += 1
                    continue
                frames_bgr, _ = read_frames(video_path)
                if not frames_bgr:
                    raise RuntimeError("no frames decoded")
                model, softmax, idx_tensor = stage_models["l2cs"]
                gaze = extract_l2cs(frames_bgr, model, softmax, idx_tensor)
                np.save(gaze_path, gaze.astype(np.float32))

            elif ARGS.stage == "au":
                au_path = OUTPUT_DIR / "au" / f"{stem}.npy"
                if not ARGS.overwrite and au_path.exists():
                    skipped += 1
                    continue
                frames_bgr, _ = read_frames(video_path)
                if not frames_bgr:
                    raise RuntimeError("no frames decoded")
                au = extract_cpp_openface(video_path)
                if au is None:
                    au = extract_openface3(frames_bgr, stage_models["openface3"])
                np.save(au_path, au.astype(np.float32))

            processed += 1
        except Exception as exc:
            failures.append((stem, str(exc)))
            print(f"[WARN] {stem}: {exc}")

    print("[SUMMARY]")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={len(failures)}")
    save_failure_log(failures, ARGS.stage)


if __name__ == "__main__":
    main()
