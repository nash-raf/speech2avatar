#!/usr/bin/env python
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import librosa
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


AU_ORDER = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU25",
    "AU26",
    "AU45",
]

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUNTIME = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mead_root", type=str, default=str(REPO_ROOT.parent / "MEAD"))
    parser.add_argument("--output_root", type=str, default=str(REPO_ROOT.parent / "generator_dataset"))
    parser.add_argument("--checkpoints_dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--skip_motion", action="store_true")
    parser.add_argument("--motion_batch_size", type=int, default=16)
    return parser.parse_args()


def ensure_face_landmarker_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"
    if not model_path.exists() or model_path.stat().st_size < 1024:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    return model_path


def discover_videos(mead_root: Path) -> List[Path]:
    videos = []
    for split in ("train", "val", "test"):
        split_root = mead_root / split
        if not split_root.exists():
            continue
        videos.extend(sorted(split_root.rglob("*.mp4")))
    return videos


def ensure_output_dirs(output_root: Path, skip_motion: bool):
    for name in ("audio", "smirk", "gaze", "au"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    if not skip_motion:
        (output_root / "motion").mkdir(parents=True, exist_ok=True)


def output_paths(output_root: Path, stem: str, skip_motion: bool) -> Dict[str, Path]:
    paths = {
        "audio": output_root / "audio" / f"{stem}.npy",
        "smirk": output_root / "smirk" / f"{stem}.pt",
        "gaze": output_root / "gaze" / f"{stem}.npy",
        "au": output_root / "au" / f"{stem}.npy",
    }
    if not skip_motion:
        paths["motion"] = output_root / "motion" / f"{stem}.pt"
    return paths


def all_outputs_exist(paths: Dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


class AudioFeatureExtractor:
    def __init__(self, wav2vec_dir: Path):
        wav2vec_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(str(wav2vec_dir), local_files_only=True)
            self.model = Wav2Vec2Model.from_pretrained(str(wav2vec_dir), local_files_only=True)
        except Exception:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.processor.save_pretrained(wav2vec_dir)
            self.model.save_pretrained(wav2vec_dir)
        self.model.eval()

    def extract(self, video_path: Path, target_len: int) -> np.ndarray:
        if target_len <= 0:
            return np.zeros((0, 768), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            speech_array, sampling_rate = librosa.load(str(wav_path), sr=16000)
            if speech_array.size == 0:
                return np.zeros((target_len, 768), dtype=np.float32)
            inputs = self.processor(
                speech_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            with torch.inference_mode():
                features = self.model(inputs.input_values).last_hidden_state.float()
            features = F.interpolate(
                features.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)[0]
            return features.cpu().numpy().astype(np.float32)
        finally:
            if wav_path.exists():
                wav_path.unlink()


class MediaPipeConditionExtractor:
    def __init__(self, model_path: Path):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.vision = vision
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )

    @staticmethod
    def _blendshape_score(blendshapes: Dict[str, float], *names: str) -> float:
        if not names:
            return 0.0
        return float(np.mean([blendshapes.get(name, 0.0) for name in names]))

    @staticmethod
    def _landmark_xy(landmarks, index: int, width: int, height: int) -> np.ndarray:
        point = landmarks[index]
        return np.array([point.x * width, point.y * height], dtype=np.float32)

    @staticmethod
    def _safe_norm(value: np.ndarray) -> float:
        return float(np.linalg.norm(value) + 1e-6)

    def _estimate_gaze(self, landmarks, width: int, height: int) -> np.ndarray:
        def eye_gaze(iris_indices, outer_idx, inner_idx, top_idx, bottom_idx):
            iris_center = np.mean(
                [self._landmark_xy(landmarks, idx, width, height) for idx in iris_indices],
                axis=0,
            )
            outer = self._landmark_xy(landmarks, outer_idx, width, height)
            inner = self._landmark_xy(landmarks, inner_idx, width, height)
            top = self._landmark_xy(landmarks, top_idx, width, height)
            bottom = self._landmark_xy(landmarks, bottom_idx, width, height)
            eye_center = 0.5 * (outer + inner)
            horiz = (iris_center[0] - eye_center[0]) / self._safe_norm(inner - outer)
            vert = (iris_center[1] - 0.5 * (top[1] + bottom[1])) / (abs(bottom[1] - top[1]) + 1e-6)
            yaw = float(np.clip(math.atan(horiz * 4.0), -1.0, 1.0))
            pitch = float(np.clip(math.atan(vert * 4.0), -1.0, 1.0))
            return np.array([pitch, yaw], dtype=np.float32)

        left = eye_gaze([474, 475, 476, 477], 33, 133, 159, 145)
        right = eye_gaze([469, 470, 471, 472], 263, 362, 386, 374)
        return ((left + right) * 0.5).astype(np.float32)

    def _extract_pose_and_cam(self, result) -> (np.ndarray, np.ndarray):
        if not result.facial_transformation_matrixes:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = 0.0

        pose = np.array([pitch, yaw, roll], dtype=np.float32)
        cam = np.tanh(translation / 100.0).astype(np.float32)
        return pose, cam

    def _blendshapes_to_aus(self, landmarks, blendshapes: Dict[str, float], width: int, height: int) -> np.ndarray:
        mouth_left = self._landmark_xy(landmarks, 61, width, height)
        mouth_right = self._landmark_xy(landmarks, 291, width, height)
        upper_lip = self._landmark_xy(landmarks, 13, width, height)
        lower_lip = self._landmark_xy(landmarks, 14, width, height)
        lip_gap = np.linalg.norm(upper_lip - lower_lip) / self._safe_norm(mouth_right - mouth_left)
        lip_part = float(np.clip(lip_gap * 3.0, 0.0, 1.0))

        au_values = {
            "AU01": self._blendshape_score(blendshapes, "browInnerUp"),
            "AU02": self._blendshape_score(blendshapes, "browOuterUpLeft", "browOuterUpRight"),
            "AU04": self._blendshape_score(blendshapes, "browDownLeft", "browDownRight"),
            "AU05": self._blendshape_score(blendshapes, "eyeWideLeft", "eyeWideRight"),
            "AU06": self._blendshape_score(blendshapes, "cheekSquintLeft", "cheekSquintRight"),
            "AU07": self._blendshape_score(blendshapes, "eyeSquintLeft", "eyeSquintRight"),
            "AU09": self._blendshape_score(blendshapes, "noseSneerLeft", "noseSneerRight"),
            "AU10": self._blendshape_score(blendshapes, "mouthUpperUpLeft", "mouthUpperUpRight"),
            "AU12": self._blendshape_score(blendshapes, "mouthSmileLeft", "mouthSmileRight"),
            "AU14": self._blendshape_score(blendshapes, "mouthDimpleLeft", "mouthDimpleRight"),
            "AU15": self._blendshape_score(blendshapes, "mouthFrownLeft", "mouthFrownRight"),
            "AU17": self._blendshape_score(blendshapes, "mouthShrugLower"),
            "AU20": self._blendshape_score(blendshapes, "mouthStretchLeft", "mouthStretchRight"),
            "AU23": self._blendshape_score(blendshapes, "mouthPressLeft", "mouthPressRight"),
            "AU25": max(lip_part, self._blendshape_score(blendshapes, "jawOpen") * 0.5),
            "AU26": self._blendshape_score(blendshapes, "jawOpen"),
            "AU45": self._blendshape_score(blendshapes, "eyeBlinkLeft", "eyeBlinkRight"),
        }
        return np.array([np.clip(au_values[name], 0.0, 1.0) for name in AU_ORDER], dtype=np.float32)

    def extract(self, video_path: Path, collect_frames: bool = False) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps < 1.0 or fps > 240.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 512
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 512
        frame_interval_ms = max(1, int(round(1000.0 / fps)))

        frames = []
        poses = []
        cams = []
        gazes = []
        aus = []

        last_pose = np.zeros(3, dtype=np.float32)
        last_cam = np.zeros(3, dtype=np.float32)
        last_gaze = np.zeros(2, dtype=np.float32)
        last_au = np.zeros(len(AU_ORDER), dtype=np.float32)
        last_timestamp_ms = -1

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if collect_frames:
                frames.append(frame_rgb)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = max(last_timestamp_ms + 1, frame_idx * frame_interval_ms)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            last_timestamp_ms = timestamp_ms

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                blendshape_entries = result.face_blendshapes[0] if result.face_blendshapes else []
                if hasattr(blendshape_entries, "categories"):
                    blendshape_entries = blendshape_entries.categories
                blendshapes = {
                    category.category_name: float(category.score)
                    for category in blendshape_entries
                }
                last_pose, last_cam = self._extract_pose_and_cam(result)
                last_gaze = self._estimate_gaze(landmarks, width, height)
                last_au = self._blendshapes_to_aus(landmarks, blendshapes, width, height)

            poses.append(last_pose.copy())
            cams.append(last_cam.copy())
            gazes.append(last_gaze.copy())
            aus.append(last_au.copy())
            frame_idx += 1

        cap.release()

        return {
            "frame_count": frame_idx,
            "frames": frames,
            "pose": np.asarray(poses, dtype=np.float32),
            "cam": np.asarray(cams, dtype=np.float32),
            "gaze": np.asarray(gazes, dtype=np.float32),
            "au": np.asarray(aus, dtype=np.float32),
        }


class MotionExtractor:
    def __init__(self, checkpoints_dir: Path, device: str, batch_size: int):
        from generator.options.base_options import BaseOptions
        from renderer.models import IMTRenderer

        parser = argparse.ArgumentParser()
        opt = BaseOptions().initialize(parser).parse_args([])
        opt.rank = device
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.renderer = IMTRenderer(opt).to(self.device).eval()

        ckpt = torch.load(checkpoints_dir / "renderer.ckpt", map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        renderer_state = {
            key.replace("gen.", ""): value
            for key, value in state_dict.items()
            if key.startswith("gen.")
        }
        self.renderer.load_state_dict(renderer_state, strict=False)

    def extract(self, frames: List[np.ndarray]) -> torch.Tensor:
        if not frames:
            return torch.zeros((0, 32), dtype=torch.float32)

        latents = []
        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                chunk = frames[start:start + self.batch_size]
                tensor = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2).float() / 255.0
                tensor = tensor.to(self.device)
                latents.append(self.renderer.latent_token_encoder(tensor).cpu())
        return torch.cat(latents, dim=0)


class RuntimeContext:
    def __init__(self, config: Dict[str, object]):
        self.output_root = Path(config["output_root"])
        self.overwrite = bool(config["overwrite"])
        self.skip_motion = bool(config["skip_motion"])
        self.motion_batch_size = int(config["motion_batch_size"])
        self.audio_extractor = AudioFeatureExtractor(Path(config["checkpoints_dir"]) / "wav2vec2-base-960h")
        model_path = ensure_face_landmarker_model(Path(config["checkpoints_dir"]) / "mediapipe")
        self.visual_extractor = MediaPipeConditionExtractor(model_path)

        if self.skip_motion:
            self.motion_extractor = None
        else:
            device = config["motion_device"]
            self.motion_extractor = MotionExtractor(Path(config["checkpoints_dir"]), device, self.motion_batch_size)

    def process_video(self, video_path: Path) -> Dict[str, object]:
        stem = video_path.stem
        paths = output_paths(self.output_root, stem, self.skip_motion)

        if not self.overwrite and all_outputs_exist(paths):
            return {"status": "skipped", "stem": stem, "reason": "existing"}

        visual = self.visual_extractor.extract(video_path, collect_frames=not self.skip_motion)
        frame_count = int(visual["frame_count"])
        if frame_count == 0:
            raise RuntimeError("video produced zero frames")

        audio = self.audio_extractor.extract(video_path, frame_count)
        arrays = {
            "audio": audio,
            "pose": visual["pose"],
            "cam": visual["cam"],
            "gaze": visual["gaze"],
            "au": visual["au"],
        }

        if self.motion_extractor is not None:
            arrays["motion"] = self.motion_extractor.extract(visual["frames"])

        lengths = [len(arrays["audio"]), len(arrays["pose"]), len(arrays["cam"]), len(arrays["gaze"]), len(arrays["au"])]
        if "motion" in arrays:
            lengths.append(len(arrays["motion"]))
        target_len = min(lengths)
        if target_len <= 0:
            raise RuntimeError("no aligned frames available after extraction")

        audio = arrays["audio"][:target_len]
        pose = torch.from_numpy(arrays["pose"][:target_len]).float()
        cam = torch.from_numpy(arrays["cam"][:target_len]).float()
        gaze = arrays["gaze"][:target_len].astype(np.float32)
        au = arrays["au"][:target_len].astype(np.float32)

        np.save(paths["audio"], audio)
        torch.save({"pose_params": pose, "cam": cam}, paths["smirk"])
        np.save(paths["gaze"], gaze)
        np.save(paths["au"], au)

        if "motion" in arrays:
            torch.save(arrays["motion"][:target_len].float(), paths["motion"])

        return {"status": "processed", "stem": stem, "frames": target_len}


def init_worker(config: Dict[str, object]):
    global RUNTIME
    RUNTIME = RuntimeContext(config)


def process_video_task(video_path_str: str) -> Dict[str, object]:
    global RUNTIME
    video_path = Path(video_path_str)
    try:
        return RUNTIME.process_video(video_path)
    except Exception as exc:
        return {"status": "failed", "stem": video_path.stem, "reason": str(exc)}


def build_runtime_config(args) -> Dict[str, object]:
    if args.skip_motion:
        motion_device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("--skip_motion was not set and CUDA is unavailable for motion extraction.")
        motion_device = f"cuda:{args.gpu_id}"

    return {
        "output_root": args.output_root,
        "checkpoints_dir": args.checkpoints_dir,
        "overwrite": args.overwrite,
        "skip_motion": args.skip_motion,
        "motion_device": motion_device,
        "motion_batch_size": args.motion_batch_size,
    }


def main():
    args = parse_args()
    mead_root = Path(args.mead_root)
    output_root = Path(args.output_root)
    checkpoints_dir = Path(args.checkpoints_dir)

    if not mead_root.exists():
        raise FileNotFoundError(f"MEAD root not found: {mead_root}")

    ensure_output_dirs(output_root, args.skip_motion)

    videos = discover_videos(mead_root)
    if not videos:
        raise RuntimeError(f"No MP4 files found under {mead_root}")

    if not args.skip_motion and args.workers > 1:
        print("[INFO] Motion extraction uses a single process to avoid GPU contention.")
        args.workers = 1

    config = build_runtime_config(args)
    started = time.time()
    processed = 0
    skipped = 0
    failed = []

    print(f"[INFO] Found {len(videos)} videos under {mead_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Motion extraction enabled: {not args.skip_motion}")
    print(f"[INFO] Workers: {args.workers}")

    if args.workers == 1:
        init_worker(config)
        iterator = (process_video_task(str(video_path)) for video_path in videos)
        progress = tqdm(iterator, total=len(videos), desc="Preparing generator dataset")
        for result in progress:
            if result["status"] == "processed":
                processed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed.append(result)
                progress.write(f"[WARN] {result['stem']}: {result['reason']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,)) as executor:
            future_to_path = {executor.submit(process_video_task, str(video_path)): video_path for video_path in videos}
            progress = tqdm(total=len(videos), desc="Preparing generator dataset")
            for future in as_completed(future_to_path):
                result = future.result()
                if result["status"] == "processed":
                    processed += 1
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    failed.append(result)
                    progress.write(f"[WARN] {result['stem']}: {result['reason']}")
                progress.update(1)
            progress.close()

    elapsed = time.time() - started
    print("\n[SUMMARY]")
    print(f"total_videos={len(videos)}")
    print(f"processed={processed}")
    print(f"skipped_existing={skipped}")
    print(f"failed={len(failed)}")
    print(f"elapsed_sec={elapsed:.2f}")

    if failed:
        failure_log = output_root / "prepare_generator_dataset_failures.txt"
        with failure_log.open("w", encoding="utf-8") as handle:
            for item in failed:
                handle.write(f"{item['stem']}\t{item['reason']}\n")
        print(f"failure_log={failure_log}")


if __name__ == "__main__":
    main()
