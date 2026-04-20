#!/usr/bin/env python3
"""Compute landmark-based motion and jitter metrics for one or more videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from scipy.signal import savgol_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rendered talking-head videos with landmark-based metrics."
    )
    parser.add_argument(
        "--video",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Video label and path. Repeat for multiple videos.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to write the metrics JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="face-alignment device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.",
    )
    return parser.parse_args()


def read_video(path: str) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames_bgr, axis=0), float(fps)


def global_metrics(frames_bgr: np.ndarray) -> dict[str, float | int]:
    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_bgr], axis=0
    ).astype(np.float32)
    deltas = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    temporal_std = gray.std(axis=0).mean()
    return {
        "frames_loaded": int(len(frames_bgr)),
        "global_delta_mean": float(deltas.mean()),
        "global_delta_p95": float(np.percentile(deltas, 95)),
        "temporal_std_mean": float(temporal_std),
    }


def get_landmarks_sequence(
    fa: face_alignment.FaceAlignment, frames_bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lms = []
    valid_idx = []
    for i, frame in enumerate(frames_bgr):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks_from_image(rgb)
        if preds and len(preds) > 0:
            lms.append(preds[0].astype(np.float32))
            valid_idx.append(i)
    if not lms:
        raise RuntimeError("no landmarks detected")
    return np.stack(lms, axis=0), np.array(valid_idx, dtype=np.int32)


def eye_dist(lm: np.ndarray) -> float:
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    return float(np.linalg.norm(left_eye - right_eye) + 1e-6)


def mouth_opening(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[62] - lm[66]) / eye_dist(lm))


def mouth_width(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[48] - lm[54]) / eye_dist(lm))


def upper_face_jitter(lms: np.ndarray) -> dict[str, float]:
    idx = list(range(17, 48))
    pts = lms[:, idx, :]
    norms = np.array([eye_dist(lm) for lm in lms], dtype=np.float32)[:, None, None]
    pts = pts / norms

    if len(pts) >= 7:
        win = min(len(pts) // 2 * 2 - 1, 11)
        smooth = (
            savgol_filter(pts, window_length=win, polyorder=2, axis=0, mode="interp")
            if win >= 5
            else pts.copy()
        )
    else:
        smooth = pts.copy()

    residual = pts - smooth
    mag = np.linalg.norm(residual, axis=-1).mean(axis=-1)
    return {
        "jitter_proxy_mean": float(mag.mean()),
        "jitter_proxy_p95": float(np.percentile(mag, 95)),
    }


def landmark_motion(lms: np.ndarray, idxs: list[int]) -> dict[str, float]:
    pts = lms[:, idxs, :]
    norms = np.array([eye_dist(lm) for lm in lms], dtype=np.float32)[:, None, None]
    pts = pts / norms
    vel = np.linalg.norm(pts[1:] - pts[:-1], axis=-1).mean(axis=-1)
    if len(vel) == 0:
        return {"mean": 0.0, "p95": 0.0}
    return {
        "mean": float(vel.mean()),
        "p95": float(np.percentile(vel, 95)),
    }


def ratio(a: float, b: float) -> float:
    return float(a / (b + 1e-8))


def compare_against_first(results: dict[str, dict]) -> dict[str, dict]:
    labels = list(results.keys())
    if len(labels) < 2:
        return {}

    base_label = labels[0]
    base = results[base_label]
    summary = {}
    for label in labels[1:]:
        other = results[label]
        summary[f"{base_label}_vs_{label}"] = {
            "global_delta_ratio": ratio(
                float(base["global_delta_mean"]), float(other["global_delta_mean"])
            ),
            "mouth_delta_ratio": ratio(
                float(base["mouth_delta_mean"]), float(other["mouth_delta_mean"])
            ),
            "mouth_open_mean_ratio": ratio(
                float(base["mouth_open_mean"]), float(other["mouth_open_mean"])
            ),
            "mouth_open_p95_ratio": ratio(
                float(base["mouth_open_p95"]), float(other["mouth_open_p95"])
            ),
            "jitter_ratio": ratio(
                float(other["jitter_proxy_mean"]), float(base["jitter_proxy_mean"])
            ),
        }
        summary[f"{label}_vs_{base_label}"] = {
            "global_delta_ratio": ratio(
                float(other["global_delta_mean"]), float(base["global_delta_mean"])
            ),
            "mouth_delta_ratio": ratio(
                float(other["mouth_delta_mean"]), float(base["mouth_delta_mean"])
            ),
            "mouth_open_mean_ratio": ratio(
                float(other["mouth_open_mean"]), float(base["mouth_open_mean"])
            ),
            "mouth_open_p95_ratio": ratio(
                float(other["mouth_open_p95"]), float(base["mouth_open_p95"])
            ),
            "jitter_ratio": ratio(
                float(base["jitter_proxy_mean"]), float(other["jitter_proxy_mean"])
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False
    )

    results: dict[str, dict] = {}
    for label, path_str in args.video:
        path = Path(path_str).expanduser().resolve()
        frames_bgr, fps = read_video(str(path))
        gm = global_metrics(frames_bgr)
        lms, valid_idx = get_landmarks_sequence(fa, frames_bgr)

        mouth_open = np.array([mouth_opening(lm) for lm in lms], dtype=np.float32)
        mouth_w = np.array([mouth_width(lm) for lm in lms], dtype=np.float32)
        mouth_motion = landmark_motion(lms, list(range(48, 68)))
        upper_motion = landmark_motion(lms, list(range(17, 48)))
        jitter = upper_face_jitter(lms)

        results[label] = {
            "path": str(path),
            "fps": fps,
            "duration_sec": float(len(frames_bgr) / fps),
            **gm,
            "landmark_frames": int(len(lms)),
            "valid_frame_indices_count": int(len(valid_idx)),
            "mouth_open_mean": float(mouth_open.mean()),
            "mouth_open_std": float(mouth_open.std()),
            "mouth_open_p95": float(np.percentile(mouth_open, 95)),
            "mouth_open_max": float(mouth_open.max()),
            "mouth_width_mean": float(mouth_w.mean()),
            "mouth_delta_mean": mouth_motion["mean"],
            "mouth_delta_p95": mouth_motion["p95"],
            "upper_face_motion_mean": upper_motion["mean"],
            "upper_face_motion_p95": upper_motion["p95"],
            **jitter,
        }

    payload = {
        "videos": results,
        "summary": compare_against_first(results),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
