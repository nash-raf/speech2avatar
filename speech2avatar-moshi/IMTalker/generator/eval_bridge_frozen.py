import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import face_alignment
import librosa
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

from generator.FM import FMGenerator
from generator.options.base_options import BaseOptions
from renderer.models import IMTRenderer


def default_moshi_repo() -> str:
    explicit = Path("/home/user/D/speech2avatar/speech2avatar-moshi/moshi")
    if explicit.exists():
        return str(explicit)
    sibling = Path(__file__).resolve().parents[2] / "moshi"
    return str(sibling)


class ReferenceDataProcessor:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

    @torch.no_grad()
    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        bboxes = self.fa.face_detector.detect_from_image(img_arr)
        valid_bboxes = [
            (int(x1), int(y1), int(x2), int(y2), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        if not valid_bboxes:
            raise ValueError("No face detected in the reference image.")

        x1, y1, x2, y2, _ = valid_bboxes[0]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = int((x2 - x1) * 0.8)
        half_h = int((y2 - y1) * 0.8)
        half = max(half_w, half_h)

        x1_new = max(cx - half, 0)
        x2_new = min(cx + half, w)
        y1_new = max(cy - half, 0)
        y2_new = min(cy + half, h)

        side = min(x2_new - x1_new, y2_new - y1_new)
        x2_new = x1_new + side
        y2_new = y1_new + side
        crop_img = img_arr[y1_new:y2_new, x1_new:x2_new]
        return Image.fromarray(crop_img)

    def default_img_loader(self, path: str) -> Image.Image:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class EvalOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--resume_ckpt", required=True, type=str)
        parser.add_argument("--renderer_path", required=True, type=str)
        parser.add_argument("--ref_path", required=True, type=str)
        parser.add_argument("--aud_path", type=str, default=None)
        parser.add_argument("--aud_feat_path", type=str, default=None)
        parser.add_argument("--smirk_path", type=str, default=None)
        parser.add_argument("--gaze_path", type=str, default=None)
        parser.add_argument("--static_pose_zero", action="store_true")
        parser.add_argument("--static_gaze_zero", action="store_true")
        parser.add_argument("--init_prev_from_ref", action="store_true", default=True)
        parser.add_argument("--no_init_prev_from_ref", dest="init_prev_from_ref", action="store_false")
        parser.add_argument("--ref_blend_alpha", type=float, default=1.0)
        parser.add_argument("--output_path", required=True, type=str)
        parser.add_argument("--metrics_json_path", required=True, type=str)
        parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
        parser.add_argument("--crop", action="store_true")
        parser.add_argument("--render_batch_size", default=4, type=int)
        parser.add_argument("--use_ema", action="store_true", default=True)
        parser.add_argument("--no_use_ema", dest="use_ema", action="store_false")
        parser.add_argument("--moshi_repo", type=str, default=default_moshi_repo())
        parser.add_argument("--mimi_hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
        return parser


def ensure_moshi_importable(moshi_repo: str) -> None:
    repo = Path(moshi_repo)
    pkg_root = repo / "moshi"
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


def load_generator_weights(model: FMGenerator, checkpoint_path: str, use_ema: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]

    candidates = [raw_state]
    for prefix in ("model.", "student.", "teacher."):
        stripped = {k[len(prefix) :]: v for k, v in raw_state.items() if k.startswith(prefix)}
        if stripped:
            candidates.append(stripped)

    model_state = model.state_dict()
    best_state = None
    best_matches = -1
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        match_count = sum(
            1 for k, v in candidate.items() if k in model_state and model_state[k].shape == v.shape
        )
        if match_count > best_matches:
            best_matches = match_count
            best_state = candidate

    if best_state is None or best_matches <= 0:
        raise RuntimeError(f"Could not find compatible generator weights in {checkpoint_path}")

    if use_ema and isinstance(checkpoint.get("ema_state_dict"), dict):
        merged = dict(best_state)
        for k, v in checkpoint["ema_state_dict"].items():
            if k in model_state and model_state[k].shape == v.shape:
                merged[k] = v
        best_state = merged

    loadable = {k: v for k, v in best_state.items() if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    loaded_prefixes = sorted({k.split(".")[0] for k in loadable.keys()})
    fresh_prefixes = sorted({k.split(".")[0] for k in model_state.keys()} - set(loaded_prefixes))
    print(f"[INFO] generator loaded from {checkpoint_path}: {len(loadable)} params")
    print(f"[INFO] Pretrained modules loaded: {loaded_prefixes}")
    print(f"[INFO] Modules left at random init: {fresh_prefixes}")
    if missing:
        print(f"[WARNING] Missing generator keys: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected generator keys: {unexpected}")


def load_renderer(renderer: IMTRenderer, checkpoint_path: str) -> None:
    renderer_ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    ae_state_dict = {k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")}
    renderer.load_state_dict(ae_state_dict, strict=False)


@torch.no_grad()
def prepare_reference(
    renderer: IMTRenderer,
    processor: ReferenceDataProcessor,
    ref_path: str,
    device: torch.device,
    crop: bool,
):
    image = processor.default_img_loader(ref_path)
    if crop:
        image = processor.process_img(image)
    s_tensor = processor.transform(image).unsqueeze(0).to(device)
    f_r, g_r = renderer.dense_feature_encoder(s_tensor)
    ref_x = renderer.latent_token_encoder(s_tensor)
    return f_r, g_r, ref_x


@torch.no_grad()
def decode_sample_to_frames(
    renderer: IMTRenderer,
    ref_x: torch.Tensor,
    f_r,
    g_r,
    sample: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    total_frames = sample.shape[1]
    ta_r = renderer.adapt(ref_x, g_r)
    m_r = renderer.latent_token_decoder(ta_r)

    g_r_exp = g_r.expand(total_frames, -1)
    sample_flat = sample.squeeze(0)
    ta_c_all = renderer.adapt(sample_flat, g_r_exp)
    m_c_all = renderer.latent_token_decoder(ta_c_all)

    all_frames = []
    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        bs = end - start
        m_c_batch = tuple(m[start:end] for m in m_c_all)
        m_r_batch = tuple(m.expand(bs, -1, -1, -1) for m in m_r)
        f_r_batch = [f.expand(bs, -1, -1, -1) for f in f_r]
        all_frames.append(renderer.decode(m_c_batch, m_r_batch, f_r_batch).clone())
    return torch.cat(all_frames, dim=0)


def save_video(frames: torch.Tensor, output_path: str, audio_path: str | None, fps: float) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frames_uint8 = frames.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu()

    if audio_path:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = Path(tmp.name)
        torchvision.io.write_video(str(temp_path), frames_hwc, fps=fps)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_path),
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.unlink(missing_ok=True)
    else:
        torchvision.io.write_video(str(output), frames_hwc, fps=fps)


def load_precomputed_features(path: str) -> torch.Tensor:
    feats = np.load(path)
    return torch.from_numpy(feats).float()


def load_raw_audio(path: str, sampling_rate: int) -> torch.Tensor:
    speech_array, _ = librosa.load(path, sr=sampling_rate)
    return torch.from_numpy(speech_array).float().unsqueeze(0)


def load_smirk_params(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    smirk_data = torch.load(path, map_location="cpu")
    pose = smirk_data["pose_params"].float().to(device)
    cam = smirk_data["cam"].float().to(device)
    return pose, cam


def load_gaze_sequence(path: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float().to(device)


def estimate_target_frames(opt, a_feat: torch.Tensor | None = None) -> int:
    if a_feat is not None:
        return int(a_feat.shape[0])
    if not opt.aud_path:
        raise ValueError("Cannot infer target frame count without --aud_path or precomputed features.")
    duration_sec = librosa.get_duration(path=opt.aud_path)
    return max(1, int(round(duration_sec * opt.fps)))


def align_condition_sequence(sequence: torch.Tensor, target_frames: int) -> torch.Tensor:
    sequence = sequence.float()
    if sequence.ndim != 2:
        raise ValueError(f"Expected condition sequence of shape (T, D), got {tuple(sequence.shape)}")
    if sequence.shape[0] == target_frames:
        return sequence
    sequence_t = sequence.transpose(0, 1).unsqueeze(0)
    aligned = torch.nn.functional.interpolate(
        sequence_t,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return aligned.squeeze(0).transpose(0, 1).contiguous()


def build_condition_inputs(opt, device: torch.device, target_frames: int) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    pose = None
    cam = None
    gaze = None

    if opt.smirk_path:
        raw_pose, raw_cam = load_smirk_params(opt.smirk_path, device)
        cam = align_condition_sequence(raw_cam, target_frames)
        if not opt.static_pose_zero:
            pose = align_condition_sequence(raw_pose, target_frames)

    if opt.static_pose_zero:
        pose = torch.zeros(target_frames, 3, dtype=torch.float32, device=device)

    if opt.gaze_path:
        gaze = align_condition_sequence(load_gaze_sequence(opt.gaze_path, device), target_frames)
    if opt.static_gaze_zero:
        gaze = torch.zeros(target_frames, 2, dtype=torch.float32, device=device)

    return gaze, pose, cam


def build_initial_stream_state(
    fm: FMGenerator,
    ref_x: torch.Tensor,
    a_proj: torch.Tensor,
    gaze: torch.Tensor | None,
    pose: torch.Tensor | None,
    cam: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    device = ref_x.device
    num_prev = fm.num_prev_frames
    batch_size = ref_x.shape[0]

    def expand_first_frame(projected: torch.Tensor) -> torch.Tensor:
        return projected[:, :1, :].expand(batch_size, num_prev, -1).clone()

    def project_prev(tensor: torch.Tensor | None, projection: torch.nn.Module) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, num_prev, fm.opt.dim_c, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        projected = projection(tensor[:, :1, :])
        return projected.expand(batch_size, num_prev, -1).clone()

    return {
        "prev_sample": ref_x.unsqueeze(1).expand(batch_size, num_prev, -1).clone(),
        "prev_a": expand_first_frame(a_proj.to(device)),
        "prev_gaze": project_prev(gaze, fm.gaze_projection),
        "prev_pose": project_prev(pose, fm.pose_projection),
        "prev_cam": project_prev(cam, fm.cam_projection),
    }


@torch.no_grad()
def extract_mimi_features(path: str, opt) -> torch.Tensor:
    """Extract Mimi continuous latents and interpolate to the model frame rate.

    Mimi emits latents at 12.5 Hz natively. The precomputed audio_mimi/*.npy
    files on disk have already been interpolated to 25 Hz (opt.fps). Any
    online fallback has to do the same, otherwise the generator will see
    features at half the expected temporal rate and any Phase 1 / Phase 2
    verdict will be contaminated.
    """
    ensure_moshi_importable(opt.moshi_repo)
    from moshi.models import loaders
    import torchaudio

    checkpoint = loaders.CheckpointInfo.from_hf_repo(opt.mimi_hf_repo)
    mimi = checkpoint.get_mimi(device=opt.device)
    mimi.eval()

    wav, sample_rate = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != mimi.sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, mimi.sample_rate)
    # latent: (D, T_mimi) after [0] drops batch dim
    latent = mimi.encode_to_latent(wav.unsqueeze(0).to(opt.device), quantize=False)[0]

    # Mimi frame rate (Hz). Prefer the attribute if Moshi exposes it, else 12.5.
    mimi_fps = float(getattr(mimi, "frame_rate", 12.5))
    target_fps = float(opt.fps)
    num_mimi_frames = latent.shape[-1]
    audio_seconds = num_mimi_frames / mimi_fps
    target_frames = max(1, int(round(audio_seconds * target_fps)))

    if target_frames != num_mimi_frames:
        # linear interpolation along the time axis
        latent = torch.nn.functional.interpolate(
            latent.unsqueeze(0),  # (1, D, T_mimi)
            size=target_frames,
            mode="linear",
            align_corners=False,
        )[0]

    return latent.transpose(0, 1).contiguous().cpu()  # (T, D)


def run_metric_tool(video_path: str, metrics_json_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "eval_landmark_metrics.py"
    cmd = [
        sys.executable,
        str(tool_path),
        "--video",
        "candidate",
        video_path,
        "--output_json",
        metrics_json_path,
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(Path(metrics_json_path).read_text())
    candidate = payload["videos"]["candidate"]
    candidate["mouth_open_max_over_mean"] = float(
        candidate["mouth_open_max"] / (candidate["mouth_open_mean"] + 1e-8)
    )
    Path(metrics_json_path).write_text(json.dumps(payload, indent=2))


def main() -> None:
    opt = EvalOptions().parse()
    opt.rank = opt.device
    device = torch.device(opt.device)

    fm = FMGenerator(opt).to(device).eval()
    renderer = IMTRenderer(opt).to(device).eval()
    load_generator_weights(fm, opt.resume_ckpt, use_ema=opt.use_ema)
    if opt.bridge_ckpt:
        adapter_sd = torch.load(opt.bridge_ckpt, map_location="cpu")
        if isinstance(adapter_sd, dict) and "state_dict" in adapter_sd and isinstance(adapter_sd["state_dict"], dict):
            adapter_sd = adapter_sd["state_dict"]
        missing, unexpected = fm.audio_adapter.load_state_dict(adapter_sd, strict=True)
        print(
            f"[INFO] Loaded bridge ckpt from {opt.bridge_ckpt}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    load_renderer(renderer, opt.renderer_path)
    processor = ReferenceDataProcessor()
    f_r, g_r, ref_x = prepare_reference(renderer, processor, opt.ref_path, device, opt.crop)

    initial_stream_state = None
    if opt.aud_feat_path:
        a_feat = load_precomputed_features(opt.aud_feat_path)
        gaze, pose, cam = build_condition_inputs(opt, device, estimate_target_frames(opt, a_feat))
        a_proj = fm._project_audio(a_feat.unsqueeze(0).to(device))
        if opt.init_prev_from_ref:
            initial_stream_state = build_initial_stream_state(fm, ref_x, a_proj, gaze, pose, cam)
        sample = fm.sample(
            {"a_feat": a_feat, "ref_x": ref_x, "gaze": gaze, "pose": pose, "cam": cam},
            a_cfg_scale=opt.a_cfg_scale,
            nfe=opt.nfe,
            seed=opt.seed,
            stream_state=initial_stream_state,
        )
    else:
        if not opt.aud_path:
            raise ValueError("Either --aud_feat_path or --aud_path must be provided.")
        use_mimi_online = (
            opt.audio_feat_dim == 512
            or opt.audio_adapter_mode == "bridge_to_768"
            or "mimi" in str(opt.audio_subdir)
            or "rt_aligned" in str(opt.audio_subdir)
        )
        if use_mimi_online:
            a_feat = extract_mimi_features(opt.aud_path, opt)
            gaze, pose, cam = build_condition_inputs(opt, device, estimate_target_frames(opt, a_feat))
            a_proj = fm._project_audio(a_feat.unsqueeze(0).to(device))
            if opt.init_prev_from_ref:
                initial_stream_state = build_initial_stream_state(fm, ref_x, a_proj, gaze, pose, cam)
            sample = fm.sample(
                {"a_feat": a_feat, "ref_x": ref_x, "gaze": gaze, "pose": pose, "cam": cam},
                a_cfg_scale=opt.a_cfg_scale,
                nfe=opt.nfe,
                seed=opt.seed,
                stream_state=initial_stream_state,
            )
        else:
            audio = load_raw_audio(opt.aud_path, opt.sampling_rate).to(device)
            gaze, pose, cam = build_condition_inputs(opt, device, estimate_target_frames(opt))
            seq_len = max(1, int(round(librosa.get_duration(path=opt.aud_path) * opt.fps)))
            a_proj = fm._project_audio(fm.audio_encoder.inference(audio, seq_len=seq_len))
            if opt.init_prev_from_ref:
                initial_stream_state = build_initial_stream_state(fm, ref_x, a_proj, gaze, pose, cam)
            sample = fm.sample(
                {"a": audio, "ref_x": ref_x, "gaze": gaze, "pose": pose, "cam": cam},
                a_cfg_scale=opt.a_cfg_scale,
                nfe=opt.nfe,
                seed=opt.seed,
                stream_state=initial_stream_state,
            )

    if opt.ref_blend_alpha < 1.0:
        ref_seq = ref_x.unsqueeze(1).expand(-1, sample.shape[1], -1)
        sample = ref_seq + float(opt.ref_blend_alpha) * (sample - ref_seq)

    frames = decode_sample_to_frames(
        renderer,
        ref_x,
        f_r,
        g_r,
        sample,
        batch_size=opt.render_batch_size,
    ).cpu()
    save_video(frames, opt.output_path, opt.aud_path, fps=opt.fps)
    run_metric_tool(opt.output_path, opt.metrics_json_path)
    print(f"Saved video: {Path(opt.output_path).resolve()}")
    print(f"Saved metrics: {Path(opt.metrics_json_path).resolve()}")


if __name__ == "__main__":
    main()
