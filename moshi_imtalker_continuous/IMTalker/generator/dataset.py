import math
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def load_pose(smirk):
    pose = smirk["pose_params"]  # (N, 3)
    cam = smirk["cam"]           # (N, 3)
    return pose, cam


def generator_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in {"a_now", "a_prev"}:
            collated[key] = pad_sequence(values, batch_first=True)
            collated[f"{key}_len"] = torch.tensor([value.shape[0] for value in values], dtype=torch.long)
        else:
            collated[key] = torch.stack(values, dim=0)
    return collated


class AudioMotionSmirkGazeDataset(Dataset):
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames
        self.audio_token_rate = opt.audio_token_rate

        root_path = Path(opt.dataset_path)
        motion_dir = root_path / "motion"
        audio_dir = root_path / opt.audio_subdir
        smirk_dir = root_path / "smirk"
        gaze_dir = root_path / "gaze"

        motion_files = sorted(list(motion_dir.glob("*.pt")))
        motion_files = motion_files[start:end]

        self.samples = []
        for motion_path in tqdm(motion_files, desc="Filtering valid samples"):
            file_stem = motion_path.stem
            audio_path = audio_dir / f"{file_stem}.npy"
            gaze_path = gaze_dir / f"{file_stem}.npy"
            smirk_path = smirk_dir / f"{file_stem}.pt"

            if not (audio_path.exists() and gaze_path.exists() and smirk_path.exists()):
                continue

            try:
                motion = torch.load(motion_path)
                smirk = torch.load(smirk_path)
                audio = np.load(audio_path, mmap_mode='r')
                gaze = np.load(gaze_path, mmap_mode='r')

                motion_len = len(motion)
                gaze_len = len(gaze)
                smirk_len = len(smirk["pose_params"])
                audio_len = self._tokens_to_num_frames(len(audio))

                min_len = min(motion_len, audio_len, gaze_len, smirk_len)

                if min_len >= self.required_len:
                    self.samples.append({
                        "motion_path": str(motion_path),
                        "audio_path": str(audio_path),
                        "smirk_path": str(smirk_path),
                        "gaze_path": str(gaze_path)
                    })
            except Exception as e:
                print(f"[Warning] Error checking file {file_stem}: {e}")
                continue

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root_path}")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

    def _tokens_to_num_frames(self, token_count):
        return int(math.floor(token_count * self.opt.fps / self.audio_token_rate))

    def _frame_to_token_index(self, frame_idx, use_ceil=False):
        token_idx = frame_idx * self.audio_token_rate / self.opt.fps
        return int(math.ceil(token_idx) if use_ceil else math.floor(token_idx))

    def _slice_audio_tokens(self, audio, start_frame, end_frame):
        start_token = self._frame_to_token_index(start_frame, use_ceil=False)
        end_token = self._frame_to_token_index(end_frame, use_ceil=True)
        end_token = min(len(audio), max(start_token + 1, end_token))
        start_token = min(start_token, end_token - 1)
        audio_slice = np.asarray(audio[start_token:end_token]).copy()
        return torch.from_numpy(audio_slice).float()

    def __len__(self):
        return len(self.samples)

    def _get_full_clip(self, index):
        item = self.samples[index]
        
        motion = torch.load(item['motion_path'])
        audio = np.load(item['audio_path'], mmap_mode='r')
        gaze = np.load(item['gaze_path'], mmap_mode='r')
        smirk = torch.load(item['smirk_path'])
        pose, cam = load_pose(smirk)

        min_len = min(len(motion), self._tokens_to_num_frames(len(audio)), len(gaze), len(pose))
        start_idx = random.randint(0, min_len - self.required_len)
        end_idx = start_idx + self.required_len
        split_idx = start_idx + self.num_prev_frames

        motion_seg = motion[start_idx:end_idx]
        gaze_seg = torch.from_numpy(gaze[start_idx:end_idx].copy()).float()
        pose_seg = pose[start_idx:end_idx]
        cam_seg = cam[start_idx:end_idx]

        audio_prev = self._slice_audio_tokens(audio, start_idx, split_idx)
        audio_clip = self._slice_audio_tokens(audio, split_idx, end_idx)

        motion_prev = motion_seg[:self.num_prev_frames]
        gaze_prev = gaze_seg[:self.num_prev_frames]
        pose_prev = pose_seg[:self.num_prev_frames]
        cam_prev = cam_seg[:self.num_prev_frames]

        motion_clip = motion_seg[self.num_prev_frames:]
        gaze_clip = gaze_seg[self.num_prev_frames:]
        pose_clip = pose_seg[self.num_prev_frames:]
        cam_clip = cam_seg[self.num_prev_frames:]

        return (motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, 
                gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev)

    def __getitem__(self, index):
        try:
            (motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, 
             gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev) = self._get_full_clip(index)
        except Exception as e:
            print(f"[Error] Failed to get clip for index {index}: {e}. Trying a random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        ref_idx = torch.randint(low=0, high=motion_seg.shape[0], size=(1,)).item()
        m_ref = motion_seg[ref_idx]

        return {
            "m_now": motion_clip,
            "a_now": audio_clip,
            "gaze_now": gaze_clip,
            "pose_now": pose_clip,
            "cam_now": cam_clip,
            "m_prev": motion_prev,
            "a_prev": audio_prev,
            "gaze_prev": gaze_prev,
            "pose_prev": pose_prev,
            "cam_prev": cam_prev,
            "m_ref": m_ref,
        }
