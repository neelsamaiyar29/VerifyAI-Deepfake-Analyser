from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Tuple

import librosa
import numpy as np
import torch
from decord import AudioReader, VideoReader, cpu
from PIL import Image

NUM_FRAMES = 16
FRAME_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _ensure_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return file_path


def _uniform_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video contains no decodable frames.")

    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64).tolist()

    indices = list(range(total_frames))
    indices.extend([indices[-1]] * (num_frames - total_frames))
    return indices


def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(frame.astype(np.uint8)).resize(
        (FRAME_SIZE, FRAME_SIZE),
        Image.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def _load_audio_with_ffmpeg(file_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(file_path),
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            str(tmp_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg audio extraction failed.")

        samples, _ = librosa.load(
            tmp_path,
            sr=AUDIO_SAMPLE_RATE,
            mono=True,
        )
        return samples.astype(np.float32)
    finally:
        tmp_path.unlink(missing_ok=True)


def process_video(path: str | Path) -> torch.Tensor:
    file_path = _ensure_file(path)

    try:
        video_reader = VideoReader(str(file_path), ctx=cpu(0))
    except Exception as exc:
        raise RuntimeError(f"Failed to open video stream for {file_path}: {exc}") from exc

    total_frames = len(video_reader)
    frame_indices = _uniform_frame_indices(total_frames, NUM_FRAMES)

    try:
        frames = video_reader.get_batch(frame_indices).asnumpy()
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video frames from {file_path}: {exc}") from exc

    frame_tensors = [_frame_to_tensor(frame) for frame in frames]
    video_tensor = torch.stack(frame_tensors, dim=0)

    if video_tensor.shape != (NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE):
        raise ValueError(
            "Processed video has unexpected shape: "
            f"{tuple(video_tensor.shape)}; expected {(NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)}."
        )

    return video_tensor


def process_audio(path: str | Path) -> torch.Tensor:
    file_path = _ensure_file(path)

    audio_error: Exception | None = None
    try:
        audio_reader = AudioReader(
            str(file_path),
            ctx=cpu(0),
            sample_rate=AUDIO_SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        audio_error = exc
    else:
        try:
            samples = _to_numpy(audio_reader[:])
        except Exception as exc:
            audio_error = exc
        else:
            if samples.ndim == 1:
                mono_audio = samples.astype(np.float32)
            elif samples.ndim == 2 and samples.shape[0] >= 1:
                mono_audio = samples.mean(axis=0).astype(np.float32)
            else:
                raise ValueError(f"Decoded audio has unexpected shape: {samples.shape}")
            audio_error = None

    if audio_error is not None:
        try:
            mono_audio = _load_audio_with_ffmpeg(file_path)
        except Exception as ffmpeg_exc:
            raise RuntimeError(
                f"Failed to decode audio from {file_path} with decord and ffmpeg fallback: "
                f"decord error={audio_error}; ffmpeg error={ffmpeg_exc}"
            ) from ffmpeg_exc

    if mono_audio.size == 0:
        raise ValueError("Decoded audio is empty.")

    mel_spectrogram = librosa.feature.melspectrogram(
        y=mono_audio,
        sr=AUDIO_SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    audio_tensor = torch.from_numpy(log_mel.astype(np.float32))
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-6)

    if audio_tensor.ndim != 2 or audio_tensor.shape[0] != N_MELS:
        raise ValueError(
            f"Processed audio has unexpected shape: {tuple(audio_tensor.shape)}; expected (64, T)."
        )

    return audio_tensor


def align(video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if video.ndim != 4:
        raise ValueError(
            f"Video tensor must have shape (T, 3, 224, 224); got {tuple(video.shape)}."
        )
    if audio.ndim != 2:
        raise ValueError(f"Audio tensor must have shape (64, T); got {tuple(audio.shape)}.")

    t_video = video.shape[0]
    t_audio = audio.shape[1]
    min_len = min(t_video, t_audio)

    if min_len <= 0:
        raise ValueError(
            f"Cannot align empty modalities: video T={t_video}, audio T={t_audio}."
        )

    video = video[:min_len]
    audio = audio[:, :min_len]

    if video.shape != (min_len, 3, FRAME_SIZE, FRAME_SIZE):
        raise ValueError(
            f"Aligned video has unexpected shape: {tuple(video.shape)}; "
            f"expected ({min_len}, 3, {FRAME_SIZE}, {FRAME_SIZE})."
        )
    if audio.ndim != 2 or audio.shape[0] != N_MELS:
        raise ValueError(
            f"Aligned audio has unexpected shape: {tuple(audio.shape)}; expected (64, T)."
        )
    assert video.shape[0] == audio.shape[1]
    return video, audio
