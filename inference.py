from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model_custom import CrossModalModel
from utils import align, process_audio, process_video

LABELS = {0: "REAL", 1: "FAKE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-modal deepfake detection inference."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    return parser.parse_args()


def load_checkpoint(model: CrossModalModel, checkpoint_path: Path, device: torch.device) -> None:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {exc}") from exc

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch while loading model_final.pth with strict=True."
        ) from exc


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    checkpoint_path = Path(__file__).resolve().parent / "model_final.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if not video_path.is_file():
        print(f"Error: video file does not exist: {video_path}", file=sys.stderr)
        return 1
    if not checkpoint_path.is_file():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    try:
        print("Loading cross-modal model...")
        model = CrossModalModel().to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()

        print("Preprocessing video frames...")
        frames = process_video(video_path)

        print("Preprocessing audio track...")
        mel = process_audio(video_path)

        print("Aligning modalities...")
        frames, mel = align(frames, mel)
        print(f"Aligned input shapes: video={tuple(frames.shape)}, audio={tuple(mel.shape)}")

        frames = frames.unsqueeze(0).float().to(device)
        mel = mel.unsqueeze(0).float().to(device)
        assert frames.shape[1] == mel.shape[2]

        print(
            "Running inference with aligned shapes "
            f"video={tuple(frames.shape)}, audio={tuple(mel.shape)}..."
        )
        with torch.no_grad():
            output, _ = model(frames, mel)
            probs = torch.softmax(output, dim=1).cpu()

        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred].item() * 100.0)
        print(f"Prediction: {LABELS[pred]} (Confidence: {confidence:.1f}%)")
        return 0
    except (FileNotFoundError, ValueError, AssertionError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
