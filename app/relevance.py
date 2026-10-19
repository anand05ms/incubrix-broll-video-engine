import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


MODEL_NAME = "ViT-B-32"
MODEL_PRETRAINED = "openai"
MODEL_VERSION = "OpenCLIP ViT-B/32"
MIN_VISUAL_SIMILARITY = 0.20
_MODEL_LOCK = threading.Lock()
_SCORING_LOCK = threading.Lock()
_SCORER: tuple[Any, Any, Any, Any] | None = None


class VisualRelevanceError(RuntimeError):
    pass


def _load_scorer() -> tuple[Any, Any, Any, Any]:
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    with _MODEL_LOCK:
        if _SCORER is None:
            try:
                import open_clip
                import torch
            except ImportError as exc:
                raise VisualRelevanceError(
                    "Frame-level matching requires the 'vision' dependency extra (open_clip_torch)."
                ) from exc
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME,
                    pretrained=MODEL_PRETRAINED,
                    device="cpu",
                    force_quick_gelu=True,
                )
                model.eval()
                tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            except Exception as exc:
                raise VisualRelevanceError(
                    f"Could not load {MODEL_VERSION}; check model-cache permissions and network access."
                ) from exc
            _SCORER = (torch, model, preprocess, tokenizer)
    return _SCORER


def _frame_similarity(frame_paths: list[Path], text: str) -> float:
    torch, model, preprocess, tokenizer = _load_scorer()
    try:
        from PIL import Image

        images = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                images.append(preprocess(image.convert("RGB")))
        image_batch = torch.stack(images)
        text_batch = tokenizer([text])
        with _SCORING_LOCK, torch.inference_mode():
            image_features = model.encode_image(image_batch)
            text_features = model.encode_text(text_batch)
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            similarities = image_features @ text_features.T
        return float(similarities.mean().item())
    except VisualRelevanceError:
        raise
    except Exception as exc:
        raise VisualRelevanceError(f"{MODEL_VERSION} could not score the extracted video frames.") from exc


def score_video_file(
    video_path: Path,
    scene: dict[str, Any],
    ffmpeg_path: str,
    ffprobe_path: str,
) -> float:
    try:
        probe = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(probe.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("Could not determine the candidate clip duration for frame matching") from exc
    if duration <= 0:
        raise ValueError("Candidate clip has no duration for frame matching")

    with tempfile.TemporaryDirectory(prefix="incubrix-frames-") as temporary_directory:
        frame_paths = []
        for index, fraction in enumerate((0.2, 0.5, 0.8)):
            timestamp = min(max(0.0, duration - 0.05), duration * fraction)
            frame_path = Path(temporary_directory) / f"frame-{index}.jpg"
            try:
                subprocess.run(
                    [
                        ffmpeg_path,
                        "-v", "error",
                        "-ss", f"{timestamp:.3f}",
                        "-i", str(video_path),
                        "-frames:v", "1",
                        "-vf", "scale=384:384:force_original_aspect_ratio=decrease",
                        "-y", str(frame_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("Could not extract a candidate frame for visual matching") from exc
            if frame_path.is_file() and frame_path.stat().st_size:
                frame_paths.append(frame_path)

        if not frame_paths:
            raise ValueError("Candidate video produced no frames for visual matching")

        hint = (scene.get("visual_hint") or "").strip()
        narration = (scene.get("text") or "").strip()
        text = f"Video footage showing {hint}. Narration: {narration}" if hint else f"Video footage showing: {narration}"
        return _frame_similarity(frame_paths, text)
