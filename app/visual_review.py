import base64
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.config import Settings


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class VisualReviewError(RuntimeError):
    pass


def review_candidate_frame(
    video_path: Path,
    scene: dict[str, Any],
    settings: Settings,
) -> dict[str, Any] | None:
    if not settings.groq_api_key or not settings.groq_vision_model:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="incubrix-review-") as directory:
            frame_path = Path(directory) / "frame.jpg"
            subprocess.run(
                [
                    settings.ffmpeg_path, "-v", "error", "-ss", "1",
                    "-i", str(video_path), "-frames:v", "1", "-vf", "scale=768:-2",
                    "-q:v", "5", "-y", str(frame_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if not frame_path.is_file() or not frame_path.stat().st_size:
                raise VisualReviewError("Could not extract a frame for Groq visual review")
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    except VisualReviewError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise VisualReviewError("Could not extract a frame for Groq visual review") from exc

    prompt = (
        "Review this stock-video frame against the requested narration scene. "
        "Respond only as JSON with boolean fields relevant, brand_logo_visible, and safety_concern. "
        "Set relevant true only when the visible footage clearly supports the scene. "
        "Flag recognizable third-party brand logos and obviously unsafe or graphic imagery. "
        "Do not infer or estimate probabilities. Scene: "
        + (scene.get("visual_hint") or scene.get("text") or "")[:1200]
    )
    body = {
        "model": settings.groq_vision_model,
        "temperature": 0,
        "max_completion_tokens": 160,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            completion = json.loads(response.read())
        content = completion["choices"][0]["message"]["content"]
        verdict = json.loads(content) if isinstance(content, str) else None
        if not isinstance(verdict, dict) or any(
            not isinstance(verdict.get(field), bool)
            for field in ("relevant", "brand_logo_visible", "safety_concern")
        ):
            raise ValueError("Vision response did not contain the required boolean fields")
        return {
            "status": "reviewed",
            "model": settings.groq_vision_model,
            "relevant": verdict["relevant"],
            "brand_logo_visible": verdict["brand_logo_visible"],
            "safety_concern": verdict["safety_concern"],
            "calibrated": False,
        }
    except urllib.error.HTTPError as exc:
        raise VisualReviewError(f"Groq visual review returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VisualReviewError("Groq visual review was unavailable or returned an invalid response") from exc
