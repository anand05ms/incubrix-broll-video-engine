import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.db import JobDatabase
from app.providers import (
    MAX_PROVIDERS_PER_SCENE,
    MAX_QUERIES_PER_SCENE,
    MAX_RESULTS_PER_QUERY,
    MAX_TOTAL_SEARCH_SECONDS_PER_SCENE,
    PROVIDER_TIMEOUT_SECONDS,
    StockProvidersExhausted,
    search_stock_clip,
)
from app.relevance import (
    MIN_CONTEXTUAL_FALLBACK_SIMILARITY,
    MIN_VISUAL_SIMILARITY,
    MODEL_VERSION,
    VisualRelevanceError,
    choose_crop_anchor,
)
from app.scene_understanding import enrich_scene_visual_hints
from app.storage import ObjectStorage


ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}
ALLOWED_RATIOS = {"16:9", "9:16", "1:1"}
CATEGORY_TERMS = {
    "finance": {"money", "saving", "save", "budget", "sales", "cost", "price", "invest", "cash", "dollar", "rupee"},
    "technology": {"computing", "software", "data", "digital", "computer", "code", "internet", "ai", "technology"},
    "nature": {"nature", "tree", "forest", "plant", "climate", "earth", "ocean", "water", "garden", "environment"},
    "travel": {"travel", "journey", "city", "road", "country", "destination", "trip", "map"},
    "learning": {
        "learn", "learning", "education", "school", "study", "studies", "studying", "teach",
        "knowledge", "lesson", "course", "student", "students", "classroom", "online",
    },
    "food": {"food", "cook", "meal", "restaurant", "recipe", "eat", "coffee"},
    "wellbeing": {"health", "wellbeing", "wellness", "care", "calm", "stress", "happy", "sad"},
}
VISUAL_SCENE_TERMS = {
    "person", "people", "worker", "workers", "office", "work", "working", "write", "writing",
    "read", "reading", "plan", "planning", "organize", "organizing", "task", "tasks",
    "calendar", "schedule", "checklist", "laptop", "computer", "smartphone", "technology",
    "business", "businesses", "student", "students", "study", "studying", "learn", "learning",
    "meeting", "team", "customer", "customers", "data", "analyze", "analysis", "exercise",
    "walking", "meditation", "food", "garden", "plant", "travel", "drive", "build", "building",
    "create", "creating", "cook", "cooking", "care", "help", "helping",
}
ABSTRACT_SCENE_TERMS = {
    "abstract", "consistency", "consistent", "improvement", "improvements", "meaningful",
    "results", "success", "failure", "potential", "opportunity", "opportunities", "future",
    "impact", "value", "values", "growth", "progress", "responsibility", "responsible",
    "guarantee", "guarantees", "purpose", "purposeful", "confidence", "trust", "innovation",
}
logger = logging.getLogger("broll")
SENSITIVE_SCENE_TERMS = {
    "accident", "attack", "bereavement", "crash", "death", "disaster", "died", "funeral",
    "grief", "murder", "suicide", "tragedy", "war",
}
PALETTES = [
    ((15, 31, 54), (34, 109, 115), (103, 208, 167)),
    ((36, 27, 61), (92, 66, 139), (243, 177, 122)),
    ((22, 46, 48), (44, 116, 101), (212, 218, 143)),
    ((52, 33, 40), (133, 73, 76), (248, 177, 142)),
    ((19, 42, 67), (53, 98, 154), (119, 190, 222)),
]
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r".+?(?:[.!?](?:[\"')\]]*)?(?=\s|$)|$)", re.DOTALL)


class EngineError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


def _run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        return result
    except FileNotFoundError as exc:
        raise EngineError("MEDIA_TOOL_UNAVAILABLE", f"Required executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EngineError("MEDIA_TOOL_TIMEOUT", f"Media command timed out: {command[0]}", retryable=True) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-1000:]
        raise EngineError("MEDIA_PROCESSING_FAILED", detail or f"Media command failed: {command[0]}") from exc


def _command_version(binary: str | None, arguments: list[str]) -> str | None:
    if not binary:
        return None
    try:
        result = subprocess.run([binary, *arguments], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0].strip() if output else None


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_remote_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise EngineError("INVALID_AUDIO_URL", "audio_url must be an HTTP(S) URL without embedded credentials")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise EngineError("AUDIO_SOURCE_UNAVAILABLE", "The audio_url host could not be resolved", retryable=True) from exc
    for address in addresses:
        ip = address[4][0]
        if not ipaddress.ip_address(ip).is_global:
            raise EngineError("INVALID_AUDIO_URL", "Private and local audio_url hosts are not allowed")


def _download_audio(url: str, destination: Path, max_bytes: int) -> None:
    _safe_remote_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "IncuBrixBroll/0.1"})
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=20) as response, destination.open("wb") as output:
            if response.status < 200 or response.status >= 300:
                raise EngineError("AUDIO_SOURCE_UNAVAILABLE", f"Audio source returned HTTP {response.status}", retryable=True)
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise EngineError("AUDIO_TOO_LARGE", f"Audio exceeds the {max_bytes}-byte upload limit")
                output.write(chunk)
    except urllib.error.HTTPError as exc:
        raise EngineError("AUDIO_SOURCE_UNAVAILABLE", f"Audio source returned HTTP {exc.code}", retryable=exc.code >= 500) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EngineError("AUDIO_SOURCE_UNAVAILABLE", "Could not download audio_url", retryable=True) from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, file_pointer: Any, code: int, message: str, headers: Any, new_url: str) -> None:
        return None


def _probe_duration(path: Path, settings: Settings) -> float:
    try:
        result = _run(
            [
                settings.ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            timeout=30,
        )
    except EngineError as exc:
        if exc.code == "MEDIA_PROCESSING_FAILED":
            raise EngineError("INVALID_AUDIO", "The input is not a supported audio file") from exc
        raise
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise EngineError("INVALID_AUDIO", "The input is not a supported audio file") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise EngineError("INVALID_AUDIO", "Audio duration must be greater than zero")
    if duration > settings.max_audio_seconds:
        raise EngineError("AUDIO_TOO_LONG", f"Audio exceeds the {settings.max_audio_seconds}-second limit")
    return duration


def _synthesize(text: str, output: Path, settings: Settings, voice: str) -> None:
    if settings.tts_provider == "mock":
        _run(
            [settings.ffmpeg_path, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1", str(output)],
            timeout=30,
        )
        return
    if settings.tts_provider == "kokoro":
        try:
            import numpy as np
            import soundfile as sf
            from kokoro import KPipeline

            normalized_voice = voice.strip().lower()
            voice_name = (
                voice
                if "_" in voice
                else "hf_alpha"
                if normalized_voice.startswith(("hi", "hindi"))
                else "bf_emma"
                if normalized_voice.startswith(("en-gb", "british"))
                else "af_heart"
            )
            language_code = {"a": "a", "b": "b", "h": "h"}.get(voice_name[0].lower(), "a")
            chunks = [audio for _, _, audio in KPipeline(lang_code=language_code)(text, voice=voice_name)]
            if not chunks:
                raise ValueError("Kokoro returned no speech audio")
            audio = np.concatenate(chunks)
            if audio.size == 0:
                raise ValueError("Kokoro returned empty speech audio")
            sf.write(str(output), audio, 24000, subtype="PCM_16")
        except ImportError as exc:
            raise EngineError(
                "TTS_UNAVAILABLE",
                "Kokoro TTS requires the optional 'kokoro' dependency extra",
            ) from exc
        except Exception as exc:
            raise EngineError("TTS_FAILED", f"Kokoro narration failed: {exc}") from exc
        return
    if settings.tts_provider == "espeak":
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            raise EngineError("TTS_UNAVAILABLE", "Install espeak-ng or configure another TTS provider")
        language = re.sub(r"[^A-Za-z_-]", "", voice.split("-")[0]) or settings.tts_voice
        _run([binary, "-v", language, "-w", str(output), text], timeout=120)
        return
    if settings.tts_provider == "pyttsx3":
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.save_to_file(text, str(output))
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            raise EngineError("TTS_FAILED", f"Local pyttsx3 narration failed: {exc}") from exc
        return
    raise EngineError("TTS_PROVIDER_UNSUPPORTED", f"Unsupported TTS_PROVIDER: {settings.tts_provider}")


def _transcribe(audio_path: Path, settings: Settings) -> list[dict[str, Any]]:
    if settings.asr_provider != "faster-whisper":
        raise EngineError("ASR_PROVIDER_UNSUPPORTED", f"Unsupported ASR_PROVIDER: {settings.asr_provider}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise EngineError(
            "ASR_UNAVAILABLE",
            "Audio-only jobs require the optional faster-whisper package; install the 'asr' extra",
        ) from exc
    try:
        model = WhisperModel(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
        segments, info = model.transcribe(str(audio_path), vad_filter=True, task="transcribe")
        transcript = [
            {"text": segment.text.strip(), "start": float(segment.start), "end": float(segment.end)}
            for segment in segments
            if segment.text.strip() and segment.end > segment.start
        ]
        language = str(getattr(info, "language", "en") or "en").lower()
        if settings.asr_translate_for_search and language.split("-")[0] != "en":
            translated_segments, _ = model.transcribe(
                str(audio_path),
                vad_filter=True,
                task="translate",
            )
            translated = [
                {"text": segment.text.strip(), "start": float(segment.start), "end": float(segment.end)}
                for segment in translated_segments
                if segment.text.strip() and segment.end > segment.start
            ]
            for part in transcript:
                best_match = max(
                    translated,
                    key=lambda candidate: max(
                        0.0,
                        min(part["end"], candidate["end"]) - max(part["start"], candidate["start"]),
                    ),
                    default=None,
                )
                if best_match:
                    part["search_text"] = best_match["text"]
    except Exception as exc:
        raise EngineError("TRANSCRIPTION_FAILED", f"Local speech transcription failed: {exc}") from exc
    if not transcript:
        raise EngineError("TRANSCRIPTION_EMPTY", "No speech was detected in the audio")
    return transcript


def _split_script(text: str) -> list[str]:
    sentences = [match.group().strip() for match in SENTENCE_RE.finditer(text.strip()) if match.group().strip()]
    return sentences or [text.strip()]


def _scene_list(data: dict[str, Any], audio_duration: float, audio_path: Path, settings: Settings) -> list[dict[str, Any]]:
    if data.get("scenes"):
        scene_inputs = data["scenes"]
        scenes = [{"text": scene["text"].strip(), "visual_hint": (scene.get("visual_hint") or "").strip()} for scene in scene_inputs]
    elif data.get("script"):
        scenes = [{"text": text, "visual_hint": ""} for text in _split_script(data["script"])]
    else:
        transcript = _transcribe(audio_path, settings)
        scenes = [
            {
                "text": part["text"],
                "visual_hint": part.get("search_text", ""),
                "start": part["start"],
                "end": part["end"],
                "_search_translation_used": bool(part.get("search_text")),
            }
            for part in transcript
        ]

    scenes = [scene for scene in scenes if scene.get("text")]
    if not scenes:
        raise EngineError("EMPTY_SCENES", "At least one non-empty scene is required")

    has_transcript_timing = all("start" in scene and "end" in scene for scene in scenes)
    if has_transcript_timing:
        for scene in scenes:
            scene["start"] = max(0.0, min(float(scene["start"]), audio_duration))
            scene["end"] = max(scene["start"], min(float(scene["end"]), audio_duration))
        timed_scenes = [scene for scene in scenes if float(scene["end"]) > float(scene["start"])]
        for index, scene in enumerate(timed_scenes):
            next_start = float(timed_scenes[index + 1]["start"]) if index + 1 < len(timed_scenes) else audio_duration
            scene["start"] = 0.0 if index == 0 else float(timed_scenes[index - 1]["end"])
            scene["end"] = max(
                scene["start"],
                min(audio_duration, (float(scene["end"]) + next_start) / 2) if index + 1 < len(timed_scenes) else audio_duration,
            )
        scenes = timed_scenes
    else:
        weights = [max(1, len(WORD_RE.findall(scene["text"]))) for scene in scenes]
        total_weight = sum(weights)
        cursor = 0.0
        for index, (scene, weight) in enumerate(zip(scenes, weights)):
            end = audio_duration if index == len(scenes) - 1 else cursor + audio_duration * weight / total_weight
            scene["start"] = cursor
            scene["end"] = max(cursor, end)
            cursor = end

    merged: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for index, scene in enumerate(scenes):
        if pending is None:
            pending = dict(scene)
        else:
            pending["text"] = f"{pending['text']} {scene['text']}".strip()
            hints = [hint for hint in (pending.get("visual_hint"), scene.get("visual_hint")) if hint]
            pending["visual_hint"] = "; ".join(hints)
            pending["end"] = scene["end"]
        remaining_seconds = audio_duration - float(scene["end"])
        if float(pending["end"]) - float(pending["start"]) >= settings.min_scene_seconds and remaining_seconds >= settings.min_scene_seconds:
            merged.append(pending)
            pending = None
    if pending:
        if merged and audio_duration - float(merged[-1]["start"]) < settings.min_scene_seconds:
            merged[-1]["text"] = f"{merged[-1]['text']} {pending['text']}".strip()
            merged[-1]["visual_hint"] = "; ".join(
                hint for hint in (merged[-1].get("visual_hint"), pending.get("visual_hint")) if hint
            )
            merged[-1]["end"] = pending["end"]
        else:
            merged.append(pending)
    return merged


def _category(text: str, hint: str) -> str:
    words = {word.lower() for word in WORD_RE.findall(f"{text} {hint}")}
    finance_context = {"money", "account", "cash", "loan", "saving", "savings", "budget", "interest", "deposit", "river", "shore"}
    tech_context = {"compute", "computing", "server", "software", "program", "programming", "code", "device", "keyboard"}
    nature_context = {"river", "shore", "water", "sky", "weather", "snake", "animal", "field", "wild"}
    if words & {"cloud", "clouds"}:
        if words & tech_context:
            return "technology"
        if words & nature_context:
            return "nature"
        return "abstract"
    if "bank" in words:
        if words & {"river", "shore", "water"}:
            return "nature"
        if words & finance_context:
            return "finance"
        return "abstract"
    if "mouse" in words:
        if words & tech_context:
            return "technology"
        return "nature" if words & nature_context else "abstract"
    if "python" in words:
        if words & tech_context:
            return "technology"
        return "nature" if words & {"snake", "animal"} else "abstract"
    if "value" in words and words & finance_context:
        return "finance"
    counts = {name: len(words & terms) for name, terms in CATEGORY_TERMS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "abstract"


def _fallback_visual_category(scene: dict[str, Any]) -> str:
    terms = {
        word.lower()
        for word in WORD_RE.findall(f"{scene.get('text', '')} {scene.get('visual_hint', '')}")
    }
    if terms & {"communication", "communicate", "chat", "message", "messages", "messaging", "phone", "smartphone", "call", "calling"}:
        return "communication"
    if terms & {"improvement", "improvements", "organize", "organizing", "organization"}:
        return "improvement"
    if terms & {"problem", "problems", "challenge", "challenges", "clutter", "overwhelm"}:
        return "problem"
    if terms & {"growth", "grow", "growing", "progress", "success", "results", "result", "consistency", "consistent"}:
        return "growth"
    if terms & {"plan", "planning", "task", "tasks", "productivity", "calendar", "schedule"}:
        return "productivity"
    if terms & {"business", "businesses", "entrepreneur", "company", "companies", "customer", "customers"}:
        return "business"
    return _category(scene.get("text", ""), scene.get("visual_hint", ""))


def _is_visually_representable(scene: dict[str, Any]) -> bool:
    terms = {
        word.lower()
        for word in WORD_RE.findall(f"{scene.get('text', '')} {scene.get('visual_hint', '')}")
    }
    if terms & VISUAL_SCENE_TERMS:
        return True
    if terms & ABSTRACT_SCENE_TERMS:
        return False
    return True


def _font(size: int, sample_text: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    has_tamil = any("\u0b80" <= char <= "\u0bff" for char in sample_text)
    has_devanagari = any("\u0900" <= char <= "\u097f" for char in sample_text)
    script_font = (
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"
        if has_tamil
        else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
        if has_devanagari
        else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    )
    for path in (
        os.getenv("FONT_PATH"),
        script_font,
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_icon(draw: ImageDraw.ImageDraw, category: str, x: int, y: int, size: int, accent: tuple[int, int, int]) -> None:
    width = max(8, size // 40)
    left, top, right, bottom = x - size // 2, y - size // 2, x + size // 2, y + size // 2
    if category == "finance":
        draw.ellipse((left, top, left + size // 2, top + size // 2), outline=accent, width=width)
        draw.text((left + size // 7, top + size // 9), "$", font=_font(size // 3), fill=accent)
        draw.line((left, bottom, left + size // 3, bottom - size // 3, left + size // 2, bottom - size // 4, right, top), fill=accent, width=width)
    elif category == "technology":
        draw.ellipse((x - size // 5, y - size // 5, x + size // 5, y + size // 5), outline=accent, width=width)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            px, py = x + dx * size // 3, y + dy * size // 3
            draw.line((x, y, px, py), fill=accent, width=width)
            draw.ellipse((px - size // 18, py - size // 18, px + size // 18, py + size // 18), fill=accent)
    elif category == "travel":
        draw.line((left, y, x - size // 8, top, x + size // 5, bottom, right, top + size // 7), fill=accent, width=width)
        draw.ellipse((x - size // 10, y - size // 10, x + size // 10, y + size // 10), outline=accent, width=width)
    elif category == "learning":
        draw.line((x, top, x, bottom), fill=accent, width=width)
        draw.arc((left, top, x + size // 8, bottom), 190, 350, fill=accent, width=width)
        draw.arc((x - size // 8, top, right, bottom), 190, 350, fill=accent, width=width)
    elif category == "food":
        draw.arc((left, top, right, bottom), 0, 180, fill=accent, width=width)
        draw.line((left, y, right, y), fill=accent, width=width)
        draw.line((x, top, x + size // 7, top - size // 5), fill=accent, width=width)
    elif category == "nature":
        draw.ellipse((x - size // 3, top, x + size // 3, y + size // 4), outline=accent, width=width)
        draw.line((x, y, x, bottom), fill=accent, width=width)
        draw.arc((left, y, x, bottom), 180, 330, fill=accent, width=width)
        draw.arc((x, y, right, bottom), 210, 360, fill=accent, width=width)
    elif category == "wellbeing":
        draw.arc((left, top, x + size // 10, y + size // 5), 180, 360, fill=accent, width=width)
        draw.arc((x - size // 10, top, right, y + size // 5), 180, 360, fill=accent, width=width)
        draw.line((left, top + size // 10, x, bottom, right, top + size // 10), fill=accent, width=width)
    else:
        draw.ellipse((left, top, right, bottom), outline=accent, width=width)
        draw.ellipse((x - size // 4, y - size // 4, x + size // 4, y + size // 4), outline=accent, width=width)
        draw.ellipse((x - size // 12, y - size // 12, x + size // 12, y + size // 12), fill=accent)


def _draw_fallback_scene(
    draw: ImageDraw.ImageDraw,
    category: str,
    width: int,
    height: int,
    accent: tuple[int, int, int],
) -> None:
    scale = min(width, height)
    ink = (37, 58, 77)
    soft_ink = (86, 111, 128)
    pale = (239, 246, 246)
    surface = (222, 232, 231)
    left, right = int(width * 0.08), int(width * 0.92)
    horizon = int(height * 0.69)

    draw.rectangle((0, horizon, width, height), fill=surface)
    draw.rounded_rectangle(
        (left, int(height * 0.1), right, horizon),
        radius=max(18, scale // 45),
        fill=(225, 238, 240),
        outline=(255, 255, 255),
        width=max(3, scale // 180),
    )
    window_left, window_top = int(width * 0.13), int(height * 0.16)
    window_right, window_bottom = int(width * 0.87), int(height * 0.58)
    draw.rounded_rectangle(
        (window_left, window_top, window_right, window_bottom),
        radius=max(12, scale // 70),
        fill=(190, 220, 225),
        outline=(255, 255, 255),
        width=max(4, scale // 100),
    )
    draw.line(
        ((window_left + window_right) // 2, window_top, (window_left + window_right) // 2, window_bottom),
        fill=(255, 255, 255),
        width=max(4, scale // 100),
    )
    draw.line(
        (window_left, (window_top + window_bottom) // 2, window_right, (window_top + window_bottom) // 2),
        fill=(255, 255, 255),
        width=max(4, scale // 100),
    )
    draw.ellipse(
        (int(width * 0.67), int(height * 0.19), int(width * 0.78), int(height * 0.39)),
        fill=accent,
    )
    draw.ellipse(
        (int(width * 0.18), int(height * 0.27), int(width * 0.25), int(height * 0.4)),
        fill=(246, 250, 242),
    )

    if category == "communication":
        phone = (int(width * 0.39), int(height * 0.25), int(width * 0.61), int(height * 0.7))
        draw.rounded_rectangle(phone, radius=scale // 28, fill=ink, outline=(255, 255, 255), width=max(4, scale // 130))
        screen = (int(width * 0.405), int(height * 0.285), int(width * 0.595), int(height * 0.66))
        draw.rounded_rectangle(screen, radius=scale // 45, fill=pale)
        draw.rounded_rectangle(
            (int(width * 0.15), int(height * 0.37), int(width * 0.4), int(height * 0.49)),
            radius=scale // 35,
            fill=(255, 255, 255),
        )
        draw.rounded_rectangle(
            (int(width * 0.6), int(height * 0.52), int(width * 0.86), int(height * 0.64)),
            radius=scale // 35,
            fill=(193, 231, 215),
        )
        for x, y, line_width in (
            (0.18, 0.405, 0.17),
            (0.18, 0.45, 0.12),
            (0.64, 0.555, 0.17),
            (0.64, 0.6, 0.12),
        ):
            draw.line(
                (int(width * x), int(height * y), int(width * (x + line_width)), int(height * y)),
                fill=soft_ink,
                width=max(3, scale // 130),
            )
        draw.ellipse(
            (int(width * 0.475), int(height * 0.675), int(width * 0.525), int(height * 0.69)),
            fill=(192, 210, 211),
        )
        return

    if category == "business":
        shop = (int(width * 0.22), int(height * 0.38), int(width * 0.78), int(height * 0.78))
        draw.rectangle(shop, fill=(250, 249, 238), outline=soft_ink, width=max(4, scale // 140))
        roof = [
            (int(width * 0.18), int(height * 0.39)),
            (int(width * 0.82), int(height * 0.39)),
            (int(width * 0.76), int(height * 0.28)),
            (int(width * 0.24), int(height * 0.28)),
        ]
        draw.polygon(roof, fill=accent, outline=soft_ink)
        for stripe in range(5):
            x = int(width * (0.25 + stripe * 0.1))
            draw.polygon(
                ((x, int(height * 0.29)), (x + int(width * 0.055), int(height * 0.29)),
                 (x + int(width * 0.075), int(height * 0.39)), (x + int(width * 0.02), int(height * 0.39))),
                fill=(255, 255, 255),
            )
        draw.rectangle(
            (int(width * 0.29), int(height * 0.45), int(width * 0.48), int(height * 0.63)),
            fill=(188, 222, 226),
            outline=soft_ink,
            width=max(3, scale // 180),
        )
        draw.rectangle(
            (int(width * 0.56), int(height * 0.47), int(width * 0.7), int(height * 0.78)),
            fill=(166, 123, 97),
            outline=soft_ink,
            width=max(3, scale // 180),
        )
        draw.ellipse(
            (int(width * 0.66), int(height * 0.6), int(width * 0.68), int(height * 0.62)),
            fill=(246, 229, 173),
        )
        return

    if category == "growth":
        axis_x, axis_y = int(width * 0.22), int(height * 0.68)
        draw.line((axis_x, int(height * 0.3), axis_x, axis_y, int(width * 0.8), axis_y), fill=soft_ink, width=max(5, scale // 120))
        for x, top in ((0.3, 0.55), (0.42, 0.48), (0.54, 0.4), (0.66, 0.31)):
            draw.rounded_rectangle(
                (int(width * x), int(height * top), int(width * (x + 0.07)), axis_y),
                radius=max(5, scale // 120),
                fill=accent,
            )
        points = (
            (int(width * 0.3), int(height * 0.5)),
            (int(width * 0.44), int(height * 0.43)),
            (int(width * 0.57), int(height * 0.34)),
            (int(width * 0.73), int(height * 0.22)),
        )
        draw.line(points, fill=ink, width=max(6, scale // 90), joint="curve")
        for x, y in points:
            radius = scale // 55
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=pale, outline=ink, width=max(4, scale // 140))
        return

    if category == "improvement":
        board = (int(width * 0.16), int(height * 0.24), int(width * 0.84), int(height * 0.69))
        draw.rounded_rectangle(board, radius=scale // 45, fill=(250, 250, 243), outline=soft_ink, width=max(4, scale // 140))
        for column, rows in enumerate((2, 3, 2)):
            x = int(width * (0.22 + column * 0.2))
            draw.rounded_rectangle(
                (x, int(height * 0.3), x + int(width * 0.15), int(height * 0.36)),
                radius=scale // 100,
                fill=accent if column == 1 else (206, 222, 220),
            )
            for row in range(rows):
                y = int(height * (0.41 + row * 0.075))
                draw.rounded_rectangle(
                    (x, y, x + int(width * 0.15), y + int(height * 0.055)),
                    radius=scale // 110,
                    fill=(225, 238, 234),
                )
                check_x = x + int(width * 0.025)
                draw.line(
                    ((check_x, y + int(height * 0.028)), (check_x + int(width * 0.01), y + int(height * 0.04)),
                     (check_x + int(width * 0.028), y + int(height * 0.015))),
                    fill=(63, 137, 112),
                    width=max(3, scale // 150),
                )
        return

    if category == "problem":
        for index, bounds in enumerate(
            ((0.17, 0.34, 0.48, 0.58), (0.42, 0.28, 0.76, 0.53), (0.29, 0.51, 0.65, 0.71))
        ):
            x1, y1, x2, y2 = bounds
            draw.rounded_rectangle(
                (int(width * x1), int(height * y1), int(width * x2), int(height * y2)),
                radius=scale // 45,
                fill=(250, 249, 243),
                outline=(180, 111, 107),
                width=max(4, scale // 140),
            )
            for row in range(2):
                y = int(height * (y1 + 0.08 + row * 0.07))
                draw.line(
                    (int(width * (x1 + 0.05)), y, int(width * (x2 - 0.05 - row * 0.03)), y),
                    fill=(137, 124, 124),
                    width=max(3, scale // 160),
                )
        draw.line(
            ((int(width * 0.25), int(height * 0.28)), (int(width * 0.43), int(height * 0.46)),
             (int(width * 0.33), int(height * 0.61)), (int(width * 0.7), int(height * 0.26))),
            fill=(186, 103, 97),
            width=max(6, scale // 100),
        )
        return

    if category == "nature":
        draw.rectangle((int(width * 0.65), int(height * 0.44), int(width * 0.68), horizon), fill=(93, 101, 78))
        for bounds in (
            (0.55, 0.32, 0.78, 0.57),
            (0.61, 0.27, 0.84, 0.52),
            (0.49, 0.4, 0.7, 0.59),
        ):
            x1, y1, x2, y2 = bounds
            draw.ellipse(
                (int(width * x1), int(height * y1), int(width * x2), int(height * y2)),
                fill=(92, 160, 123),
            )
        draw.line((0, horizon, width, horizon), fill=(135, 176, 144), width=max(4, scale // 120))
        return

    desk_top = int(height * 0.72)
    draw.rounded_rectangle(
        (int(width * 0.1), desk_top, int(width * 0.9), int(height * 0.77)),
        radius=max(8, scale // 100),
        fill=(176, 139, 112),
    )
    draw.rectangle(
        (int(width * 0.14), int(height * 0.77), int(width * 0.17), int(height * 0.96)),
        fill=(130, 100, 86),
    )
    draw.rectangle(
        (int(width * 0.83), int(height * 0.77), int(width * 0.86), int(height * 0.96)),
        fill=(130, 100, 86),
    )

    if category == "learning":
        book_left, book_top = int(width * 0.18), int(height * 0.55)
        book_right, book_bottom = int(width * 0.42), int(height * 0.7)
        draw.polygon(
            ((book_left, book_bottom), (book_left, book_top), (width // 2, book_top + scale // 45),
             (width // 2, book_bottom)),
            fill=(250, 249, 238),
            outline=soft_ink,
        )
        draw.polygon(
            ((width // 2, book_bottom), (width // 2, book_top + scale // 45), (book_right, book_top),
             (book_right, book_bottom)),
            fill=(255, 255, 249),
            outline=soft_ink,
        )
        for line_index in range(3):
            y = book_top + scale // 28 + line_index * scale // 24
            draw.line((book_left + scale // 40, y, width // 2 - scale // 35, y), fill=(179, 193, 193), width=max(2, scale // 240))
            draw.line((width // 2 + scale // 35, y, book_right - scale // 40, y), fill=(179, 193, 193), width=max(2, scale // 240))
    elif category == "productivity":
        card_left, card_top = int(width * 0.18), int(height * 0.43)
        card_right, card_bottom = int(width * 0.4), int(height * 0.67)
        draw.rounded_rectangle(
            (card_left, card_top, card_right, card_bottom),
            radius=max(10, scale // 75),
            fill=(252, 251, 243),
            outline=soft_ink,
            width=max(3, scale // 180),
        )
        draw.rounded_rectangle(
            (card_left, card_top, card_right, card_top + scale // 20),
            radius=max(8, scale // 85),
            fill=accent,
        )
        for row in range(3):
            y = card_top + scale // 13 + row * scale // 19
            box_left = card_left + scale // 35
            draw.rounded_rectangle(
                (box_left, y, box_left + scale // 45, y + scale // 45),
                radius=max(2, scale // 300),
                outline=soft_ink,
                width=max(2, scale // 240),
            )
            draw.line(
                (box_left + scale // 20, y + scale // 50, card_right - scale // 35, y + scale // 50),
                fill=soft_ink,
                width=max(2, scale // 240),
            )
    elif category == "food":
        plate_box = (int(width * 0.18), int(height * 0.53), int(width * 0.42), int(height * 0.69))
        draw.ellipse(plate_box, fill=(250, 250, 239), outline=soft_ink, width=max(4, scale // 150))
        draw.ellipse(
            (plate_box[0] + scale // 30, plate_box[1] + scale // 30,
             plate_box[2] - scale // 30, plate_box[3] - scale // 30),
            outline=(192, 205, 198),
            width=max(3, scale // 190),
        )
        for food_x, food_y, food_color in (
            (0.25, 0.59, (220, 143, 97)),
            (0.31, 0.62, (109, 160, 117)),
            (0.35, 0.57, (228, 191, 101)),
        ):
            radius = scale // 26
            cx, cy = int(width * food_x), int(height * food_y)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=food_color)

    screen_left, screen_top = int(width * 0.48), int(height * 0.37)
    screen_right, screen_bottom = int(width * 0.79), int(height * 0.68)
    draw.rounded_rectangle(
        (screen_left, screen_top, screen_right, screen_bottom),
        radius=max(12, scale // 70),
        fill=ink,
    )
    inset = max(10, scale // 55)
    panel_left, panel_top = screen_left + inset, screen_top + inset
    panel_right, panel_bottom = screen_right - inset, screen_bottom - inset
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=max(6, scale // 100),
        fill=pale,
    )
    if category in {"technology", "business", "finance", "productivity", "abstract"}:
        chart_floor = panel_bottom - scale // 35
        bar_gap = scale // 45
        bar_width = max(10, (panel_right - panel_left - 5 * bar_gap) // 4)
        heights = (0.23, 0.38, 0.31, 0.52)
        for bar_index, fraction in enumerate(heights):
            bar_x = panel_left + bar_gap + bar_index * (bar_width + bar_gap)
            bar_height = int((panel_bottom - panel_top) * fraction)
            draw.rounded_rectangle(
                (bar_x, chart_floor - bar_height, bar_x + bar_width, chart_floor),
                radius=max(4, scale // 140),
                fill=accent if bar_index == 3 else (113, 170, 167),
            )
        draw.line(
            (
                panel_left + bar_gap,
                panel_top + scale // 28,
                panel_left + bar_gap * 3,
                panel_top + scale // 19,
                panel_left + bar_gap * 5,
                panel_top + scale // 40,
            ),
            fill=soft_ink,
            width=max(3, scale // 180),
        )
        for node_x, node_y in (
            (panel_left + scale // 35, panel_top + scale // 13),
            (panel_left + scale // 12, panel_top + scale // 17),
            (panel_left + scale // 7, panel_top + scale // 11),
        ):
            radius = max(5, scale // 95)
            draw.ellipse((node_x - radius, node_y - radius, node_x + radius, node_y + radius), fill=accent)
    else:
        card_left, card_top = panel_left + scale // 35, panel_top + scale // 35
        draw.rounded_rectangle(
            (card_left, card_top, panel_right - scale // 35, panel_bottom - scale // 35),
            radius=max(5, scale // 110),
            fill=(255, 255, 255),
            outline=(204, 218, 218),
            width=max(2, scale // 200),
        )
        for row in range(3):
            y = card_top + scale // 24 + row * scale // 22
            draw.ellipse((card_left + scale // 65, y - scale // 110, card_left + scale // 45, y + scale // 110), fill=accent)
            draw.line((card_left + scale // 32, y, panel_right - scale // 24, y), fill=soft_ink, width=max(2, scale // 220))

    base_y = screen_bottom + scale // 100
    draw.polygon(
        ((screen_left - scale // 45, base_y), (screen_right + scale // 45, base_y),
         (screen_right + scale // 14, desk_top), (screen_left - scale // 14, desk_top)),
        fill=(120, 140, 148),
    )
    draw.line(
        (screen_left - scale // 14, desk_top, screen_right + scale // 14, desk_top),
        fill=(232, 240, 237),
        width=max(4, scale // 100),
    )
    draw.ellipse(
        (int(width * 0.84), int(height * 0.6), int(width * 0.9), int(height * 0.7)),
        fill=(110, 154, 116),
    )
    draw.rectangle(
        (int(width * 0.86), int(height * 0.66), int(width * 0.88), desk_top),
        fill=(110, 121, 92),
    )


def _create_artwork(
    scene: dict[str, Any],
    index: int,
    ratio: str,
    output: Path,
    style: str | None = None,
    include_text: bool = True,
) -> None:
    width, height = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
    digest = hashlib.sha256(f"{scene['text']}:{index}:{style or ''}".encode()).digest()
    dark, middle, accent = PALETTES[digest[0] % len(PALETTES)]
    if not include_text:
        dark = tuple(round(value * 0.35 + 130 * 0.65) for value in dark)
        middle = tuple(round(value * 0.45 + 170 * 0.55) for value in middle)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        blend = y / max(1, height - 1)
        color = tuple(round(dark[i] * (1 - blend) + middle[i] * blend) for i in range(3))
        for x in range(width):
            pixels[x, y] = color
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    glow_alpha = 42 if not include_text else 16
    overlay_draw.ellipse((width * 0.54, -height * 0.36, width * 1.04, height * 0.55), fill=(*accent, glow_alpha))
    overlay_draw.ellipse((-width * 0.3, height * 0.58, width * 0.42, height * 1.24), fill=(*accent, glow_alpha - 3))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    category = (
        _fallback_visual_category(scene)
        if not include_text
        else _category(scene["text"], scene.get("visual_hint", ""))
    )
    if not include_text:
        _draw_fallback_scene(draw, category, width, height, accent)
        image.save(output, format="PNG", optimize=True)
        return

    icon_size = min(width, height) * (
        0.52 if not include_text else 0.31 if ratio == "9:16" else 0.28
    )
    icon_x = int(width * 0.5) if not include_text or ratio == "9:16" else int(width * 0.76)
    icon_y = int(height * (0.5 if not include_text else 0.3 if ratio == "9:16" else 0.46))
    if not include_text:
        ring_color = (*accent, 105)
        ring_radius = int(icon_size * 0.8)
        draw.ellipse(
            (icon_x - ring_radius, icon_y - ring_radius, icon_x + ring_radius, icon_y + ring_radius),
            outline=ring_color,
            width=max(4, int(icon_size * 0.018)),
        )
        draw.arc(
            (icon_x - ring_radius * 1.25, icon_y - ring_radius * 1.25,
             icon_x + ring_radius * 1.25, icon_y + ring_radius * 1.25),
            205,
            330,
            fill=(*accent, 170),
            width=max(6, int(icon_size * 0.025)),
        )
    _draw_icon(draw, category, icon_x, icon_y, int(icon_size), accent)

    if not include_text:
        image.save(output, format="PNG", optimize=True)
        return

    margin = int(width * 0.075)
    badge_font = _font(max(20, int(min(width, height) * 0.025)))
    title_font = _font(
        max(38, int(min(width, height) * (0.055 if ratio != "9:16" else 0.05))),
        scene["text"],
    )
    label = category.upper() if category != "abstract" else "IDEA"
    draw.rounded_rectangle((margin, margin, margin + int(width * 0.27), margin + int(height * 0.065)), radius=18, fill=(*accent, 48), outline=(*accent, 190), width=2)
    draw.text((margin + 18, margin + 8), label, font=badge_font, fill=(255, 255, 255, 240))

    text = scene["text"].strip()
    max_text_width = int(width * (0.83 if ratio == "9:16" else 0.58))
    lines = _wrap(draw, text, title_font, max_text_width)
    line_height = int(title_font.size * 1.25)
    max_lines = 5 if ratio == "9:16" else 4
    lines = lines[:max_lines]
    text_y = int(height * (0.56 if ratio == "9:16" else 0.53))
    for line in lines:
        draw.text((margin, text_y), line, font=title_font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 35))
        text_y += line_height
    if scene.get("visual_hint"):
        hint_font = _font(max(16, int(min(width, height) * 0.022)), scene["visual_hint"])
        hint = " ".join(scene["visual_hint"].split())
        hint_lines = _wrap(draw, hint, hint_font, max_text_width)
        for line in hint_lines[:2]:
            draw.text((margin, min(text_y + 16, height - margin - hint_font.size)), line, font=hint_font, fill=(220, 232, 238, 240))
            text_y += int(hint_font.size * 1.4)

    image.save(output, format="PNG", optimize=True)
def _srt_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _captions(scenes: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{i}\n{_srt_time(scene['start'])} --> {_srt_time(scene['end'])}\n{scene['text']}"
        for i, scene in enumerate(scenes, 1)
    ) + "\n"


def _render_clip(image_path: Path, output: Path, ratio: str, duration: float, settings: Settings) -> None:
    width, height = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
    overscan_width, overscan_height = round(width * 1.04), round(height * 1.04)
    video_filter = (
        f"scale={overscan_width}:{overscan_height}:force_original_aspect_ratio=increase,"
        f"crop={overscan_width}:{overscan_height},"
        f"zoompan=z='min(zoom+0.0005,1.035)':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps=30,"
        "format=yuv420p"
    )
    _run(
        [
            settings.ffmpeg_path, "-y", "-loop", "1", "-framerate", "30", "-i", str(image_path),
            "-t", f"{duration:.3f}",
            "-vf", video_filter,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(output),
        ],
        timeout=240,
    )


def _render_stock_clip(
    video_path: Path,
    output: Path,
    ratio: str,
    duration: float,
    settings: Settings,
    crop_anchor: tuple[float, float] = (0.5, 0.5),
) -> None:
    width, height = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
    crop_x, crop_y = (min(1.0, max(0.0, value)) for value in crop_anchor)
    _run(
        [
            settings.ffmpeg_path, "-y", "-stream_loop", "-1", "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}:x='(iw-ow)*{crop_x:.3f}':y='(ih-oh)*{crop_y:.3f}',"
                "fps=30,eq=saturation=0.95:contrast=1.02,format=yuv420p"
            ),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(output),
        ],
        timeout=300,
    )


def _validate_visual_asset(path: Path, kind: str, ratio: str | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise EngineError(
            "VISUAL_ASSET_INVALID",
            f"Scene visual asset is missing or empty: {path.name}",
        )
    if kind == "stock":
        return
    try:
        with Image.open(path) as image:
            if ratio:
                expected_size = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
                if image.size != expected_size:
                    raise EngineError(
                        "VISUAL_ASSET_INVALID",
                        f"Scene illustration has size {image.size}, expected {expected_size}",
                    )
            if image.mode not in {"RGB", "RGBA"}:
                raise EngineError("VISUAL_ASSET_INVALID", f"Scene illustration mode is not RGB/RGBA: {image.mode}")
            if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                raise EngineError("VISUAL_ASSET_INVALID", "Scene illustration contains transparent pixels")
            frame = image.convert("RGB").resize((32, 32))
            pixels = list(frame.getdata())
    except (OSError, ValueError) as exc:
        raise EngineError(
            "VISUAL_ASSET_INVALID",
            f"Scene illustration cannot be decoded: {path.name}",
        ) from exc
    brightness = sum(sum(pixel) / 3 for pixel in pixels) / len(pixels)
    visible_pixels = sum(max(pixel) > 12 for pixel in pixels)
    if (
        brightness <= 2
        or visible_pixels < len(pixels) * 0.01
        or len(set(pixels)) < 5
    ):
        raise EngineError(
            "VISUAL_ASSET_INVALID",
            f"Scene illustration is effectively black, blank, or uniform: {path.name}",
        )


def _validate_rendered_segment(
    path: Path,
    settings: Settings,
    expected_duration: float,
    require_visible: bool,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise EngineError("SCENE_RENDER_INVALID", f"Rendered scene segment is missing or empty: {path.name}")
    try:
        result = subprocess.run(
            [
                settings.ffprobe_path,
                "-v", "error",
                "-show_entries", "stream=codec_name,codec_type,width,height,avg_frame_rate",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        probe = json.loads(result.stdout)
        video = next(
            (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        duration = float(probe.get("format", {}).get("duration", 0))
        frame_rate = video.get("avg_frame_rate", "0/0") if video else "0/0"
        numerator, denominator = (float(part) for part in frame_rate.split("/", maxsplit=1))
        fps = numerator / denominator if denominator else 0
        if (
            not video
            or video.get("codec_name") != "h264"
            or int(video.get("width", 0)) <= 0
            or int(video.get("height", 0)) <= 0
            or fps < 29
            or duration <= 0
            or abs(duration - expected_duration) > max(0.15, expected_duration * 0.02)
        ):
            raise EngineError(
                "SCENE_RENDER_INVALID",
                f"Scene segment must contain H.264 video at 30 fps for {expected_duration:.3f}s",
            )
        if require_visible:
            frame = subprocess.run(
                [
                    settings.ffmpeg_path,
                    "-v", "error",
                    "-ss", f"{min(expected_duration / 2, max(0, duration - 0.05)):.3f}",
                    "-i", str(path),
                    "-frames:v", "1",
                    "-vf", "scale=32:24,format=gray",
                    "-f", "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=20,
            ).stdout
            if len(frame) != 32 * 24 or sum(frame) / len(frame) <= 2:
                raise EngineError("SCENE_RENDER_INVALID", f"Scene segment is visually blank: {path.name}")
    except EngineError:
        raise
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        raise EngineError("SCENE_RENDER_INVALID", f"Could not validate scene segment: {path.name}") from exc


def _create_fallback_image(
    scene: dict[str, Any],
    index: int,
    ratio: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _create_artwork(scene, index, ratio, output, include_text=False)
    _validate_visual_asset(output, "procedural", ratio)


def _validate_rendered_video(output: Path, settings: Settings, expected_duration: float) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        raise EngineError("MEDIA_OUTPUT_INVALID", "Rendered video is missing or empty")
    try:
        result = subprocess.run(
            [
                settings.ffprobe_path,
                "-v", "error",
                "-show_entries", "stream=codec_type,width,height,avg_frame_rate",
                "-show_entries", "format=duration",
                "-of", "json",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        duration = float(probe.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0)) if video_stream else 0
        height = int(video_stream.get("height", 0)) if video_stream else 0
        frame_rate = video_stream.get("avg_frame_rate", "0/0") if video_stream else "0/0"
        rate_parts = frame_rate.split("/", maxsplit=1)
        fps = (
            float(rate_parts[0]) / float(rate_parts[1])
            if len(rate_parts) == 2 and float(rate_parts[1]) != 0
            else float(rate_parts[0])
        ) if rate_parts else 0
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise EngineError("MEDIA_OUTPUT_INVALID", "Could not validate the rendered video's streams") from exc
    if (
        not video_stream
        or not audio_stream
        or width <= 0
        or height <= 0
        or fps <= 0
        or duration <= 0
        or abs(duration - expected_duration) > max(1.0, expected_duration * 0.02)
    ):
        raise EngineError(
            "MEDIA_OUTPUT_INVALID",
            "Rendered video must contain valid audio/video streams and cover the narration duration",
        )

    brightness_samples: list[float] = []
    for fraction in (0.1, 0.5, 0.9):
        timestamp = min(max(0.0, duration * fraction), max(0.0, duration - 0.05))
        try:
            frame = subprocess.run(
                [
                    settings.ffmpeg_path,
                    "-v", "error",
                    "-ss", f"{timestamp:.3f}",
                    "-i", str(output),
                    "-frames:v", "1",
                    "-vf", "crop=iw:ih*0.75:0:0,scale=32:24,format=gray",
                    "-f", "rawvideo",
                    "-",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise EngineError("MEDIA_OUTPUT_INVALID", "Could not inspect rendered video frames") from exc
        if len(frame) != 32 * 24:
            raise EngineError("MEDIA_OUTPUT_INVALID", "Rendered video produced an invalid sampled frame")
        brightness_samples.append(sum(frame) / len(frame))
    if max(brightness_samples) <= 2:
        raise EngineError("MEDIA_OUTPUT_INVALID", "Rendered video frames are entirely black")


def _render_video(
    visuals: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    audio_path: Path,
    ratio: str,
    output: Path,
    captions_path: Path,
    burn_in: bool,
    duration: float,
    settings: Settings,
    work_dir: Path,
) -> None:
    clip_paths: list[Path] = []
    for i, (visual, scene) in enumerate(zip(visuals, scenes)):
        clip_path = work_dir / f"clip_{ratio.replace(':', '_')}_{i:03}.mp4"
        scene_duration = max(0.05, float(scene["end"]) - float(scene["start"]))
        visual_path = Path(visual["path"])
        visual_kind = (
            "procedural_fallback"
            if visual["kind"] != "stock" and visual.get("fallback_used")
            else visual["kind"]
        )
        if visual_kind == "procedural_fallback":
            visual["kind"] = visual_kind
        try:
            _validate_visual_asset(visual_path, visual_kind, ratio)
            if visual_kind == "stock":
                anchors = visual.get("crop_anchors") or {}
                _render_stock_clip(
                    visual_path,
                    clip_path,
                    ratio,
                    scene_duration,
                    settings,
                    anchors.get(ratio, (0.5, 0.5)),
                )
            else:
                _render_clip(visual_path, clip_path, ratio, scene_duration, settings)
            _validate_rendered_segment(clip_path, settings, scene_duration, require_visible=True)
        except (EngineError, OSError) as exc:
            fallback_path = (
                work_dir
                / "artwork"
                / ratio.replace(":", "_")
                / f"scene_{i:03}_render_fallback.png"
            )
            _create_fallback_image(scene, i, ratio, fallback_path)
            _render_clip(fallback_path, clip_path, ratio, scene_duration, settings)
            _validate_rendered_segment(clip_path, settings, scene_duration, require_visible=True)
            logger.warning(
                "Scene visual replaced with procedural fallback after render validation failed",
                extra={
                    "scene_id": f"scene-{i:03}",
                    "failed_visual_path": str(visual_path),
                    "fallback_path": str(fallback_path),
                    "fallback_reason": str(exc),
                },
            )
            visual_path = fallback_path
            visual_kind = "procedural_fallback"
            visual["kind"] = visual_kind
            visual["path"] = visual_path
            visual["fallback_used"] = True
        image_info = None
        if visual_kind != "stock":
            with Image.open(visual_path) as image:
                image_info = (image.width, image.height, image.mode)
            assert image_info is not None
        logger.info(
            "Scene visual path",
            extra={
                "scene_id": f"scene-{i:03}",
                "visual_type": visual_kind,
                "visual_path": str(visual_path),
                "segment_path": str(clip_path),
                "duration_seconds": scene_duration,
                "provider": visual.get("provider"),
                "candidate": visual.get("candidate"),
                **(
                    {
                        "image_width": image_info[0],
                        "image_height": image_info[1],
                        "image_mode": image_info[2],
                        "has_visible_pixels": True,
                    }
                    if visual_kind != "stock"
                    else {}
                ),
            },
        )
        clip_paths.append(clip_path)

    concat_list = work_dir / f"concat_{ratio.replace(':', '_')}.txt"
    concat_list.write_text(
        "".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in clip_paths),
        encoding="utf-8",
    )
    if burn_in:
        subtitle_path = captions_path.as_posix().replace(":", r"\:").replace("'", r"\'")
        subtitle_style = (
            "FontSize=14,BorderStyle=3,BackColour=&H80000000,"
            "Outline=2,OutlineColour=&H80000000,Shadow=0,MarginL=48,MarginR=48,MarginV=58"
        )
        video_filter = f"subtitles='{subtitle_path}':force_style='{subtitle_style}',format=yuv420p"
        command = [
            settings.ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(audio_path),
            "-t", f"{duration:.3f}", "-vf", video_filter, "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
    else:
        command = [
            settings.ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(audio_path),
            "-t", f"{duration:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
        ]
    _run(command, timeout=600)
    _validate_rendered_video(output, settings, duration)


def _voiceover_fallback_visual(
    scene_index: int,
    attempts: list[dict[str, str]],
    search_duration_ms: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "kind": "procedural",
        "metadata": {
            "source": "procedural-illustration",
            "source_url": None,
            "license": "Original scene-specific illustration generated by this service; no third-party media used",
            "attribution": "None required",
            "ai_generated": False,
            "relevance_method": "Narration category illustration; no stock footage was available",
            "selection_reason": "Scene-specific text-free illustration used only because no verified stock footage was available",
            "commercial_use": "Original generated illustration",
            "watermark_status": "No watermark added by this service",
            "visual_status": "procedural_illustration",
            "fallback": True,
            "fallback_reason": reason,
            "provider_attempts": attempts,
            "search_duration_ms": search_duration_ms,
            "fallback_decision": (
                f"Scene {scene_index} uses a text-free scene illustration; narration and captions continue. "
                f"Reason: {reason}"
            ),
        },
    }


def process_job(
    job_id: str,
    request_data: dict[str, Any],
    upload_path: str | None,
    db: JobDatabase,
    settings: Settings,
    storage: ObjectStorage,
    progress: Callable[[int], None],
) -> dict[str, Any]:
    started = time.monotonic()
    work_dir = settings.data_dir / "work" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path: Path
    warnings: list[str] = []
    try:
        if upload_path:
            audio_path = Path(upload_path)
        elif request_data.get("audio_url"):
            suffix = Path(urllib.parse.urlparse(request_data["audio_url"]).path).suffix.lower()
            audio_path = work_dir / f"input{suffix if suffix in ALLOWED_AUDIO_SUFFIXES else '.audio'}"
            _download_audio(request_data["audio_url"], audio_path, settings.max_upload_bytes)
        else:
            audio_path = work_dir / "narration.wav"
            script = request_data["script"] or " ".join(scene["text"] for scene in request_data.get("scenes", []))
            _synthesize(script, audio_path, settings, request_data.get("voice") or settings.tts_voice)

        duration = _probe_duration(audio_path, settings)
        scenes = _scene_list(request_data, duration, audio_path, settings)
        scenes, scene_planning_notice, scene_planning_provider = enrich_scene_visual_hints(scenes, settings)
        if scene_planning_notice:
            warnings.append(scene_planning_notice)
        if not scenes:
            raise EngineError("EMPTY_SCENES", "No speech scenes were available to render")

        captions_text = _captions(scenes)
        captions_path = work_dir / "captions.srt"
        captions_path.write_text(captions_text, encoding="utf-8")
        progress(20)

        ratios = request_data.get("aspect_ratios") or ["16:9"]
        used_video_ids: set[int | str] = set()
        visuals: list[dict[str, Any]] = []
        for i, scene in enumerate(scenes):
            scene_search_started = time.monotonic()
            scene_id = f"scene-{i:03}"
            scene_words = {word.lower() for word in WORD_RE.findall(scene["text"])}
            uses_stock = settings.visual_provider != "procedural"
            if not uses_stock:
                visual = {
                    "kind": "procedural",
                    "metadata": {
                        "source": "procedural-illustration",
                        "visual_status": "voiceover_primary",
                        "fallback": False,
                        "fallback_reason": "stock_search_disabled",
                        "provider_attempts": [],
                        "search_duration_ms": 0,
                    },
                }
            elif not _is_visually_representable(scene):
                visual = _voiceover_fallback_visual(
                    i,
                    [],
                    0,
                    "abstract_or_non_visual_scene",
                )
                warnings.append(
                    f"Scene {i} is abstract or non-visual; narration and captions remain primary."
                )
            elif scene_words & SENSITIVE_SCENE_TERMS:
                message = f"Stock search skipped for sensitive scene {i}; human-selected footage is required."
                visual = _voiceover_fallback_visual(
                    i,
                    [],
                    0,
                    "sensitive_scene_requires_human_footage",
                )
                warnings.append(f"{message} Narration and captions continue without a stock search.")
            elif request_data.get("avoid"):
                message = (
                    f"Stock search skipped for scene {i} because avoid constraints cannot be verified automatically."
                )
                visual = _voiceover_fallback_visual(
                    i,
                    [],
                    0,
                    "avoid_constraints_require_human_review",
                )
                warnings.append(f"{message} Narration and captions continue without a stock search.")
            else:
                try:
                    stock_path = work_dir / f"stock_{i:03}.mp4"
                    metadata = search_stock_clip(
                        {**scene, "_scene_id": scene_id},
                        used_video_ids,
                        stock_path,
                        settings,
                    )
                    metadata["visual_status"] = "stock_broll"
                    visual = {"kind": "stock", "path": stock_path, "metadata": metadata}
                except StockProvidersExhausted as exc:
                    search_duration_ms = round((time.monotonic() - scene_search_started) * 1000)
                    visual = _voiceover_fallback_visual(
                        i,
                        exc.attempts,
                        search_duration_ms,
                        "no_verified_stock_within_search_budget",
                    )
                    message = "No sufficiently relevant footage found; scene-specific illustration used."
                    message += f" Scene {i}: video continues with narration."
                    warnings.append(message)
                except Exception as exc:
                    search_duration_ms = round((time.monotonic() - scene_search_started) * 1000)
                    message = f"Stock search failed for scene {i}: {exc}"
                    visual = _voiceover_fallback_visual(
                        i,
                        [{"provider": "stock_search", "status": "unavailable", "reason": str(exc)}],
                        search_duration_ms,
                        "stock_search_error",
                    )
                    warnings.append(f"{message}. Continuing with narration and a supporting visual.")
            metadata = visual.get("metadata") or {}
            search_duration_ms = metadata.get(
                "search_duration_ms",
                round((time.monotonic() - scene_search_started) * 1000),
            )
            logger.info(
                "Scene visual selection completed",
                extra={
                    "scene_id": scene_id,
                    "visual_status": metadata.get("visual_status", "stock_broll"),
                    "providers_attempted": list(
                        dict.fromkeys(
                            attempt.get("provider", "unknown")
                            for attempt in metadata.get("provider_attempts", [])
                        )
                    ),
                    "search_duration_ms": search_duration_ms,
                    "selected_provider": metadata.get("source"),
                    "fallback_reason": metadata.get("fallback_reason"),
                },
            )
            if metadata.get("fallback_decision"):
                warnings.append(f"Scene {i}: {metadata['fallback_decision']}")
            visuals.append(visual)

        for index, visual in enumerate(visuals):
            if visual["kind"] != "stock":
                continue
            try:
                visual["crop_anchors"] = {
                    ratio: choose_crop_anchor(
                        visual["path"],
                        scenes[index],
                        ratio,
                        settings.ffmpeg_path,
                        settings.ffprobe_path,
                    )
                    for ratio in ratios
                }
            except VisualRelevanceError as exc:
                raise EngineError("VISUAL_CROP_FAILED", str(exc)) from exc
        image_paths: dict[str, list[Path]] = {}
        for ratio in ratios:
            ratio_dir = work_dir / "artwork" / ratio.replace(":", "_")
            ratio_dir.mkdir(parents=True, exist_ok=True)
            image_paths[ratio] = []
            for i, scene in enumerate(scenes):
                if visuals[i]["kind"] == "stock":
                    image_paths[ratio].append(visuals[i]["path"])
                else:
                    image_path = ratio_dir / f"scene_{i:03}.png"
                    is_last_resort = bool((visuals[i].get("metadata") or {}).get("fallback"))
                    _create_artwork(
                        scene,
                        i,
                        ratio,
                        image_path,
                        request_data.get("style"),
                        not is_last_resort,
                    )
                    image_paths[ratio].append(image_path)
        progress(40)

        captions_options = request_data.get("captions") or {"burn_in": False, "srt": True}
        artifacts: dict[str, str] = {}
        rendered_assets_by_ratio: dict[str, list[dict[str, Any]]] = {}
        for index, ratio in enumerate(ratios):
            filename = f"video_{ratio.replace(':', '_')}.mp4"
            output_path = work_dir / filename
            ratio_visuals = [
                    {
                        "kind": visuals[i]["kind"],
                        "path": image_path,
                        "provider": (visuals[i].get("metadata") or {}).get("source"),
                        "candidate": (visuals[i].get("metadata") or {}).get("video_id"),
                        "fallback_used": bool((visuals[i].get("metadata") or {}).get("fallback")),
                        "crop_anchors": visuals[i].get("crop_anchors"),
                    }
                    for i, image_path in enumerate(image_paths[ratio])
                ]
            _render_video(
                ratio_visuals,
                scenes,
                audio_path,
                ratio,
                output_path,
                captions_path,
                bool(captions_options.get("burn_in", False)),
                duration,
                settings,
                work_dir,
            )
            for scene_index, ratio_visual in enumerate(ratio_visuals):
                selected_visual = visuals[scene_index]
                if ratio_visual["kind"] != "procedural_fallback" or selected_visual["kind"] != "stock":
                    continue
                selected_metadata = selected_visual.get("metadata") or {}
                selected_visual["kind"] = "procedural"
                selected_visual["path"] = ratio_visual["path"]
                selected_visual["metadata"] = {
                    "source": "procedural-illustration",
                    "source_url": None,
                    "license": "Original scene-specific illustration generated by this service; no third-party media used",
                    "attribution": "None required",
                    "ai_generated": False,
                    "relevance_method": "Generated procedural visual after the selected stock clip failed render validation",
                    "selection_reason": "The scene-specific stock clip failed rendering validation; a generated visual was rendered instead",
                    "commercial_use": "Original generated illustration",
                    "watermark_status": "No watermark added by this service",
                    "visual_status": "procedural_illustration",
                    "fallback": True,
                    "fallback_reason": "stock_clip_failed_render_validation",
                    "provider_attempts": selected_metadata.get("provider_attempts", []),
                    "search_duration_ms": selected_metadata.get("search_duration_ms", 0),
                    "fallback_decision": "Selected stock media failed renderer validation; a scene-specific procedural visual was rendered.",
                    "render_fallback_from": {
                        key: selected_metadata.get(key)
                        for key in ("source", "video_id", "source_url", "license", "attribution")
                    },
                }
                warnings.append(
                    f"Scene {scene_index} selected stock failed render validation; a procedural visual was rendered."
                )
                for future_ratio in ratios[index + 1 :]:
                    future_path = (
                        work_dir
                        / "artwork"
                        / future_ratio.replace(":", "_")
                        / f"scene_{scene_index:03}_render_fallback.png"
                    )
                    _create_fallback_image(scenes[scene_index], scene_index, future_ratio, future_path)
                    image_paths[future_ratio][scene_index] = future_path
            rendered_assets_by_ratio[ratio] = [
                {
                    "kind": visual["kind"],
                    "source_path": visual["path"],
                    "segment_path": work_dir / f"clip_{ratio.replace(':', '_')}_{scene_index:03}.mp4",
                    "duration": max(
                        0.05,
                        float(scenes[scene_index]["end"]) - float(scenes[scene_index]["start"]),
                    ),
                    "fallback_used": bool(visual.get("fallback_used"))
                    or visual["kind"] == "procedural_fallback",
                }
                for scene_index, visual in enumerate(ratio_visuals)
            ]
            artifacts[filename] = f"/api/v1/jobs/{job_id}/artifacts/{filename}"
            progress(45 + int(35 * (index + 1) / len(ratios)))

        timeline_scenes = []
        for i, scene in enumerate(scenes):
            category = _fallback_visual_category(scene)
            provenance = visuals[i]["metadata"]
            if visuals[i]["kind"] != "stock":
                provenance = dict(visuals[i].get("metadata") or {})
                provenance.setdefault("source", "procedural-illustration")
                provenance.setdefault("source_url", None)
                provenance.setdefault(
                    "license",
                    "Original graphic generated by this service; no third-party media used",
                )
                provenance.setdefault("attribution", "None required")
                provenance.setdefault("ai_generated", False)
                provenance.setdefault(
                    "relevance_method",
                    "No frame-level semantic score; narration remains primary",
                )
                provenance.setdefault(
                    "selection_reason",
                    f"Created an original {category} illustration from the narration and visual hint",
                )
                provenance.setdefault("commercial_use", "Original procedural artwork")
                provenance.setdefault("watermark_status", "No watermark added by this service")
                provenance.setdefault("visual_category", category)
                provenance.setdefault("relevance_score", None)
            clip_id = f"{job_id}-scene-{i:03}"
            fallback_used = bool(provenance.get("fallback"))
            timeline_scenes.append(
                {
                    "scene_id": f"scene-{i:03}",
                    "text": scene["text"],
                    "narration": scene["text"],
                    "start": round(float(scene["start"]), 3),
                    "end": round(float(scene["end"]), 3),
                    "visual_status": (visuals[i].get("metadata") or {}).get(
                        "visual_status",
                        "stock_broll" if visuals[i]["kind"] == "stock" else "voiceover_primary",
                    ),
                    "visual_assets": {
                        ratio: {
                            "kind": rendered_assets_by_ratio[ratio][i]["kind"],
                            "path": rendered_assets_by_ratio[ratio][i]["segment_path"]
                            .relative_to(work_dir)
                            .as_posix(),
                            "source_path": rendered_assets_by_ratio[ratio][i]["source_path"]
                            .relative_to(work_dir)
                            .as_posix(),
                            "duration": round(rendered_assets_by_ratio[ratio][i]["duration"], 3),
                            "fallback_used": rendered_assets_by_ratio[ratio][i]["fallback_used"],
                        }
                        for ratio in ratios
                    },
                    "clip": {
                        "clip_id": clip_id,
                        "asset_id": clip_id,
                        "provider_asset_id": provenance.get("video_id"),
                        "provider": provenance.get("source"),
                        "visual_category": category if visuals[i]["kind"] != "stock" else None,
                        "relevance_score": provenance.get("relevance_score"),
                        "fallback_used": fallback_used,
                        "reused_clip": False,
                        **{key: value for key, value in provenance.items() if key not in {"path", "video_id"}},
                    },
                }
            )
        provider_attempts = [
            {
                "scene_id": f"scene-{index:03}",
                "selected_provider": (visual.get("metadata") or {}).get("source"),
                "attempts": (visual.get("metadata") or {}).get("provider_attempts", []),
                "fallback_decision": (visual.get("metadata") or {}).get("fallback_decision"),
            }
            for index, visual in enumerate(visuals)
            if (visual.get("metadata") or {}).get("provider_attempts")
        ]
        missing_provider_credentials = sorted(
            {
                attempt["reason"]
                for scene_attempt in provider_attempts
                for attempt in scene_attempt["attempts"]
                if attempt.get("status") == "skipped"
                and "required credentials are not configured" in attempt.get("reason", "")
            }
        )
        warnings.extend(
            f"Stock provider unavailable: {reason}. Configure the listed environment variable(s) to enable it."
            for reason in missing_provider_credentials
        )
        timeline = {
            "job_id": job_id,
            "duration_seconds": round(duration, 3),
            "provider_attempts": provider_attempts,
            "scenes": timeline_scenes,
        }
        timeline_path = work_dir / "timeline.json"
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        if captions_options.get("srt", True):
            artifacts["captions.srt"] = f"/api/v1/jobs/{job_id}/artifacts/captions.srt"
        artifacts["timeline.json"] = f"/api/v1/jobs/{job_id}/artifacts/timeline.json"

        elapsed = round(time.monotonic() - started, 3)
        if any(visual["kind"] == "stock" for visual in visuals):
            used_sources = sorted(
                {visual["metadata"]["source"] for visual in visuals if visual["kind"] == "stock"}
            )
            warnings.append(
                f"Narration-derived search keywords were sent to {', '.join(used_sources)}."
            )
            warnings.append(
                f"{MODEL_VERSION} checked three sampled frames per selected clip; it does not verify "
                "watermark status, AI-generation disclosure, or content safety."
            )
        if any(visual["kind"] == "procedural" for visual in visuals):
            warnings.append(
                "Scenes without verified stock footage use original scene-category illustrations; "
                "these procedural visuals are not assigned stock-frame relevance scores."
            )
        uses_tts = not request_data.get("audio_url") and upload_path is None
        uses_asr = not request_data.get("script") and not request_data.get("scenes")
        espeak_binary = shutil.which("espeak-ng") or shutil.which("espeak")
        tts_version = (
            _command_version(espeak_binary, ["--version"])
            if settings.tts_provider == "espeak"
            else _package_version("pyttsx3")
            if settings.tts_provider == "pyttsx3"
            else _package_version("kokoro")
            if settings.tts_provider == "kokoro"
            else None
        ) if uses_tts else None
        asr_version = _package_version("faster-whisper") if uses_asr else None
        ffmpeg_version = _command_version(settings.ffmpeg_path, ["-version"])
        ffprobe_version = _command_version(settings.ffprobe_path, ["-version"])
        visual_review_used = any(
            visual["kind"] == "stock" and visual["metadata"].get("visual_review")
            for visual in visuals
        )
        groq_used = scene_planning_provider == "groq" or visual_review_used
        has_stock_visuals = any(visual["kind"] == "stock" for visual in visuals)
        if uses_tts and not tts_version:
            warnings.append("The TTS provider version could not be determined.")
        if uses_asr and not asr_version:
            warnings.append("The ASR provider version could not be determined.")
        if not ffmpeg_version or not ffprobe_version:
            warnings.append("A media-tool version could not be determined.")
        for index, visual in enumerate(visuals):
            metadata = visual.get("metadata") or {}
            if metadata.get("fallback_reason") == "low_confidence_contextual_match":
                warnings.append(
                    f"LOW_CONFIDENCE_BROLL_FALLBACK: scene-{index:03} selected the highest-scoring "
                    "verified stock candidate below the contextual relevance threshold."
                )
        manifest = {
            "job_id": job_id,
            "models": {
                "visuals": {
                    "provider": (
                        " + ".join(
                            sorted({visual["metadata"]["source"] for visual in visuals if visual["kind"] == "stock"})
                            + ["OpenCLIP"]
                        )
                        if has_stock_visuals
                        else "procedural-illustration"
                    ),
                    "version": MODEL_VERSION if has_stock_visuals else "0.1.0",
                    "crop_method": (
                        "OpenCLIP semantic anchor chosen from five aspect-ratio crop positions"
                        if has_stock_visuals
                        else None
                    ),
                },
                "tts": {
                    "provider": settings.tts_provider if uses_tts else None,
                    "version": tts_version,
                    "voice": request_data.get("voice") or settings.tts_voice if uses_tts else None,
                },
                "asr": {
                    "provider": settings.asr_provider if uses_asr else None,
                    "version": asr_version,
                    "model": settings.asr_model if uses_asr else None,
                    "search_translation_used": any(
                        scene.get("_search_translation_used") for scene in scenes
                    ),
                },
                "scene_planning": {
                    "provider": scene_planning_provider,
                    "model": settings.groq_model if scene_planning_provider == "groq" else None,
                },
                "visual_review": {
                    "provider": "groq" if visual_review_used else None,
                    "model": settings.groq_vision_model if visual_review_used else None,
                    "calibrated": False if visual_review_used else None,
                },
            },
            "media_tools": {"ffmpeg": ffmpeg_version, "ffprobe": ffprobe_version},
            "visual_provider_attempts": provider_attempts,
            "visual_fallback_selections": [
                {
                    "scene_id": f"scene-{index:03}",
                    "visual_status": (visual.get("metadata") or {}).get("visual_status"),
                    "provider": (visual.get("metadata") or {}).get("source"),
                    "source": (visual.get("metadata") or {}).get("source_url"),
                    "license": (visual.get("metadata") or {}).get("license"),
                    "attribution": (visual.get("metadata") or {}).get("attribution"),
                    "relevance_score": (visual.get("metadata") or {}).get("relevance_score"),
                    "fallback_reason": (visual.get("metadata") or {}).get("fallback_reason"),
                }
                for index, visual in enumerate(visuals)
                if (visual.get("metadata") or {}).get("fallback")
            ],
            "settings": {
                "aspect_ratios": ratios,
                "style": request_data.get("style"),
                "avoid": request_data.get("avoid", []),
                "captions": captions_options,
                "fps": 30,
                "minimum_visual_similarity": MIN_VISUAL_SIMILARITY,
                "minimum_contextual_fallback_similarity": MIN_CONTEXTUAL_FALLBACK_SIMILARITY,
                "last_resort_broll_fallback": (
                    "Best-scoring media/license-verified candidate is selected only when all candidates "
                    "score below minimum_contextual_fallback_similarity; human relevance review required."
                ),
                "max_queries_per_scene": MAX_QUERIES_PER_SCENE,
                "max_results_per_query": MAX_RESULTS_PER_QUERY,
                "max_providers_per_scene": MAX_PROVIDERS_PER_SCENE,
                "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
                "max_total_search_seconds_per_scene": MAX_TOTAL_SEARCH_SECONDS_PER_SCENE,
                "require_stock_footage": settings.require_stock_footage,
            },
            "seed": None,
            "timings": {"duration_seconds": round(duration, 3), "processing_seconds": elapsed},
            "cost_estimate_usd": None if groq_used else 0.0,
            "cost_basis": (
                "Groq scene-planning or visual-review usage is not priced in this manifest; check account billing. Hosting and compute are excluded."
                if groq_used
                else "No paid API calls; hosting and compute costs are excluded."
            ),
            "warnings": warnings,
            "rights": {
                "third_party_media": any(visual["kind"] == "stock" for visual in visuals),
                "commercial_use": (
                    "See each selected clip's license in timeline.json; verify current terms before use."
                    if any(visual["kind"] == "stock" for visual in visuals)
                    else "Original generated visuals; review applicable service terms before deployment."
                ),
                "attribution": (
                    "Per-clip attribution is included in timeline.json."
                    if any(visual["kind"] == "stock" for visual in visuals)
                    else "None required for generated illustrations."
                ),
                "ai_generated_status": (
                    "Source generation status is not consistently disclosed; inspect every selected clip."
                    if has_stock_visuals
                    else "Generated procedural illustrations are not AI-generated."
                ),
                "watermark_status": "Not independently verified for selected Pexels clips." if has_stock_visuals else "No watermark is added by this service.",
                "content_safety": (
                    "Some close-scoring frames may receive an un-calibrated Groq review; human review is required."
                    if visual_review_used
                    else "Selected video frames are not independently safety-screened; human review required."
                    if has_stock_visuals
                    else "Generated visuals contain no realistic faces or invented brand logos."
                ),
            },
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["manifest.json"] = f"/api/v1/jobs/{job_id}/artifacts/manifest.json"
        progress(90)

        media_types = {
            ".mp4": "video/mp4",
            ".srt": "application/x-subrip",
            ".json": "application/json",
        }
        for name in artifacts:
            path = work_dir / name
            storage.put_file(f"{job_id}/{name}", path, media_types[path.suffix])
        progress(100)
        return {"files": artifacts, "duration_seconds": round(duration, 3), "warnings": manifest["warnings"]}
    except EngineError:
        raise
    except Exception as exc:
        raise EngineError("JOB_PROCESSING_FAILED", str(exc), retryable=False) from exc
