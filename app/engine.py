import hashlib
import importlib.metadata
import ipaddress
import json
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
from app.providers import search_pexels_clip
from app.relevance import MIN_VISUAL_SIMILARITY, MODEL_VERSION, VisualRelevanceError
from app.storage import ObjectStorage


ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}
ALLOWED_RATIOS = {"16:9", "9:16", "1:1"}
CATEGORY_TERMS = {
    "finance": {"money", "saving", "save", "budget", "sales", "cost", "price", "invest", "cash", "dollar", "rupee"},
    "technology": {"computing", "software", "data", "digital", "computer", "code", "internet", "ai", "technology"},
    "nature": {"nature", "tree", "forest", "plant", "climate", "earth", "ocean", "water", "garden", "environment"},
    "travel": {"travel", "journey", "city", "road", "country", "destination", "trip", "map"},
    "learning": {"learn", "education", "school", "study", "teach", "knowledge", "lesson", "course"},
    "food": {"food", "cook", "meal", "restaurant", "recipe", "eat", "coffee"},
    "wellbeing": {"health", "wellbeing", "wellness", "care", "calm", "stress", "happy", "sad"},
}
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
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


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
        except VisualRelevanceError as exc:
            raise EngineError("VISUAL_SCORING_UNAVAILABLE", str(exc)) from exc
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
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
        model = WhisperModel(settings.asr_model, device=settings.asr_device, compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), vad_filter=True)
        transcript = [
            {"text": segment.text.strip(), "start": float(segment.start), "end": float(segment.end)}
            for segment in segments
            if segment.text.strip() and segment.end > segment.start
        ]
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
        scenes = [{"text": part["text"], "visual_hint": "", "start": part["start"], "end": part["end"]} for part in transcript]

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


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        os.getenv("FONT_PATH"),
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


def _create_artwork(scene: dict[str, Any], index: int, ratio: str, output: Path, style: str | None = None) -> None:
    width, height = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
    digest = hashlib.sha256(f"{scene['text']}:{index}:{style or ''}".encode()).digest()
    dark, middle, accent = PALETTES[digest[0] % len(PALETTES)]
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        blend = y / max(1, height - 1)
        color = tuple(round(dark[i] * (1 - blend) + middle[i] * blend) for i in range(3))
        for x in range(width):
            pixels[x, y] = color
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse((width * 0.54, -height * 0.36, width * 1.04, height * 0.55), fill=(*accent, 16))
    draw.ellipse((-width * 0.3, height * 0.58, width * 0.42, height * 1.24), fill=(*accent, 13))

    category = _category(scene["text"], scene.get("visual_hint", ""))
    icon_size = min(width, height) * (0.31 if ratio == "9:16" else 0.28)
    icon_x = int(width * 0.5) if ratio == "9:16" else int(width * 0.76)
    icon_y = int(height * (0.3 if ratio == "9:16" else 0.46))
    _draw_icon(draw, category, icon_x, icon_y, int(icon_size), accent)

    margin = int(width * 0.075)
    badge_font = _font(max(20, int(min(width, height) * 0.025)))
    title_font = _font(max(38, int(min(width, height) * (0.055 if ratio != "9:16" else 0.05))))
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
        hint_font = _font(max(16, int(min(width, height) * 0.022)))
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


def _render_stock_clip(video_path: Path, output: Path, ratio: str, duration: float, settings: Settings) -> None:
    width, height = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}[ratio]
    _run(
        [
            settings.ffmpeg_path, "-y", "-stream_loop", "-1", "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30,eq=saturation=0.95:contrast=1.02,format=yuv420p",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(output),
        ],
        timeout=300,
    )


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
        if visual["kind"] == "stock":
            _render_stock_clip(visual["path"], clip_path, ratio, scene_duration, settings)
        else:
            _render_clip(visual["path"], clip_path, ratio, scene_duration, settings)
        clip_paths.append(clip_path)

    concat_list = work_dir / f"concat_{ratio.replace(':', '_')}.txt"
    concat_list.write_text(
        "".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in clip_paths),
        encoding="utf-8",
    )
    if burn_in:
        subtitle_path = captions_path.as_posix().replace(":", r"\:").replace("'", r"\'")
        video_filter = f"subtitles='{subtitle_path}',format=yuv420p"
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
        if not scenes:
            raise EngineError("EMPTY_SCENES", "No speech scenes were available to render")

        captions_text = _captions(scenes)
        captions_path = work_dir / "captions.srt"
        captions_path.write_text(captions_text, encoding="utf-8")
        progress(20)

        ratios = request_data.get("aspect_ratios") or ["16:9"]
        used_video_ids: set[int] = set()
        visuals: list[dict[str, Any]] = []
        for i, scene in enumerate(scenes):
            visual = None
            scene_words = {word.lower() for word in WORD_RE.findall(scene["text"])}
            if settings.visual_provider == "pexels" and scene_words & SENSITIVE_SCENE_TERMS:
                warnings.append(f"Stock search skipped for sensitive scene {i}; used a procedural illustration.")
            elif settings.visual_provider == "pexels" and request_data.get("avoid"):
                warnings.append(f"Stock search skipped for scene {i} because avoid constraints cannot be verified automatically; used a procedural illustration.")
            elif settings.visual_provider == "pexels":
                try:
                    stock_path = work_dir / f"stock_{i:03}.mp4"
                    metadata = search_pexels_clip(scene, used_video_ids, stock_path, settings)
                    if metadata:
                        visual = {"kind": "stock", "path": stock_path, "metadata": metadata}
                    else:
                        warnings.append(f"No Pexels result for scene {i}; used an original procedural illustration.")
                except Exception as exc:
                    warnings.append(f"Pexels unavailable for scene {i}; used an original procedural illustration ({exc}).")
            elif settings.visual_provider != "procedural":
                raise EngineError("VISUAL_PROVIDER_UNSUPPORTED", f"Unsupported VISUAL_PROVIDER: {settings.visual_provider}")
            if visual is None:
                visual = {"kind": "procedural", "metadata": None}
            visuals.append(visual)

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
                    _create_artwork(scene, i, ratio, image_path, request_data.get("style"))
                    image_paths[ratio].append(image_path)
        progress(40)

        captions_options = request_data.get("captions") or {"burn_in": False, "srt": True}
        artifacts: dict[str, str] = {}
        for index, ratio in enumerate(ratios):
            filename = f"video_{ratio.replace(':', '_')}.mp4"
            output_path = work_dir / filename
            _render_video(
                [
                    {"kind": visuals[i]["kind"], "path": image_path}
                    for i, image_path in enumerate(image_paths[ratio])
                ],
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
            artifacts[filename] = f"/api/v1/jobs/{job_id}/artifacts/{filename}"
            progress(45 + int(35 * (index + 1) / len(ratios)))

        timeline_scenes = []
        for i, scene in enumerate(scenes):
            category = _category(scene["text"], scene.get("visual_hint", ""))
            provenance = visuals[i]["metadata"] or {
                "source": "procedural-illustration",
                "source_url": None,
                "license": "Original graphic generated by this service; no third-party media used",
                "attribution": "None required",
                "ai_generated": False,
                "relevance_method": "No frame-level semantic score; original category illustration with narration text",
                "selection_reason": f"Created an original {category} illustration from the narration and visual hint",
                "commercial_use": "Original procedural artwork",
                "watermark_status": "No watermark added by this service",
                "content_review": "No realistic faces or brand logos are generated; visual relevance is not independently verified",
            }
            clip_id = f"{job_id}-scene-{i:03}"
            timeline_scenes.append(
                {
                    "scene_id": f"scene-{i:03}",
                    "text": scene["text"],
                    "start": round(float(scene["start"]), 3),
                    "end": round(float(scene["end"]), 3),
                    "clip": {
                        "clip_id": clip_id,
                        "asset_id": clip_id,
                        "provider_asset_id": provenance.get("video_id"),
                        **{key: value for key, value in provenance.items() if key not in {"path", "video_id"}},
                    },
                }
            )
        timeline = {"job_id": job_id, "duration_seconds": round(duration, 3), "scenes": timeline_scenes}
        timeline_path = work_dir / "timeline.json"
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        if captions_options.get("srt", True):
            artifacts["captions.srt"] = f"/api/v1/jobs/{job_id}/artifacts/captions.srt"
        artifacts["timeline.json"] = f"/api/v1/jobs/{job_id}/artifacts/timeline.json"

        elapsed = round(time.monotonic() - started, 3)
        if any(visual["kind"] == "stock" for visual in visuals):
            warnings.append("Narration-derived search keywords were sent to Pexels.")
            warnings.append(
                f"{MODEL_VERSION} checked three sampled frames per selected clip; it does not verify "
                "watermark status, AI-generation disclosure, or content safety."
            )
        if any(visual["kind"] == "procedural" for visual in visuals):
            warnings.append(
                "Procedural illustration scenes do not receive a frame-level relevance score and are not a substitute for B-roll."
            )
        uses_tts = not request_data.get("audio_url") and upload_path is None
        uses_asr = not request_data.get("script") and not request_data.get("scenes")
        espeak_binary = shutil.which("espeak-ng") or shutil.which("espeak")
        tts_version = (
            _command_version(espeak_binary, ["--version"])
            if settings.tts_provider == "espeak"
            else _package_version("pyttsx3") if settings.tts_provider == "pyttsx3" else "test fixture"
        ) if uses_tts else None
        asr_version = _package_version("faster-whisper") if uses_asr else None
        ffmpeg_version = _command_version(settings.ffmpeg_path, ["-version"])
        ffprobe_version = _command_version(settings.ffprobe_path, ["-version"])
        if uses_tts and not tts_version:
            warnings.append("The TTS provider version could not be determined.")
        if uses_asr and not asr_version:
            warnings.append("The ASR provider version could not be determined.")
        if not ffmpeg_version or not ffprobe_version:
            warnings.append("A media-tool version could not be determined.")
        manifest = {
            "job_id": job_id,
            "models": {
                "visuals": {
                    "provider": "Pexels + OpenCLIP" if any(visual["kind"] == "stock" for visual in visuals) else "procedural-illustration",
                    "version": MODEL_VERSION if any(visual["kind"] == "stock" for visual in visuals) else "0.1.0",
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
                },
            },
            "media_tools": {"ffmpeg": ffmpeg_version, "ffprobe": ffprobe_version},
            "settings": {
                "aspect_ratios": ratios,
                "style": request_data.get("style"),
                "avoid": request_data.get("avoid", []),
                "captions": captions_options,
                "fps": 30,
                "minimum_visual_similarity": MIN_VISUAL_SIMILARITY,
            },
            "seed": None,
            "timings": {"duration_seconds": round(duration, 3), "processing_seconds": elapsed},
            "cost_estimate_usd": 0.0,
            "cost_basis": "No paid API calls; hosting and compute costs are excluded.",
            "warnings": warnings,
            "rights": {
                "third_party_media": any(visual["kind"] == "stock" for visual in visuals),
                "commercial_use": "Pexels License applies to selected Pexels clips; review current provider terms before use." if any(visual["kind"] == "stock" for visual in visuals) else "Original generated illustrations; review applicable service terms before deployment.",
                "attribution": "Pexels and photographer attribution is included per stock clip." if any(visual["kind"] == "stock" for visual in visuals) else "None required for generated illustrations.",
                "ai_generated_status": "Pexels does not disclose source generation status; inspect every selected clip." if any(visual["kind"] == "stock" for visual in visuals) else "Procedural illustrations are not AI-generated.",
                "watermark_status": "Not independently verified for selected Pexels clips." if any(visual["kind"] == "stock" for visual in visuals) else "No watermark is added by this service.",
                "content_safety": "Selected video frames are not independently safety-screened; human review required." if any(visual["kind"] == "stock" for visual in visuals) else "Generated procedural illustrations contain no realistic faces or invented brand logos.",
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
