import ipaddress
import json
import re
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from app.config import Settings
from app.relevance import MIN_VISUAL_SIMILARITY, MODEL_VERSION, VisualRelevanceError, score_video_file


PEXELS_LICENSE_URL = "https://www.pexels.com/license/"
PEXELS_API_URL = "https://api.pexels.com/videos/search"
MAX_STOCK_CLIP_BYTES = 100 * 1024 * 1024
MIN_RELEVANCE_SCORE = 3
MAX_VERIFICATION_CANDIDATES = 3
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "for", "from", "get",
    "had", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "there", "these", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your",
}


def _title_terms(url: str) -> set[str]:
    path = urllib.parse.urlparse(url).path
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", path)
        if len(word) > 2 and word.lower() not in STOP_WORDS and not word.isdigit()
    }


def _relevance(scene: dict[str, Any], video_url: str) -> tuple[int, int]:
    query_terms = set(_query(scene).split())
    matched_terms = query_terms & _title_terms(video_url)
    score = (5 * len(matched_terms) + len(query_terms) - 1) // len(query_terms)
    return score, len(matched_terms)


def _public_https_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(address[4][0]).is_global for address in addresses)


def _query(scene: dict[str, Any]) -> str:
    terms = []
    for token in (scene.get("visual_hint", "") + " " + scene["text"]).split():
        word = "".join(char for char in token if char.isalnum()).lower()
        if len(word) > 2 and word not in STOP_WORDS and word not in terms:
            terms.append(word)
    return " ".join(terms[:6]) or "abstract background"


def _download_video(url: str, output: Path) -> None:
    if not _public_https_url(url):
        raise ValueError("Pexels returned a non-public or non-HTTPS video URL")
    request = urllib.request.Request(url, headers={"User-Agent": "IncuBrixBroll/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response, output.open("wb") as destination:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_STOCK_CLIP_BYTES:
                raise ValueError("Pexels video exceeds the configured download limit")
            destination.write(chunk)


def _validate_video(path: Path, settings: Settings) -> None:
    try:
        result = subprocess.run(
            [
                settings.ffprobe_path, "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams or int(streams[0].get("nb_read_frames") or 0) < 1:
            raise ValueError("Downloaded video contains no decodable frames")
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Downloaded Pexels video is corrupted or unreadable") from exc


def search_pexels_clip(
    scene: dict[str, Any],
    used_video_ids: set[int],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    if not settings.pexels_api_key:
        raise ValueError("VISUAL_PROVIDER=pexels requires PEXELS_API_KEY")
    query = urllib.parse.urlencode({"query": _query(scene), "per_page": 15})
    request = urllib.request.Request(
        f"{PEXELS_API_URL}?{query}",
        headers={"Authorization": settings.pexels_api_key, "User-Agent": "IncuBrixBroll/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Pexels video search returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError("Pexels video search is unavailable") from exc

    candidates = []
    for position, video in enumerate(payload.get("videos", [])):
        if video.get("id") in used_video_ids or not video.get("url"):
            continue
        score, matched_terms = _relevance(scene, video["url"])
        if score < MIN_RELEVANCE_SCORE:
            continue
        files = [
            item for item in video.get("video_files", [])
            if item.get("link") and item.get("file_type") == "video/mp4"
        ]
        if not files:
            continue
        selected_file = max(
            files,
            key=lambda item: (
                min(int(item.get("width") or 0), 1920) * min(int(item.get("height") or 0), 1920),
                item.get("quality") == "hd",
            ),
        )
        candidates.append((score, -position, matched_terms, video, selected_file))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    verified_candidates = []
    for score, position, matched_terms, video, video_file in candidates[:MAX_VERIFICATION_CANDIDATES]:
        candidate_path = destination.with_name(f"{destination.stem}.candidate-{abs(position)}.mp4")
        try:
            _download_video(video_file["link"], candidate_path)
            _validate_video(candidate_path, settings)
            visual_similarity = score_video_file(
                candidate_path,
                scene,
                settings.ffmpeg_path,
                settings.ffprobe_path,
            )
        except VisualRelevanceError:
            candidate_path.unlink(missing_ok=True)
            raise
        except (OSError, TimeoutError, urllib.error.URLError, ValueError):
            candidate_path.unlink(missing_ok=True)
            continue
        if visual_similarity >= MIN_VISUAL_SIMILARITY:
            verified_candidates.append(
                (visual_similarity, score, position, matched_terms, video, candidate_path)
            )
        else:
            candidate_path.unlink(missing_ok=True)
    if not verified_candidates:
        return None

    visual_similarity, score, _, matched_terms, video, candidate_path = max(
        verified_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    for candidate in verified_candidates:
        if candidate[5] != candidate_path:
            candidate[5].unlink(missing_ok=True)
    candidate_path.replace(destination)
    used_video_ids.add(int(video["id"]))
    user = video.get("user") or {}
    photographer = user.get("name") or "Pexels contributor"
    photographer_url = user.get("url") or video.get("url") or "https://www.pexels.com/"
    return {
        "path": destination,
        "source": "Pexels",
        "source_url": video.get("url"),
        "license": f"Pexels License ({PEXELS_LICENSE_URL})",
        "attribution": f'Video by {photographer} on Pexels ({photographer_url})',
        "photographer_name": photographer,
        "photographer_url": photographer_url,
        "commercial_use": f"Permitted subject to the Pexels License ({PEXELS_LICENSE_URL})",
        "ai_generated": None,
        "ai_generated_status": "Not disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": "Video frames are not independently checked for relevance, safety, or logos",
        "relevance_method": f"{MODEL_VERSION} mean cosine similarity across three sampled video frames, with Pexels title-term filtering",
        "semantic_similarity": round(visual_similarity, 4),
        "title_match_score": score,
        "visual_model": MODEL_VERSION,
        "selection_reason": (
            f'Selected a unique Pexels clip with {matched_terms} title-term matches and '
            f'{visual_similarity:.3f} {MODEL_VERSION} frame similarity '
            f'(minimum {MIN_VISUAL_SIMILARITY:.2f}) for "{_query(scene)}"'
        ),
        "video_id": int(video["id"]),
    }
