import hashlib
import ipaddress
import json
import logging
import math
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.relevance import (
    MIN_CONTEXTUAL_FALLBACK_SIMILARITY,
    MIN_VISUAL_SIMILARITY,
    MODEL_VERSION,
    VisualRelevanceError,
    score_video_file,
)
from app.visual_review import VisualReviewError, review_candidate_frame


PEXELS_LICENSE_URL = "https://www.pexels.com/license/"
PEXELS_API_URL = "https://api.pexels.com/videos/search"
PIXABAY_LICENSE_URL = "https://pixabay.com/service/license-summary/"
PIXABAY_API_URL = "https://pixabay.com/api/videos/"
OPENVERSE_API_URL = "https://api.openverse.org/v1/videos/"
OPENVERSE_TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
INTERNET_ARCHIVE_SEARCH_URL = "https://archive.org/advancedsearch.php"
INTERNET_ARCHIVE_METADATA_URL = "https://archive.org/metadata/"
INTERNET_ARCHIVE_LICENSES = {
    *(f"https://creativecommons.org/licenses/by/{version}/" for version in ("1.0", "2.0", "2.5", "3.0", "4.0")),
    *(f"https://creativecommons.org/licenses/by-sa/{version}/" for version in ("1.0", "2.0", "2.5", "3.0", "4.0")),
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/mark/1.0/",
}
WIKIMEDIA_USER_AGENT = "IncuBrixBroll/0.1 (stock-video search; contact: admin@incubrix.com)"
MAX_STOCK_CLIP_BYTES = 100 * 1024 * 1024
MIN_RELEVANCE_SCORE = 3
MAX_QUERIES_PER_SCENE = 2
MAX_RESULTS_PER_QUERY = 5
MAX_PROVIDERS_PER_SCENE = 2
PROVIDER_TIMEOUT_SECONDS = 4.0
MAX_TOTAL_SEARCH_SECONDS_PER_SCENE = 6.0
MAX_VERIFICATION_CANDIDATES = MAX_RESULTS_PER_QUERY
PIXABAY_CACHE_SECONDS = 24 * 60 * 60
_OPENVERSE_TOKEN_LOCK = threading.Lock()
_OPENVERSE_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "for", "from", "get",
    "had", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "there", "these", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your",
}
QUERY_STOP_WORDS = STOP_WORDS | {
    "able", "begin", "begins", "day", "days", "every", "increasingly", "just", "make",
    "millions", "simple", "start", "starts", "using", "uses", "ways",
}
CONTEXTUAL_SEARCH_CONCEPTS = (
    (
        {"plan", "planning", "organize", "organizing", "task", "tasks", "productivity", "distraction", "schedule", "calendar", "checklist", "time"},
        (
            "person writing a daily task list",
            "calendar schedule planning at desk",
            "focused office worker organizing tasks",
        ),
    ),
    (
        {"artificial", "intelligence", "ai", "automation", "automating", "analysis", "analyzing", "data"},
        (
            "person analyzing data on computer",
            "artificial intelligence computer",
            "business worker using digital technology",
            "computer screen data analysis office",
        ),
    ),
    (
        {"decision", "decisions", "responsible", "responsibility", "judgment", "judgement"},
        (
            "professional reviewing information before decision",
            "team discussing documents at workplace",
        ),
    ),
    (
        {"student", "students", "education", "school", "study", "studying", "teach", "lesson", "learn", "learning"},
        (
            "student using laptop for online learning",
            "student studying with books and laptop",
        ),
    ),
    (
        {"chat", "message", "messages", "communicate", "communication", "smartphone", "phone", "call"},
        (
            "person using smartphone messaging app",
            "friends chatting on mobile phone",
        ),
    ),
    (
        {"business", "businesses", "company", "companies", "customer", "customers", "entrepreneur"},
        (
            "small business owner serving customers",
            "local shop owner at storefront",
        ),
    ),
    (
        {"growth", "grow", "growing", "progress", "success", "results", "result"},
        (
            "business growth chart and rising graph",
            "team celebrating business progress",
        ),
    ),
)
logger = logging.getLogger("broll")


class StockProvider(Protocol):
    def __call__(
        self,
        scene: dict[str, Any],
        used_video_ids: set[int | str],
        destination: Path,
        settings: Settings,
    ) -> dict[str, Any] | None: ...


class StockProvidersExhausted(ValueError):
    def __init__(self, attempts: list[dict[str, str]]) -> None:
        self.attempts = attempts
        summary = "; ".join(
            f"{item['provider']} {item['status']}: {item['reason']}"
            for item in attempts
        )
        super().__init__(f"No configured stock provider returned relevant, verified footage. {summary}")


class StockProviderHTTPError(ValueError):
    def __init__(self, provider_host: str, status_code: int) -> None:
        self.provider_host = provider_host
        self.status_code = status_code
        super().__init__(f"Stock provider at {provider_host} returned HTTP {status_code}")


def _title_terms(url: str) -> set[str]:
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", path)
        if len(word) > 2 and word.lower() not in STOP_WORDS and not word.isdigit()
    }


def _relevance(scene: dict[str, Any], video_url: str) -> tuple[int, int]:
    return _relevance_text(scene, " ".join(_title_terms(video_url)))


def _relevance_text(scene: dict[str, Any], title: str) -> tuple[int, int]:
    query_terms = set(_query(scene).split())
    title_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", title)
        if len(word) > 2 and word.lower() not in STOP_WORDS and not word.isdigit()
    }
    matched_terms = query_terms & title_terms
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


def _search_query_variants(scene: dict[str, Any]) -> list[str]:
    def words(text: str) -> list[str]:
        return list(
            dict.fromkeys(
                word.lower()
                for word in re.findall(r"[A-Za-z0-9]+", text)
                if len(word) > 2 and word.lower() not in QUERY_STOP_WORDS
            )
        )

    hint = words(scene.get("visual_hint", ""))
    narration = words(scene.get("text", ""))
    variants: list[str] = []
    if hint:
        variants.append(" ".join(hint[:6]))
    scene_words = set(hint + narration)
    contextual_queries: list[str] = []
    for triggers, concepts in CONTEXTUAL_SEARCH_CONCEPTS:
        if scene_words & triggers:
            contextual_queries.extend(concepts[:2])
    variants.extend(contextual_queries[:2])
    if narration:
        narrative_focus = list(dict.fromkeys(narration[:3] + narration[-3:]))
        variants.append(" ".join(narrative_focus[:6]))
        combined = list(dict.fromkeys(hint[:3] + narration[-3:]))
        variants.append(" ".join(combined[:6]))
    result = list(dict.fromkeys(" ".join(variant.split()) for variant in variants if variant))
    return result[:MAX_QUERIES_PER_SCENE] or [_query(scene)]


def _call_provider_with_timeout(
    provider: StockProvider,
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, Exception | None, bool]:
    result_queue: queue.Queue[tuple[dict[str, Any] | None, Exception | None]] = queue.Queue(maxsize=1)
    timed_out = threading.Event()

    def run() -> None:
        try:
            result = provider(scene, used_video_ids, destination, settings)
            error = None
        except Exception as exc:
            result = None
            error = exc
        if timed_out.is_set():
            destination.unlink(missing_ok=True)
            return
        result_queue.put((result, error))
        if timed_out.is_set():
            destination.unlink(missing_ok=True)

    threading.Thread(target=run, daemon=True, name="broll-provider-search").start()
    try:
        result, error = result_queue.get(timeout=max(0.001, timeout_seconds))
        return result, error, False
    except queue.Empty:
        timed_out.set()
        try:
            result, error = result_queue.get_nowait()
            return result, error, False
        except queue.Empty:
            destination.unlink(missing_ok=True)
            return None, TimeoutError(f"provider exceeded {timeout_seconds:.1f} seconds"), True


def _download_video(url: str, output: Path) -> None:
    if not _public_https_url(url):
        raise ValueError("Stock provider returned a non-public or non-HTTPS video URL")
    request = urllib.request.Request(url, headers={"User-Agent": WIKIMEDIA_USER_AGENT})
    with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response, output.open("wb") as destination:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_STOCK_CLIP_BYTES:
                raise ValueError("Stock video exceeds the configured download limit")
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
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams or int(streams[0].get("nb_read_frames") or 0) < 1:
            raise ValueError("Downloaded video contains no decodable frames")
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Downloaded stock video is corrupted or unreadable") from exc


def _allowed_commons_license(metadata: dict[str, Any]) -> tuple[str, str] | None:
    short_name = metadata.get("LicenseShortName", {}).get("value", "")
    license_url = metadata.get("LicenseUrl", {}).get("value", "")
    normalized = short_name.strip().lower()
    if normalized in {"cc0", "public domain", "pd"}:
        return short_name, license_url
    if normalized.startswith(("cc by ", "cc by-sa ")):
        return short_name, license_url
    return None


def search_wikimedia_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:video {_query(scene)}",
            "gsrnamespace": 6,
            "gsrlimit": MAX_RESULTS_PER_QUERY,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "format": "json",
            "formatversion": 2,
        }
    )
    request = urllib.request.Request(
        f"{WIKIMEDIA_API_URL}?{query}",
        headers={"User-Agent": WIKIMEDIA_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Wikimedia Commons search returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError("Wikimedia Commons video search is unavailable") from exc

    candidates = []
    for position, page in enumerate(payload.get("query", {}).get("pages", [])):
        video_id = str(page.get("pageid", ""))
        if not video_id or video_id in used_video_ids:
            continue
        image_info = (page.get("imageinfo") or [{}])[0]
        video_url = image_info.get("url", "")
        if not video_url or not image_info.get("mime", "").startswith("video/"):
            continue
        license_info = _allowed_commons_license(image_info.get("extmetadata") or {})
        if not license_info:
            continue
        score, matched_terms = _relevance(
            scene,
            f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page.get('title', ''), safe='')}",
        )
        if matched_terms == 0:
            continue
        candidates.append((score, -position, matched_terms, page, image_info, license_info))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    verified_candidates = []
    for score, position, matched_terms, page, image_info, license_info in candidates[:MAX_VERIFICATION_CANDIDATES]:
        verified = _verify_stock_file(
            image_info["url"], scene, destination, settings, abs(position) + 3000
        )
        if verified:
            visual_similarity, candidate_path, visual_review = verified
            verified_candidates.append(
                (
                    visual_similarity, score, position, matched_terms, page, image_info,
                    license_info, visual_review, candidate_path,
                )
            )
    if not verified_candidates:
        return None

    visual_similarity, score, _, matched_terms, page, image_info, license_info, visual_review, candidate_path = max(
        verified_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    for candidate in verified_candidates:
        if candidate[8] != candidate_path:
            candidate[8].unlink(missing_ok=True)
    candidate_path.replace(destination)
    used_video_ids.add(str(page["pageid"]))

    extmetadata = image_info.get("extmetadata") or {}
    artist = extmetadata.get("Artist", {}).get("value", "")
    artist = re.sub(r"<[^>]+>", "", artist).strip() or "Wikimedia Commons contributor"
    license_name, license_url = license_info
    source_url = image_info.get("descriptionurl") or (
        f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page.get('title', ''), safe='')}"
    )
    return {
        "path": destination,
        "source": "Wikimedia Commons",
        "source_url": source_url,
        "license": f"{license_name} ({license_url})" if license_url else license_name,
        "license_url": license_url,
        "attribution": f"{artist} ({license_name})",
        "photographer_name": artist,
        "photographer_url": source_url,
        "commercial_use": f"Permitted subject to {license_name}; follow its attribution and share-alike terms.",
        "ai_generated": None,
        "ai_generated_status": "Not consistently disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": _review_summary(visual_review),
        "visual_review": visual_review,
        "logo_status": _review_flag(visual_review, "brand_logo_visible"),
        "safety_status": _review_flag(visual_review, "safety_concern"),
        "relevance_method": f"{MODEL_VERSION} semantic frame scoring plus Commons-title overlap",
        "semantic_similarity": round(visual_similarity, 4),
        "visual_model": MODEL_VERSION,
        "title_match_score": score,
        "title_matches": matched_terms,
        "candidate_count": len(verified_candidates),
        "selection_reason": (
            f"Selected a unique Wikimedia Commons video under {license_name}; "
            f"matched {matched_terms} title terms and scored {visual_similarity:.3f} against the scene."
        ),
        "video_id": str(page["pageid"]),
    }


def _internet_archive_queries(scene: dict[str, Any]) -> list[str]:
    return _search_query_variants(scene)


def _internet_archive_license(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    license_url = value.strip().lower().replace("http://", "https://")
    if license_url not in INTERNET_ARCHIVE_LICENSES:
        return None
    if "/licenses/by-sa/" in license_url:
        name = "CC BY-SA"
    elif "/licenses/by/" in license_url:
        name = "CC BY"
    elif "/publicdomain/zero/" in license_url:
        name = "CC0"
    else:
        name = "Public Domain Mark"
    return name, license_url


def search_internet_archive_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    candidates: dict[str, dict[str, Any]] = {}
    for query_text in _internet_archive_queries(scene):
        query = urllib.parse.urlencode(
            {
                "q": f"mediatype:movies AND ({query_text})",
                "fl[]": ["identifier", "title", "description", "licenseurl"],
                "rows": MAX_RESULTS_PER_QUERY,
                "page": 1,
                "output": "json",
            },
            doseq=True,
        )
        payload = _search_json(f"{INTERNET_ARCHIVE_SEARCH_URL}?{query}")
        docs = payload.get("response", {}).get("docs", [])
        for item in docs:
            identifier = item.get("identifier")
            if identifier and identifier not in used_video_ids:
                existing = candidates.get(identifier)
                if existing is None:
                    candidates[identifier] = item

    ranked = sorted(
        candidates.values(),
        key=lambda item: _relevance_text(
            scene,
            f"{item.get('title', '')} {item.get('description', '')}",
        ),
        reverse=True,
    )
    verified_candidates = []
    for position, item in enumerate(ranked[:MAX_VERIFICATION_CANDIDATES]):
        identifier = str(item["identifier"])
        encoded_identifier = urllib.parse.quote(identifier, safe="")
        metadata_payload = _search_json(
            f"{INTERNET_ARCHIVE_METADATA_URL}{encoded_identifier}/metadata"
        )
        metadata = metadata_payload.get("result", {})
        license_info = _internet_archive_license(metadata.get("licenseurl"))
        if not license_info:
            continue
        files_payload = _search_json(
            f"{INTERNET_ARCHIVE_METADATA_URL}{encoded_identifier}/files"
        )
        files = files_payload.get("result", [])
        video_files = [
            file_info
            for file_info in files
            if file_info.get("source") == "original"
            and str(file_info.get("format", "")).lower() in {"mpeg4", "matroska", "ogg video", "quicktime"}
            and Path(str(file_info.get("name", ""))).suffix.lower()
            in {".mp4", ".m4v", ".webm", ".mkv", ".mov"}
            and 0 < int(file_info.get("size") or 0) <= MAX_STOCK_CLIP_BYTES
        ]
        if not video_files:
            continue

        title = str(metadata.get("title") or item.get("title") or identifier)
        description = str(metadata.get("description") or item.get("description") or "")
        score, matched_terms = _relevance_text(scene, f"{title} {description}")
        file_info = max(video_files, key=lambda entry: int(entry["size"]))
        filename = str(file_info["name"])
        video_url = (
            f"https://archive.org/download/{urllib.parse.quote(identifier, safe='')}/"
            f"{urllib.parse.quote(filename, safe='')}"
        )
        verified = _verify_stock_file(
            video_url, scene, destination, settings, position + 5000
        )
        if not verified:
            continue
        visual_score, candidate_path, visual_review = verified
        verified_candidates.append(
            (
                visual_score,
                score,
                position,
                matched_terms,
                identifier,
                metadata,
                license_info,
                visual_review,
                candidate_path,
            )
        )
    if not verified_candidates:
        return None

    (
        visual_score,
        score,
        _,
        matched_terms,
        identifier,
        metadata,
        license_info,
        visual_review,
        candidate_path,
    ) = max(verified_candidates, key=lambda item: (item[0], item[1], -item[2]))
    for candidate in verified_candidates:
        if candidate[8] != candidate_path:
            candidate[8].unlink(missing_ok=True)
    candidate_path.replace(destination)
    used_video_ids.add(identifier)
    license_name, license_url = license_info
    source_url = f"https://archive.org/details/{urllib.parse.quote(identifier, safe='')}"
    return {
        "path": destination,
        "source": "Internet Archive",
        "source_url": source_url,
        "license": f"{license_name} ({license_url})",
        "license_url": license_url,
        "attribution": str(metadata.get("creator") or "Internet Archive contributor"),
        "photographer_name": str(metadata.get("creator") or "Internet Archive contributor"),
        "photographer_url": source_url,
        "commercial_use": f"Permitted subject to the {license_name} license and its terms.",
        "ai_generated": None,
        "ai_generated_status": "Not consistently disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": _review_summary(visual_review),
        "visual_review": visual_review,
        "logo_status": _review_flag(visual_review, "brand_logo_visible"),
        "safety_status": _review_flag(visual_review, "safety_concern"),
        "relevance_method": f"{MODEL_VERSION} frame scoring plus Internet Archive metadata search",
        "semantic_similarity": round(visual_score, 4),
        "visual_model": MODEL_VERSION,
        "title_match_score": score,
        "title_matches": matched_terms,
        "candidate_count": len(verified_candidates),
        "selection_reason": (
            f"Selected a licensed Internet Archive video scoring {visual_score:.3f} "
            f"against the scene under {license_name}."
        ),
        "video_id": identifier,
    }


def search_pexels_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    if not settings.pexels_api_key:
        raise ValueError("VISUAL_PROVIDER=pexels requires PEXELS_API_KEY")
    query = urllib.parse.urlencode({"query": _query(scene), "per_page": MAX_RESULTS_PER_QUERY})
    request = urllib.request.Request(
        f"{PEXELS_API_URL}?{query}",
        headers={"Authorization": settings.pexels_api_key, "User-Agent": "IncuBrixBroll/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
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
        verified = _verify_stock_file(
            video_file["link"], scene, destination, settings, abs(position) + 4000
        )
        if verified:
            visual_similarity, candidate_path, visual_review = verified
            verified_candidates.append(
                (visual_similarity, score, position, matched_terms, video, visual_review, candidate_path)
            )
    if not verified_candidates:
        return None

    visual_similarity, score, _, matched_terms, video, visual_review, candidate_path = max(
        verified_candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    for candidate in verified_candidates:
        if candidate[6] != candidate_path:
            candidate[6].unlink(missing_ok=True)
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
        "license_url": PEXELS_LICENSE_URL,
        "attribution": f'Video by {photographer} on Pexels ({photographer_url})',
        "photographer_name": photographer,
        "photographer_url": photographer_url,
        "commercial_use": f"Permitted subject to the Pexels License ({PEXELS_LICENSE_URL})",
        "ai_generated": None,
        "ai_generated_status": "Not disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": _review_summary(visual_review),
        "visual_review": visual_review,
        "logo_status": _review_flag(visual_review, "brand_logo_visible"),
        "safety_status": _review_flag(visual_review, "safety_concern"),
        "relevance_method": f"{MODEL_VERSION} mean cosine similarity across three sampled video frames, with Pexels title-term filtering",
        "semantic_similarity": round(visual_similarity, 4),
        "title_match_score": score,
        "candidate_count": len(verified_candidates),
        "visual_model": MODEL_VERSION,
        "selection_reason": (
            f'Selected a unique Pexels clip with {matched_terms} title-term matches and '
            f'{visual_similarity:.3f} {MODEL_VERSION} frame similarity '
            f'(minimum {MIN_VISUAL_SIMILARITY:.2f}) for "{_query(scene)}"'
        ),
        "video_id": int(video["id"]),
    }


def _search_json(url: str, headers: dict[str, str] | None = None, data: bytes | None = None) -> dict[str, Any]:
    started = time.monotonic()
    provider_host = urllib.parse.urlsplit(url).hostname or "unknown"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": WIKIMEDIA_USER_AGENT, **(headers or {})},
        data=data,
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        logger.warning(
            "Stock catalog request failed",
            extra={
                "provider_host": provider_host,
                "request_method": request.method,
                "http_status": exc.code,
                "request_duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        raise StockProviderHTTPError(provider_host, exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        timed_out = isinstance(reason, TimeoutError)
        logger.warning(
            "Stock catalog request failed",
            extra={
                "provider_host": provider_host,
                "request_method": request.method,
                "failure_type": type(reason).__name__,
                "request_duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        if timed_out:
            raise ValueError("Stock provider request timed out") from exc
        raise ValueError("Stock provider search is unavailable") from exc
    except json.JSONDecodeError as exc:
        logger.warning(
            "Stock catalog request failed",
            extra={
                "provider_host": provider_host,
                "request_method": request.method,
                "failure_type": type(exc).__name__,
                "request_duration_ms": round((time.monotonic() - started) * 1000),
            },
        )
        raise ValueError("Stock provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stock provider returned an invalid response")
    logger.info(
        "Stock catalog request completed",
        extra={
            "provider_host": provider_host,
            "request_method": request.method,
            "http_status": 200,
            "request_duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return payload


def _cached_pixabay_search(url: str, settings: Settings) -> dict[str, Any]:
    cache_dir = settings.data_dir / "provider-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pixabay-{hashlib.sha256(url.encode()).hexdigest()}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if time.time() - float(cached["cached_at"]) < PIXABAY_CACHE_SECONDS:
            return cached["payload"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        pass
    payload = _search_json(url)
    temporary_path = cache_path.with_name(
        f"{cache_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary_path.write_text(
        json.dumps({"cached_at": time.time(), "payload": payload}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)
    return payload


def _review_summary(review: dict[str, Any] | None) -> str:
    if not review:
        return "No Groq visual review was performed; human review required."
    return (
        f"Groq {review['model']} reviewed one frame; relevance/logo/safety verdicts are "
        "un-calibrated and human review is still required."
    )


def _review_flag(review: dict[str, Any] | None, key: str) -> str:
    if not review:
        return "not-reviewed"
    return "flagged" if review[key] else "not-flagged"


def _verify_stock_file(
    video_url: str,
    scene: dict[str, Any],
    destination: Path,
    settings: Settings,
    candidate_number: int,
) -> tuple[float, Path, dict[str, Any] | None] | None:
    candidate_path = destination.with_name(f"{destination.stem}.candidate-{candidate_number}.mp4")
    try:
        _download_video(video_url, candidate_path)
        _validate_video(candidate_path, settings)
        score = score_video_file(candidate_path, scene, settings.ffmpeg_path, settings.ffprobe_path)
    except VisualRelevanceError:
        candidate_path.unlink(missing_ok=True)
        raise
    except VisualReviewError:
        candidate_path.unlink(missing_ok=True)
        raise
    except (OSError, TimeoutError, urllib.error.URLError, ValueError):
        candidate_path.unlink(missing_ok=True)
        return None
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        candidate_path.unlink(missing_ok=True)
        return None
    review = None
    if score < MIN_VISUAL_SIMILARITY + 0.08:
        try:
            review = review_candidate_frame(candidate_path, scene, settings)
        except VisualReviewError:
            candidate_path.unlink(missing_ok=True)
            raise
        if review and (
            not review["relevant"]
            or review["brand_logo_visible"]
            or review["safety_concern"]
        ):
            candidate_path.unlink(missing_ok=True)
            return None
    return score, candidate_path, review


def search_pixabay_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    if not settings.pixabay_api_key:
        raise ValueError("VISUAL_PROVIDER=pixabay requires PIXABAY_API_KEY")
    query = urllib.parse.urlencode(
        {"key": settings.pixabay_api_key, "q": _query(scene), "per_page": MAX_RESULTS_PER_QUERY, "safesearch": "true"}
    )
    payload = _cached_pixabay_search(f"{PIXABAY_API_URL}?{query}", settings)
    candidates: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]] = []
    for position, video in enumerate(payload.get("hits", [])):
        video_id = str(video.get("id", ""))
        page_url = video.get("pageURL", "")
        if not video_id or video_id in used_video_ids or not page_url:
            continue
        score, matched_terms = _relevance_text(
            scene, f"{page_url} {video.get('tags', '')}"
        )
        if matched_terms == 0 or score < MIN_RELEVANCE_SCORE:
            continue
        variants = video.get("videos") or {}
        files = [item for item in variants.values() if item.get("url")]
        if not files:
            continue
        selected = max(
            files,
            key=lambda item: (
                min(int(item.get("width") or 0), 1920)
                * min(int(item.get("height") or 0), 1080),
                -int(item.get("size") or 0),
            ),
        )
        candidates.append((score, -position, matched_terms, video, selected))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    verified: list[tuple[float, int, int, int, dict[str, Any], dict[str, Any] | None, Path]] = []
    for score, position, matched_terms, video, selected in candidates[:MAX_VERIFICATION_CANDIDATES]:
        result = _verify_stock_file(
            selected["url"], scene, destination, settings, abs(position) + 1000
        )
        if result:
            visual_score, path, visual_review = result
            verified.append((visual_score, score, position, matched_terms, video, visual_review, path))
            if visual_score >= MIN_VISUAL_SIMILARITY:
                break
    if not verified:
        return None
    visual_score, score, _, matched_terms, video, visual_review, chosen_path = max(
        verified, key=lambda item: (item[0], item[1], item[2])
    )
    for candidate in verified:
        if candidate[6] != chosen_path:
            candidate[6].unlink(missing_ok=True)
    chosen_path.replace(destination)
    video_id = str(video["id"])
    used_video_ids.add(video_id)
    photographer = str(video.get("user") or "Pixabay contributor")
    source_url = str(video.get("pageURL") or "https://pixabay.com/videos/")
    return {
        "path": destination,
        "source": "Pixabay",
        "source_url": source_url,
        "license": f"Pixabay Content License ({PIXABAY_LICENSE_URL})",
        "license_url": PIXABAY_LICENSE_URL,
        "attribution": f"Video by {photographer} on Pixabay",
        "photographer_name": photographer,
        "photographer_url": source_url,
        "commercial_use": f"See Pixabay Content License ({PIXABAY_LICENSE_URL})",
        "ai_generated": None,
        "ai_generated_status": "Not disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": _review_summary(visual_review),
        "visual_review": visual_review,
        "logo_status": _review_flag(visual_review, "brand_logo_visible"),
        "safety_status": _review_flag(visual_review, "safety_concern"),
        "relevance_method": f"{MODEL_VERSION} frame scoring plus Pixabay tags and title-term filtering",
        "semantic_similarity": round(visual_score, 4),
        "visual_model": MODEL_VERSION,
        "title_match_score": score,
        "title_matches": matched_terms,
        "candidate_count": len(verified),
        "selection_reason": f"Selected a Pixabay video scoring {visual_score:.3f} against the scene.",
        "video_id": video_id,
    }


def _openverse_token(settings: Settings) -> str:
    if not settings.openverse_client_id or not settings.openverse_client_secret:
        raise ValueError("VISUAL_PROVIDER=openverse requires OPENVERSE_CLIENT_ID and OPENVERSE_CLIENT_SECRET")
    cache_key = hashlib.sha256(settings.openverse_client_id.encode()).hexdigest()
    with _OPENVERSE_TOKEN_LOCK:
        cached_token = _OPENVERSE_TOKEN_CACHE.get(cache_key)
        if cached_token and cached_token[1] > time.time() + 30:
            return cached_token[0]
    form = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": settings.openverse_client_id,
            "client_secret": settings.openverse_client_secret,
        }
    ).encode()
    try:
        payload = _search_json(
            OPENVERSE_TOKEN_URL,
            {"Content-Type": "application/x-www-form-urlencoded"},
            form,
        )
    except StockProviderHTTPError as exc:
        if exc.status_code == 401:
            raise ValueError(
                "Openverse rejected the configured client credentials (HTTP 401); "
                "verify OPENVERSE_CLIENT_ID and OPENVERSE_CLIENT_SECRET."
            ) from exc
        raise
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("Openverse did not issue an access token")
    expires_in = max(60, int(payload.get("expires_in") or 3600))
    with _OPENVERSE_TOKEN_LOCK:
        _OPENVERSE_TOKEN_CACHE[cache_key] = (token, time.time() + expires_in)
    return token


def search_openverse_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    token = _openverse_token(settings)
    query = urllib.parse.urlencode({"q": _query(scene), "page_size": MAX_RESULTS_PER_QUERY})
    payload = _search_json(
        f"{OPENVERSE_API_URL}?{query}",
        {"Authorization": f"Bearer {token}"},
    )
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for position, item in enumerate(payload.get("results", [])):
        asset_id = str(item.get("id") or item.get("foreign_landing_url") or "")
        video_url = item.get("url")
        license_name = str(item.get("license") or "").lower().replace("-", "")
        license_url = str(item.get("license_url") or "")
        if not asset_id or asset_id in used_video_ids or not video_url:
            continue
        if license_name not in {"cc0", "pdm", "by", "bysa"}:
            continue
        score, matched_terms = _relevance_text(
            scene, f"{item.get('title', '')} {item.get('foreign_landing_url', '')}"
        )
        if matched_terms == 0:
            continue
        candidates.append((score, -position, matched_terms, item))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    verified: list[tuple[float, int, int, int, dict[str, Any], dict[str, Any] | None, Path]] = []
    for score, position, matched_terms, item in candidates[:MAX_VERIFICATION_CANDIDATES]:
        result = _verify_stock_file(
            item["url"], scene, destination, settings, abs(position) + 2000
        )
        if result:
            visual_score, path, visual_review = result
            verified.append((visual_score, score, position, matched_terms, item, visual_review, path))
    if not verified:
        return None
    visual_score, score, _, matched_terms, item, visual_review, chosen_path = max(
        verified, key=lambda candidate: (candidate[0], candidate[1], candidate[2])
    )
    for candidate in verified:
        if candidate[6] != chosen_path:
            candidate[6].unlink(missing_ok=True)
    chosen_path.replace(destination)
    asset_id = str(item.get("id") or item.get("foreign_landing_url"))
    used_video_ids.add(asset_id)
    license_name = str(item.get("license") or "Unknown").upper()
    license_url = str(item.get("license_url") or "")
    creator = str(item.get("creator") or "Openverse contributor")
    source_url = str(item.get("foreign_landing_url") or item.get("url") or "")
    return {
        "path": destination,
        "source": "Openverse",
        "source_url": source_url,
        "license": f"{license_name} ({license_url})" if license_url else license_name,
        "license_url": license_url,
        "attribution": f"{creator} ({license_name})",
        "photographer_name": creator,
        "photographer_url": source_url,
        "commercial_use": f"Permitted subject to the {license_name} license and its terms.",
        "ai_generated": None,
        "ai_generated_status": "Not consistently disclosed by the provider; requires review",
        "watermark_status": "Not independently verified",
        "content_review": _review_summary(visual_review),
        "visual_review": visual_review,
        "logo_status": _review_flag(visual_review, "brand_logo_visible"),
        "safety_status": _review_flag(visual_review, "safety_concern"),
        "relevance_method": f"{MODEL_VERSION} semantic frame scoring plus Openverse title-term filtering",
        "semantic_similarity": round(visual_score, 4),
        "visual_model": MODEL_VERSION,
        "title_match_score": score,
        "title_matches": matched_terms,
        "candidate_count": len(verified),
        "selection_reason": f"Selected an Openverse video under {license_name}, scoring {visual_score:.3f}.",
        "video_id": asset_id,
    }


def search_stock_clip(
    scene: dict[str, Any],
    used_video_ids: set[int | str],
    destination: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    providers: dict[str, StockProvider] = {
        "wikimedia": search_wikimedia_clip,
        "pexels": search_pexels_clip,
        "pixabay": search_pixabay_clip,
        "openverse": search_openverse_clip,
    }
    if settings.internet_archive_enabled:
        providers["internet_archive"] = search_internet_archive_clip
    configured_order = list(settings.visual_provider_order)
    if settings.internet_archive_enabled and "internet_archive" not in configured_order:
        configured_order.append("internet_archive")
    elif not settings.internet_archive_enabled:
        configured_order = [provider for provider in configured_order if provider != "internet_archive"]
    if any(provider not in providers for provider in configured_order):
        raise ValueError("VISUAL_PROVIDER_ORDER contains an unsupported stock provider")
    if settings.visual_provider == "auto":
        ordered = configured_order
    elif settings.visual_provider in providers:
        ordered = [settings.visual_provider] + [
            provider for provider in configured_order if provider != settings.visual_provider
        ]
    else:
        raise ValueError(f"Unsupported stock provider: {settings.visual_provider}")

    attempts: list[dict[str, str]] = []
    search_started = time.monotonic()
    query_variants = _search_query_variants(scene)[:MAX_QUERIES_PER_SCENE]
    verified: list[tuple[float, float, int, dict[str, Any], dict[str, str]]] = []
    scene_id = str(scene.get("_scene_id") or destination.stem)

    def log_selection_summary(
        winner_info: tuple[float, float, int, dict[str, Any], dict[str, str]] | None,
        fallback_used: bool,
    ) -> None:
        logger.info(
            "Stock scene candidate selection completed",
            extra={
                "scene_id": scene_id,
                "generated_queries": query_variants,
                "visual_status": "stock_broll" if winner_info else "voiceover_primary",
                "providers_attempted": list(dict.fromkeys(item["provider"] for item in attempts)),
                "provider_attempts": attempts,
                "search_duration_ms": round((time.monotonic() - search_started) * 1000),
                "selected_provider": winner_info[4]["provider"] if winner_info else None,
                "provider": winner_info[4]["provider"] if winner_info else None,
                "candidate_count": sum(int(item[3].get("candidate_count") or 1) for item in verified),
                "best_score": winner_info[0] if winner_info else None,
                "selected_candidate": (
                    str(winner_info[3].get("video_id") or winner_info[3].get("source_url") or "unknown")
                    if winner_info
                    else None
                ),
                "fallback_used": fallback_used,
            },
        )

    active_providers: list[str] = []
    for provider in ordered:
        missing_credential = (
            "PIXABAY_API_KEY"
            if provider == "pixabay" and not settings.pixabay_api_key
            else "OPENVERSE_CLIENT_ID and OPENVERSE_CLIENT_SECRET"
            if provider == "openverse"
            and not (settings.openverse_client_id and settings.openverse_client_secret)
            else "PEXELS_API_KEY"
            if provider == "pexels" and not settings.pexels_api_key
            else None
        )
        if missing_credential:
            attempts.append(
                {
                    "provider": provider,
                    "status": "skipped",
                    "reason": f"required credentials are not configured ({missing_credential})",
                }
            )
            continue
        if len(active_providers) < MAX_PROVIDERS_PER_SCENE:
            active_providers.append(provider)
        else:
            attempts.append(
                {
                    "provider": provider,
                    "status": "skipped",
                    "reason": f"per-scene provider limit reached ({MAX_PROVIDERS_PER_SCENE})",
                }
            )

    for provider_index, provider in enumerate(active_providers):
        if provider_index >= len(query_variants):
            break
        query = query_variants[provider_index]
        remaining_seconds = MAX_TOTAL_SEARCH_SECONDS_PER_SCENE - (time.monotonic() - search_started)
        if remaining_seconds <= 0:
            attempts.append(
                {
                    "provider": provider,
                    "status": "timeout",
                    "reason": "per-scene stock search time budget was exhausted",
                }
            )
            break
        query_index = provider_index
        candidate_path = destination.with_name(
            f"{destination.stem}.{provider}.{query_index}.candidate{destination.suffix}"
        )
        query_scene = {**scene, "visual_hint": query}
        attempt = {
            "provider": provider,
            "query": query[:160],
            "query_phase": "scene_specific",
            "status": "no_relevant_clip",
            "reason": "no licensed clip passed semantic relevance and media verification",
        }
        provider_used_ids = set(used_video_ids)
        provider_timeout = min(PROVIDER_TIMEOUT_SECONDS, remaining_seconds)
        attempt_started = time.monotonic()
        selected, provider_error, timed_out = _call_provider_with_timeout(
            providers[provider],
            query_scene,
            provider_used_ids,
            candidate_path,
            settings,
            provider_timeout,
        )
        attempt["duration_ms"] = str(round((time.monotonic() - attempt_started) * 1000))
        if timed_out:
            attempt.update(status="timeout", reason=f"provider exceeded {provider_timeout:.1f} seconds")
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        if provider_error:
            attempt.update(status="unavailable", reason=str(provider_error))
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        if selected is None:
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue

        selected = dict(selected)
        selected_path = selected.get("path")
        if not isinstance(selected_path, Path) or selected_path != candidate_path:
            attempt.update(
                status="unavailable",
                reason="provider did not return the requested path for its verified media",
            )
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        if not candidate_path.is_file():
            attempt.update(
                status="unavailable",
                reason="provider reported a verified clip but did not produce its media file",
            )
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        media_path = candidate_path
        candidate_id = selected.get("video_id")
        if (
            isinstance(candidate_id, (int, str))
            and not isinstance(candidate_id, bool)
            and candidate_id in used_video_ids
        ):
            attempt.update(
                status="no_relevant_clip",
                reason="provider returned a duplicate clip already used in this job",
            )
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        attempt.update(
            status="candidate_verified",
            reason="licensed clip passed media verification and received a finite semantic score",
        )
        raw_score = selected.get("semantic_similarity")
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float))
            and not isinstance(raw_score, bool)
            and math.isfinite(raw_score)
            and -1.0 <= raw_score <= 1.0
            else None
        )
        if score is None:
            attempt.update(
                status="no_relevant_clip",
                reason="candidate did not provide a finite semantic score in the valid cosine range",
            )
            candidate_path.unlink(missing_ok=True)
            attempts.append(attempt)
            continue
        attempt["relevance_score"] = f"{score:.4f}"
        raw_candidate_count = selected.get("candidate_count", 1)
        candidate_count = (
            int(raw_candidate_count)
            if isinstance(raw_candidate_count, int)
            and not isinstance(raw_candidate_count, bool)
            and raw_candidate_count > 0
            else 1
        )
        attempt["candidate_count"] = str(candidate_count)
        attempts.append(attempt)
        raw_title_score = selected.get("title_match_score", 0)
        title_score = (
            float(raw_title_score)
            if isinstance(raw_title_score, (int, float))
            and not isinstance(raw_title_score, bool)
            and math.isfinite(raw_title_score)
            else 0.0
        )
        selected["path"] = media_path
        verified.append((score, title_score, -provider_index, selected, attempt))
        break

    if not verified:
        log_selection_summary(None, False)
        raise StockProvidersExhausted(attempts)

    winner = max(verified, key=lambda item: (item[0], item[1], item[2]))
    fallback_used = winner[0] < MIN_VISUAL_SIMILARITY
    if winner[0] < MIN_CONTEXTUAL_FALLBACK_SIMILARITY:
        fallback_reason = "low_confidence_contextual_match"
    elif fallback_used:
        fallback_reason = "no_high_confidence_match"
    else:
        fallback_reason = None
    result = dict(winner[3])
    winner_attempt = winner[4]
    for candidate in verified:
        candidate_path = candidate[3].get("path")
        if candidate is winner:
            if isinstance(candidate_path, Path) and candidate_path.name and candidate_path.is_file():
                candidate_path.replace(destination)
                result["path"] = destination
            continue
        if isinstance(candidate_path, Path):
            candidate_path.unlink(missing_ok=True)
    winner_attempt.update(
        status="selected",
        reason=(
            "highest semantic score among verified provider candidates; "
            "below the normal confidence threshold"
            if fallback_used
            else "highest semantic score among verified provider candidates"
        ),
    )
    result["provider_attempts"] = attempts
    result["fallback"] = fallback_used
    result["relevance_score"] = round(winner[0], 4)
    if fallback_used:
        result["fallback_reason"] = fallback_reason
        result["selection_reason"] = (
            f"{result.get('selection_reason', 'Selected the best verified stock candidate')} "
            f"Used as a {'last-resort ' if fallback_reason == 'low_confidence_contextual_match' else ''}"
            f"contextual fallback at {winner[0]:.3f}; normal threshold is "
            f"{MIN_VISUAL_SIMILARITY:.2f} and contextual fallback floor is "
            f"{MIN_CONTEXTUAL_FALLBACK_SIMILARITY:.2f}."
        )
        if fallback_reason == "low_confidence_contextual_match":
            result["fallback_decision"] = (
                f"Selected the highest-scoring media/license-verified candidate at {winner[0]:.3f}; "
                f"no candidate reached the contextual floor {MIN_CONTEXTUAL_FALLBACK_SIMILARITY:.2f}. "
                "This last-resort match requires human relevance review."
            )
        else:
            result["fallback_decision"] = (
                f"Selected the best contextual candidate at {winner[0]:.3f}; no candidate "
                f"reached the normal relevance threshold {MIN_VISUAL_SIMILARITY:.2f}."
            )
    failed_before_selection = [
        attempt["provider"]
        for attempt in attempts
        if attempt is not winner_attempt
        and attempt["status"] in {"unavailable", "no_relevant_clip"}
    ]
    if failed_before_selection:
        provider_fallback = (
            f"Selected {winner_attempt['provider']} after "
            + ", ".join(dict.fromkeys(failed_before_selection))
            + " did not provide a relevant verified clip."
        )
        if result.get("fallback_decision"):
            result["fallback_decision"] += f" {provider_fallback}"
        else:
            result["fallback_decision"] = provider_fallback
    video_id = result.get("video_id")
    if isinstance(video_id, (int, str)) and not isinstance(video_id, bool):
        used_video_ids.add(video_id)
    result["visual_status"] = "stock_broll"
    result["search_duration_ms"] = round((time.monotonic() - search_started) * 1000)
    log_selection_summary(winner, fallback_used)
    return result
