import json
import re
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_GROQ_RESPONSE_BYTES = 256 * 1024
RULE_HINTS = (
    ({"artificial", "intelligence", "ai", "machine", "learning", "neural"}, "person using artificial intelligence on computer"),
    ({"plan", "plans", "planning", "schedule", "calendar", "tasks", "task", "team", "teams", "productivity", "organize", "organizing"}, "team planning meeting with task board"),
    ({"analyze", "analyzing", "analysis", "data", "dataset", "information"}, "data analysis on computer screen"),
    ({"student", "students", "education", "school", "study", "studying", "lesson", "learn", "learning"}, "student learning with laptop and books"),
    ({"chat", "message", "messages", "communicate", "communication", "smartphone", "phone", "call"}, "person using smartphone messaging app"),
    ({"business", "businesses", "company", "companies", "customer", "customers", "entrepreneur"}, "small business owner serving customers"),
    ({"organize", "organizing", "organization", "improvement", "improve", "improving"}, "organized project task board"),
    ({"problem", "problems", "challenge", "challenges", "clutter", "overwhelm"}, "person organizing cluttered work tasks"),
    ({"growth", "grow", "growing", "progress", "success"}, "business growth chart and rising graph"),
    ({"nurse", "patient", "hospital", "doctor", "clinic", "medical"}, "healthcare professionals caring for patients"),
    ({"garden", "plant", "flower", "seed", "tree", "grow"}, "people gardening and tending green plants"),
    ({"budget", "saving", "money", "spending", "finance", "bank", "loan"}, "people budgeting and managing personal finances"),
    ({"computer", "software", "code", "digital", "technology", "data"}, "people working with computers and technology"),
    ({"travel", "journey", "city", "road", "destination", "trip"}, "travelers exploring real places and city streets"),
    ({"learn", "education", "school", "study", "teach", "lesson"}, "students learning in a classroom"),
    ({"cook", "food", "meal", "restaurant", "recipe", "kitchen"}, "people preparing and sharing food"),
    ({"health", "wellbeing", "wellness", "care", "exercise", "sleep"}, "people practicing healthy daily routines"),
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _offline_hint(text: str) -> str:
    words = {word.lower() for word in WORD_PATTERN.findall(text)}
    for terms, hint in RULE_HINTS:
        if words & terms:
            return hint
    content_words = [word for word in WORD_PATTERN.findall(text) if len(word) > 2]
    return " ".join(content_words[:6]) or "real-world illustrative footage"


def _with_offline_hints(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(scene) for scene in scenes]
    for scene in result:
        if not scene.get("visual_hint"):
            scene["visual_hint"] = _offline_hint(scene["text"])
    return result


def _groq_hints(scenes: list[dict[str, Any]], settings: Settings) -> dict[int, str]:
    targets = [
        {"index": index, "narration": scene["text"]}
        for index, scene in enumerate(scenes)
        if not scene.get("visual_hint")
    ]
    prompt = (
        "For each narration segment, write one concise visual search hint for relevant, realistic, "
        "licence-clean stock footage. Do not invent events, people, brands, or facts. Return only JSON "
        'in this shape: {"hints":[{"index":0,"visual_hint":"..."}]}. Include every supplied index exactly once.\n'
        + json.dumps(targets, ensure_ascii=False)
    )
    body = json.dumps(
        {
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You create faithful visual search hints for narrated video scenes.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw_response = response.read(MAX_GROQ_RESPONSE_BYTES + 1)
    if len(raw_response) > MAX_GROQ_RESPONSE_BYTES:
        raise ValueError("Groq scene-planning response exceeded the size limit")
    payload = json.loads(raw_response)
    content = payload["choices"][0]["message"]["content"]
    result = json.loads(content)
    hints = result["hints"]
    if not isinstance(hints, list):
        raise ValueError("Groq scene-planning response must contain a hints list")

    expected_indexes = {item["index"] for item in targets}
    parsed: dict[int, str] = {}
    for item in hints:
        if not isinstance(item, dict):
            raise ValueError("Groq scene-planning entries must be objects")
        index = item.get("index")
        hint = item.get("visual_hint")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in expected_indexes
            or index in parsed
            or not isinstance(hint, str)
        ):
            raise ValueError("Groq returned an invalid scene-planning entry")
        normalized_hint = " ".join(hint.split())
        if not normalized_hint or len(normalized_hint) > 200:
            raise ValueError("Groq returned an empty or oversized scene visual hint")
        parsed[index] = normalized_hint
    if set(parsed) != expected_indexes:
        raise ValueError("Groq did not return a visual hint for every scene")
    return parsed


def enrich_scene_visual_hints(
    scenes: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None, str]:
    if all(scene.get("visual_hint") for scene in scenes):
        return scenes, None, "provided"
    if not settings.groq_api_key:
        return (
            _with_offline_hints(scenes),
            "Groq is not configured; deterministic offline scene rules supplied visual search hints.",
            "offline-rules",
        )

    try:
        hints = _groq_hints(scenes, settings)
    except (OSError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError, urllib.error.URLError):
        return (
            _with_offline_hints(scenes),
            "Groq scene planning failed; deterministic offline scene rules supplied visual search hints.",
            "offline-rules",
        )

    enriched = [dict(scene) for scene in scenes]
    for index, hint in hints.items():
        enriched[index]["visual_hint"] = hint
    return enriched, "Narration text was sent to Groq for scene visual planning.", "groq"
