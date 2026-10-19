import argparse
import csv
import sys
from pathlib import Path


def evaluate(
    ratings_path: Path,
    expected_cases: int,
    minimum_rating: int,
    threshold: float,
    expected_case_ids: set[str],
) -> int:
    with ratings_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "scene_id", "rating", "reviewer", "notes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Ratings CSV must contain these columns: {', '.join(sorted(required))}")
        ratings = list(reader)

    if not ratings:
        print("No scene ratings were provided.")
        return 1

    seen: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    passing = 0
    for row in ratings:
        case_id = (row.get("case_id") or "").strip()
        scene_id = (row.get("scene_id") or "").strip()
        reviewer = (row.get("reviewer") or "").strip()
        if not case_id or not scene_id or not reviewer:
            raise ValueError("Each rating needs a case_id, scene_id, and reviewer.")
        key = (case_id, scene_id)
        if key in seen:
            raise ValueError(f"Duplicate rating for {case_id}/{scene_id}.")
        seen.add(key)
        case_ids.add(case_id)
        try:
            rating = int(row["rating"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rating for {case_id}/{scene_id} must be an integer from 1 to 5.") from exc
        if rating < 1 or rating > 5:
            raise ValueError(f"Rating for {case_id}/{scene_id} must be an integer from 1 to 5.")
        passing += rating >= minimum_rating

    score = passing / len(ratings)
    missing_cases = expected_case_ids - case_ids
    unknown_cases = case_ids - expected_case_ids
    print(f"Scenes rated: {len(ratings)}")
    print(f"Cases covered: {len(case_ids & expected_case_ids)}/{expected_cases}")
    print(f"Scenes rated {minimum_rating}-5: {passing}/{len(ratings)} ({score:.1%})")
    print(f"Required threshold: {threshold:.1%}")
    if missing_cases:
        print(f"Cases not rated: {', '.join(sorted(missing_cases))}")
    if unknown_cases:
        print(f"Unknown case IDs: {', '.join(sorted(unknown_cases))}")
    passed = not missing_cases and not unknown_cases and score >= threshold
    print("Evaluation: PASS" if passed else "Evaluation: NOT MET")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Score human scene-relevance ratings for the 10-case evaluation pack.")
    parser.add_argument("ratings", type=Path, help="CSV with case_id, scene_id, rating, reviewer, and notes columns")
    pack_dir = Path(__file__).resolve().parent.parent / "evaluation" / "inputs"
    expected_case_ids = {path.stem for path in pack_dir.glob("*.json")}
    parser.add_argument("--expected-cases", type=int, default=len(expected_case_ids))
    parser.add_argument("--minimum-rating", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    if args.expected_cases < 1 or not 1 <= args.minimum_rating <= 5 or not 0 <= args.threshold <= 1:
        parser.error("expected-cases must be positive; minimum-rating must be 1-5; threshold must be 0-1")
    try:
        if args.expected_cases != len(expected_case_ids):
            parser.error(f"expected-cases must match the evaluation pack ({len(expected_case_ids)})")
        return evaluate(args.ratings, args.expected_cases, args.minimum_rating, args.threshold, expected_case_ids)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
