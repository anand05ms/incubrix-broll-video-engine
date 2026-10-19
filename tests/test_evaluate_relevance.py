import csv

import pytest

from scripts.evaluate_relevance import evaluate


def write_ratings(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "scene_id", "rating", "reviewer", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def test_evaluation_passes_only_when_threshold_and_case_coverage_are_met(tmp_path, capsys):
    ratings_path = tmp_path / "ratings.csv"
    case_ids = {f"case-{index}" for index in range(10)}
    rows = [
        {"case_id": case_id, "scene_id": "scene-000", "rating": 4 if index < 9 else 3, "reviewer": "reviewer", "notes": ""}
        for index, case_id in enumerate(sorted(case_ids))
    ]
    write_ratings(ratings_path, rows)

    assert evaluate(ratings_path, expected_cases=10, minimum_rating=4, threshold=0.85, expected_case_ids=case_ids) == 0
    assert "90.0%" in capsys.readouterr().out


def test_evaluation_fails_when_not_all_cases_are_covered(tmp_path):
    ratings_path = tmp_path / "ratings.csv"
    case_ids = {f"case-{index}" for index in range(10)}
    rows = [
        {"case_id": "case-0", "scene_id": f"scene-{index}", "rating": 5, "reviewer": "reviewer", "notes": ""}
        for index in range(10)
    ]
    write_ratings(ratings_path, rows)

    assert evaluate(ratings_path, expected_cases=10, minimum_rating=4, threshold=0.85, expected_case_ids=case_ids) == 1


def test_evaluation_rejects_duplicate_scene_ratings(tmp_path):
    ratings_path = tmp_path / "ratings.csv"
    row = {"case_id": "case-1", "scene_id": "scene-000", "rating": 5, "reviewer": "reviewer", "notes": ""}
    write_ratings(ratings_path, [row, row])

    with pytest.raises(ValueError, match="Duplicate rating"):
        evaluate(
            ratings_path,
            expected_cases=1,
            minimum_rating=4,
            threshold=0.85,
            expected_case_ids={"case-1"},
        )
