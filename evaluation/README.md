# Human relevance evaluation pack

The ten JSON files in `inputs/` are reproducible script-to-video cases spanning distinct visual topics. Run them against the service with a configured Pexels key:

```powershell
python scripts/batch.py evaluation/inputs --base-url http://localhost:8000 --output-dir evaluation-results
```

For each completed video, watch every scene and compare its footage with the narration. Record one rating per scene in `ratings-template.csv`, using a 1–5 scale:

- 1: unrelated or misleading
- 2: weak connection
- 3: generally related but not specific
- 4: clearly illustrates the narrated meaning
- 5: precise, useful visual match

Use the input filename without `.json` as `case_id`, and the `scene_id` from that video's `timeline.json`. Add reviewer initials and notes for ratings below 4. The pass criterion is at least 85% of rated scenes scoring 4 or 5, with all ten cases represented:

```powershell
python scripts/evaluate_relevance.py evaluation/ratings-template.csv
```

The template intentionally contains no scores. A passing score must come from human review of rendered footage; this pack does not claim a result in advance. Also review `timeline.json` and the video for licence/attribution completeness, repeated clips, watermarks, safety, and any undisclosed AI-generated media. OpenCLIP frame scoring assists relevance matching but does not replace human evaluation or inspect watermarks/safety.
