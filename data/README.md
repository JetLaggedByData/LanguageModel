# Data

| Path | Committed | Description |
|---|---|---|
| `raw/` | No | `internet_archive_scifi_v3.txt` (~149MB) — download separately |
| `chunks/` | No | 50k instruction-format JSONL — regenerate via `prepare_dataset.py` |
| `stories/` | Yes | Pre-generated V3 stories (JSON) — read by the Story Library page |

## Prepare dataset (V2 fine-tuning)

```bash
python data/prepare_dataset.py   # build chunks/ from raw corpus
python data/verify_dataset.py    # sanity-check the JSONL
```

## Story JSON format

```json
{
  "title": "...", "logline": "...", "characters": [...],
  "chapters": [{"num": 1, "title": "...", "content": "...", "critique_score": 0.82}],
  "completion_status": "complete"
}
```
