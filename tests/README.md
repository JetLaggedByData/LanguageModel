# Tests

## Run

```bash
pytest tests/ -v                                      # all tests
pytest tests/ -v --cov=v3_agentic --cov-report=term  # with coverage
pytest tests/ -v --ignore=tests/test_gpu.py           # skip GPU tests (CI)
```

## Coverage

`test_v3_pipeline.py` covers:
- State schema validation and initialisation (`TestInitialState`)
- Routing logic after Critic and AdvanceChapter nodes
- `advance_chapter_node` state transitions
- `extract_json` and `trim_to_sentence` utilities
- Full graph integration with stubbed agents (no GPU required)

## Notebooks

Training and exploration notebooks kept here for reference:

| Notebook | Purpose |
|---|---|
| `V1_Training.ipynb` | V1 LSTM training run |
| `V2_Finetuning.ipynb` | V2 QLoRA fine-tuning walkthrough |
| `V3_AgenticPipeline.ipynb` | V3 pipeline end-to-end exploration |

## Checkpoints

`checkpoints/lstm_checkpoints/checkpt_best.pt` — best V1 checkpoint, loaded by the Streamlit app.
Model weights (`*.pt`, `*.safetensors`) are gitignored; only this checkpoint is committed.
