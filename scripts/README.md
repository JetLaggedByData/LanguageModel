# Scripts

| Script | Purpose |
|---|---|
| `run.sh` | Launch the Streamlit app with correct CUDA library paths |
| `pregenerate_stories.py` | Run the V3 pipeline on 5 seed prompts and commit the output to `data/stories/` |

## Usage

```bash
bash scripts/run.sh                    # auto-detects GPU
LITE_MODE=1 bash scripts/run.sh        # force CPU/lite mode

python scripts/pregenerate_stories.py          # full run (~2–3h)
python scripts/pregenerate_stories.py --dry-run  # 1 chapter each (~15 min)
```

Pre-generated stories are read by the Story Library page in the deployed app,
so this script should be run and results committed before deploying.
