# Contest Submission Guide

This file is for the final source-code package and evaluator replay, not for day-to-day development.

## What Should Be Included

Keep the following in the contest source-code package:

- `src/`
- `scripts/`
- `weights/`
- `results/`
- `requirements.txt`
- `README.md`
- `ONLINE_DEMO_DEPLOY.md`
- this file

Keep `data/` only if the contest requires bundled data samples or if specific scripts cannot run without them. If the full raw dataset is too large, include:

- a small sample for replay
- dataset source description in the design document
- the exact filenames expected by scripts

## What Should Be Excluded

Do not include:

- `.venv/`
- `.idea/`
- `__pycache__/`
- `src/web_gcs/node_modules/`
- `src/web_gcs/dist/`
- temporary logs produced during development

## Minimal Replay Path

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Replay evaluation:

```powershell
python scripts/evaluate_v3.py --device cpu
```

Replay dashboard:

```powershell
python src/web_gcs/app.py
cd src/web_gcs
npm install
npm run dev
python ..\inference\real_time.py --scenario gcs --device cpu
```

If a public demo page is used for contest submission, prefer a static anonymous deployment rather than a local machine endpoint.

## Evidence Mapping

Recommended mapping between package contents and the design document:

- `results/confusion_matrices.png`: scenario-level confusion results
- `results/roc_curve_gcs.png`: GCS ROC evidence
- `results/ablation_bars_gcs.png`: ablation evidence
- `results/baseline_comparison_gcs.png`: baseline comparison
- `results/few_shot_gcs.csv`: few-shot table source
- `results/few_shot_learning_curves.png`: few-shot trend figure
- `results/edge_benchmark_cpu.csv`: CPU latency and throughput evidence

## Anonymity Check

Before submission, inspect:

- document headers and footers
- PDF metadata
- video opening/ending frames
- browser bookmarks and OS usernames in recordings
- repository name or deployment title if a demo link is submitted

## Recommended Packaging Command

Use [scripts/package_submission.ps1](F:/Project/HTL-UAV-IDS-V3/scripts/package_submission.ps1) to assemble a clean source package in `release/`.
