
# HTL-UAV-IDS

HTL-UAV-IDS is a lightweight intrusion detection system for heterogeneous UAV communication environments. The project focuses on three constraints that commonly appear in edge security deployment:

- heterogeneous feature spaces across deployment points
- limited labeled data in target scenarios
- tight latency and parameter budgets on edge devices

The core method combines a learnable `DomainProjector`, a shared frozen `Backbone`, and a lightweight classifier. Source-domain knowledge is first learned from a large-scale IoT intrusion dataset and then transferred to UAV-side scenarios with low-cost target adaptation.

## Project Scope

The current implementation covers three target scenarios from the UAV-NIDD setting:

- `uav`: WiFi frame-layer UAV node traffic
- `ap`: mixed-layer access point traffic
- `gcs`: flow-statistics traffic at the ground control station

The engineering pipeline also includes:

- target-domain training and evaluation
- ablation and baseline comparison
- few-shot analysis
- edge inference simulation
- FastAPI gateway and Vue-based monitoring dashboard

## Repository Layout

```text
F:\Project\HTL-UAV-IDS-V3
├── data/                     # raw datasets used by training/evaluation
├── logs/                     # runtime alerts and logs
├── results/                  # figures and csv outputs
├── scripts/                  # training / evaluation / plotting entry points
├── src/
│   ├── data_engine/          # preprocessing and dataset pipeline
│   ├── inference/            # real-time inference and explanation
│   ├── models/               # HTL and baseline models
│   ├── training/             # source pretraining and target finetuning
│   └── web_gcs/              # FastAPI gateway + Vue dashboard
├── weights/                  # saved model weights and preprocessors
├── ONLINE_DEMO_DEPLOY.md     # static demo deployment note
├── README_submission.md      # contest-oriented packaging and replay guide
└── requirements.txt
```

## Environment

- Python `3.9+`
- PyTorch `2.x`
- Node.js `18+`

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Core Commands

Source pretraining:

```powershell
python scripts/pretrain.py
```

Target-domain finetuning:

```powershell
python scripts/finetune.py --scenario uav --epochs 20
python scripts/finetune.py --scenario ap --epochs 20
python scripts/finetune.py --scenario gcs --epochs 20 --device cuda
```

Unified evaluation:

```powershell
python scripts/evaluate_v3.py --device cuda
```

Ablation, baselines, and few-shot analysis:

```powershell
python scripts/train_ablations_v3.py --scenario gcs --epochs 20
python scripts/train_dl_baselines.py --scenario gcs --epochs 20
python scripts/few_shot_gcs.py --device cuda
```

Plotting and edge benchmark:

```powershell
python scripts/plot_results_v3.py
python scripts/benchmark_edge_v3.py --device cpu
```

## Demo Workflow

Start the gateway:

```powershell
python src/web_gcs/app.py
```

Start the frontend:

```powershell
cd src/web_gcs
npm run dev
```

Start simulated edge inference:

```powershell
python src/inference/real_time.py --scenario gcs --device cpu
```

Static deployment notes are in [ONLINE_DEMO_DEPLOY.md](F:/Project/HTL-UAV-IDS-V3/ONLINE_DEMO_DEPLOY.md).

## Notes For Contest Packaging

This repository is a working project directory, not the final submission package. For contest submission:

- exclude local virtual environments and downloaded frontend dependencies
- exclude generated caches and local IDE metadata
- keep only source, required weights, required results, and replay instructions

Use [README_submission.md](F:/Project/HTL-UAV-IDS-V3/README_submission.md) as the packaging reference.


