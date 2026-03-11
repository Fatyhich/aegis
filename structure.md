# Project Structure

See `README.md` for overview. This file documents the source layout.

```
aegis/
├── src/
│   └── aegis/                  # Installable package (uv pip install -e .)
│       ├── encoders/           # Visual encoders: DINOv2, Florence-2, NaRadIO
│       ├── segmentation/       # SAM 1/2 initialization
│       ├── config/             # YAML config loader + Pydantic schemas
│       ├── graph/              # Knowledge graph: nodes, edges, exporters
│       ├── pipeline/           # Pipeline orchestrator + chunk processor
│       ├── preprocessing/      # Mask cropping utilities (MaskCropper)
│       ├── storage/            # ChunkGraphStorage, file naming
│       ├── tracking/           # Anchors, matching, tracking, bbox utils
│       ├── metrics/            # Unsupervised quality metrics
│       └── visualization/      # Mask/anchor/graph visualization
│
├── evaluation/
│   ├── core/                   # Reusable eval modules
│   │   ├── eval_metrics.py     # AUPRC, R@k metrics
│   │   ├── model_infer.py      # MASt3R inference wrapper
│   │   ├── segmentor.py        # FastSAM segmentation wrapper
│   │   ├── ground_truth_generator.py
│   │   └── vizenc_inference.py # VizEnc segment matcher
│   ├── scripts/                # Runnable evaluation scripts
│   ├── datasets/               # Dataset interfaces (Replica, VKITTI2)
│   ├── sampling/               # Image pair sampling
│   ├── configs/                # Eval YAML configs
│   └── data/                   # pairs_*.json files
│
├── scripts/                    # CLI entry points
├── notebooks/                  # Standalone pipeline notebooks/scripts
├── visualization/              # Gradio web app for graph visualization
├── configs/                    # Pipeline YAML configs
├── third_party/                # README: what external repos to clone
├── tests/
└── data/                       # Assets (logo, diagrams)
```

## Installation

```bash
uv pip install -e .
```

## Third-Party Setup

See `third_party/README.md` for required external repositories.
