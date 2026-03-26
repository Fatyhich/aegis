## Project Context
- Name: AEGIS — Adaptive Environment Graph Identification System
- Goal: Dynamic anchor object identification for adaptive hierarchical knowledge graph construction across heterogeneous environments (Master's thesis, Skoltech)
- Stack: Python 3.11+, PyTorch 2.4+, uv, YACS config, Pydantic schemas
- Datasets: SCAND (indoor/outdoor navigation), EgoWalk (city walking)
- Benchmarks: HM-EQA/Explore-EQA (267 HM3D scenes), task-driven object navigation

## Architecture
- `src/aegis/` — installable package (`uv pip install -e .`)
  - `encoders/` — DINOv2, Florence-2, NaRadIO visual encoders
  - `segmentation/` — SAM 1/2 initialization
  - `graph/` — knowledge graph: nodes, edges, exporters
  - `pipeline/` — orchestrator + chunk processor
  - `tracking/` — anchors, matching, tracking, bbox utils
  - `metrics/` — unsupervised quality metrics
  - `visualization/` — mask/anchor/graph visualization
- `training/` — SegMASt3R fine-tuning pipeline (Accelerate, TensorBoard)
- `evaluation/` — eval scripts, dataset interfaces (Replica, VKITTI2), metrics (AUPRC, R@k)
- `configs/` — pipeline YAML configs
- `scripts/` — CLI entry points
- `notebooks/` — standalone pipeline notebooks
- `visualization/` — Gradio web app for graph viz
- `third_party/` — external repo references (see third_party/README.md)

## Code Style
- Linter: ruff (line-length=100, target py311, select E/F/I)
- Tests: pytest, test files in `tests/`
- Install: `uv pip install -e .` or `uv pip install -e ".[training,eval,viz,dev]"`

## Key Decisions
- Three anchor selection strategies: VLM-based, frequency-based, visual feature clustering
- Three-level adaptive hierarchy: anchor nodes → associated objects → atomic objects
- RGB-only processing (no depth required) via VLMs
- SegMASt3R training: frozen MASt3R backbone + Sinkhorn OT matcher
- Two model variants: Sinkhorn (`training/models/sinkhorn.py`) and LightGlue (`training/models/lightglue.py`)
- VGGT backbone variant: frozen VGGT Aggregator + LightGlue head (`training/models/vggt_lightglue.py`)
- MAX_MASKS = 100 (paper M=100)

## Training Pipeline — Critical Design Constraint
MASt3R decoder uses cross-attention between view0 and view1. Descriptors for img0
DEPEND on which img1 it's paired with. Consequences:
- `extract_desc(img0, img1)` must receive BOTH images → `backbone(view0, view1)`
- Per-image precompute is INVALID — only per-pair precompute works
- Precompute output: `<pair_dsc_root>/<scene>/<name_i>__<name_j>.pt` → `{"dsc0": (M,24), "dsc1": (N,24)}`
- Config key: `DATASET.PAIR_DSC_ROOT` (old `PRECOMPUTED_FEAT_ROOT` is deprecated/removed)
- Precomputed mode only supported for LightGlue arch (sinkhorn uses online backbone)

## VGGT Backbone
- Submodule at `third_party/vggt/` — call `setup_vggt_path()` before any `vggt.*` imports
- Image normalization: [0, 1] range (Aggregator normalizes internally with ResNet mean/std)
  — MASt3R uses [-1, 1]. Do NOT mix normalization conventions.
- Descriptor dim: 2048 (concatenated frame+global tokens) vs MASt3R 24-dim
- `MODEL.ARCH = "vggt"` requires `DATASET.PAIR_DSC_ROOT` (online too expensive)
- Precompute script: `training/scripts/precompute_vggt_features.py`
- Disk warning: VGGT precomputed descriptors are ~85× larger than MASt3R — store as fp16
  (~6GB per 1K pairs vs ~70MB for MASt3R)
- Config: `training/config/segvggt_train.yaml`
- Weights: `third_party/vggt_weights.pt` (local) or `facebook/VGGT-1B` (HuggingFace)

## Experiment Tracking
- All runs logged to TensorBoard
- Run naming convention: YYYY-MM-DD_<short-description>
- Never overwrite a run, create new one
- Checkpoints: /mnt/vol1/checkpoints/fatykhich/<run_name>/
- Local results: results/segmast3r_repro/

## Known Issues (evaluation/)
- `evaluation/core/model_infer.py:65` — `if dataset_type == "mapfree" or "hm3d"` always True
- `evaluation/datasets/replica_dataset.py:19` — bare import `from ground_truth_generator` fails
- `evaluation/scripts/eval_vizenc_vkitti2.py:454` — `Image.fromarray` on int32 crashes

## After Every Change
1. Review diff for style compliance (ruff)
2. Update this CLAUDE.md if behavior/structure changed
3. If model architecture changed — note it in ARCHITECTURE.md
4. If metric changed — log to experiments/log.md with reason
