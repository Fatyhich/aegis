# Third-Party Dependencies

These repositories must be cloned manually before running the pipeline.
They are excluded from git (see `.gitignore`).

## Required

### segment-anything (SAM 1)
Used by: `aegis.segmentation.sam` (`init_sam1`)

```bash
git clone https://github.com/facebookresearch/segment-anything.git
```

Download checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

### RayFronts (NaRadIO encoder)
Used by: `aegis.encoders.naradio` (`load_naradio_encoder`)

```bash
git clone https://github.com/RayFronts/RayFronts.git
```

## Optional

### sam2 (SAM 2)
Used by: `aegis.segmentation.sam` (`init_sam2`)

```bash
git clone https://github.com/facebookresearch/sam2.git
```

### segmast3r (Evaluation baseline)
Used by: `evaluation/core/model_infer.py`

```bash
git clone <internal-repo-url> segmast3r
```

### vl-kgp (VL-KnG baseline)
```bash
git clone https://github.com/VL-KnG/VL-KnG.git vl-kgp
cd vl-kgp && uv pip install -e .
```

## Expected Layout After Cloning

```
aegis/
├── segment-anything/   # SAM 1
├── sam2/               # SAM 2 (optional)
├── RayFronts/          # NaRadIO encoder
├── segmast3r/          # Evaluation model (optional)
└── vl-kgp/             # VL-KnG baseline (optional)
```

## Checkpoints

Place model checkpoints in `checkpoints/`:

```
checkpoints/
├── sam_vit_h_4b8939.pth          # SAM ViT-H
├── sam_vit_l_0b3195.pth          # SAM ViT-L (optional)
├── sam_vit_b_01ec64.pth          # SAM ViT-B (optional)
└── ultralytics/
    └── FastSAM-x.pt              # FastSAM (for evaluation)
```
