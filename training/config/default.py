"""
Default config for SegMASt3R training.
Keeps our keys separate from the original project's default.py.
"""
from yacs.config import CfgNode as CN

_CN = CN()

# ── Model ─────────────────────────────────────────────────────────
_CN.MODEL = CN()
_CN.MODEL.MAST3R_CKPT = ""
_CN.MODEL.ARCH        = "sinkhorn"   # sinkhorn | lightglue

# LightGlue-style architecture params (MODEL.ARCH = "lightglue")
_CN.MODEL.LG = CN()
_CN.MODEL.LG.PROJ_DIM            = 128   # project MASt3R 24-dim → this
_CN.MODEL.LG.N_LAYERS            = 3     # self+cross attention blocks
_CN.MODEL.LG.N_HEADS             = 4     # attention heads
_CN.MODEL.LG.GRAD_CHECKPOINT     = False # gradient checkpointing (saves VRAM)
_CN.MODEL.LG.DEEP_SUPERVISION    = False # compute loss at every layer
_CN.MODEL.LG.LAMBDA_MATCH        = 1.0   # weight for matchability BCE loss

# ── Misc ──────────────────────────────────────────────────────────
_CN.DEBUG    = False
_CN.SAVE_DIR = "results/segmast3r_repro"
_CN.RESUME   = ""          # path to checkpoint, or "" for fresh start

# ── Feature matcher ───────────────────────────────────────────────
_CN.FEATURE_MATCHER = CN()
_CN.FEATURE_MATCHER.TYPE = "Sinkhorn"
_CN.FEATURE_MATCHER.SINKHORN = CN()
_CN.FEATURE_MATCHER.SINKHORN.NUM_IT             = 50
_CN.FEATURE_MATCHER.SINKHORN.DUSTBIN_SCORE_INIT = 1.0

# ── Dataset ───────────────────────────────────────────────────────
_CN.DATASET = CN()
_CN.DATASET.METADATA_PATH = ""
_CN.DATASET.DATA_ROOT     = ""
_CN.DATASET.SEGDATA_ROOT  = ""
_CN.DATASET.PAIRS_ROOT    = ""
_CN.DATASET.HEIGHT        = 336
_CN.DATASET.WIDTH         = 512
_CN.DATASET.RESIZE_MODE   = "square"    # square | longest_side
_CN.DATASET.VAL_FRACTION          = 0.02
_CN.DATASET.PRECOMPUTED_FEAT_ROOT = ""  # path to precomputed pooled descriptors; "" = disabled

# ── Training ──────────────────────────────────────────────────────
_CN.TRAINING = CN()
_CN.TRAINING.BATCH_SIZE      = 36
_CN.TRAINING.NUM_WORKERS     = 8
_CN.TRAINING.PREFETCH_FACTOR = 2
_CN.TRAINING.LR              = 1e-4
_CN.TRAINING.WEIGHT_DECAY    = 1e-4
_CN.TRAINING.EPOCHS          = 5
_CN.TRAINING.GRAD_CLIP       = 0.0
_CN.TRAINING.LR_SCHEDULER    = "cosine"  # cosine | none
_CN.TRAINING.WARMUP_STEPS    = 500
_CN.TRAINING.LOG_INTERVAL    = 100
_CN.TRAINING.VAL_INTERVAL    = 5000
_CN.TRAINING.SAVE_INTERVAL   = 5000

# ── Accelerate ────────────────────────────────────────────────────
_CN.ACCELERATE = CN()
_CN.ACCELERATE.MIXED_PRECISION = "no"   # no | bf16 | fp16

cfg = _CN
