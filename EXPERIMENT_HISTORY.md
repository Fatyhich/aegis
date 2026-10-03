# AEGIS — Experiment History & Results

> Adaptive Environment Graph Identification System
> Master's Thesis, Skoltech — Denis Fatykhov, Advisor: Prof. Gonzalo Ferrer

---

## 1. Project Overview

**Задача:** Segment Matching — нахождение соответствий между сегментами (объектами) на парах/группах изображений одной сцены.

**Подход:** Замороженный backbone (MASt3R или VGGT) извлекает визуальные дескрипторы, маски сегментации пулятся в per-segment дескрипторы, обучаемая LightGlue-style голова строит соответствия.

**Датасет для обучения:** ScanNet++ (indoor) — пары кадров с масками instance segmentation, ground truth соответствия по instance ID совпадению.

**Бенчмарки для оценки:**
- **Replica** (indoor, 8 сцен) — 3200 пар, бины по углу поворота: [0-45°], [45-90°], [90-135°], [135-180°]
- **Virtual KITTI 2** (outdoor, driving) — 3200 пар, бины: [0-20°], [20-40°], [40-60°], [60-90°]

**Метрики:** AUPRC (Area Under Precision-Recall Curve), R@1, R@5

---

## 2. Хронология разработки

### Фаза 1: Инфраструктура и AEGIS pipeline (январь–февраль 2026)

**Январь 31 — первый коммит.** Создан пакет `src/aegis/` с полным пайплайном:
- Визуальные энкодеры: DINOv2, Florence-2, NaRadIO
- Сегментация: SAM 1/2
- Граф знаний: узлы, рёбра, экспортёры
- Оркестратор + chunk processor
- Якорный трекинг и визуализация

**Февраль 13 — слияние трёх веток в main:**
1. `feature/infra` — Docker, конфиги, скрипты запуска, тесты
2. `feature/vizenc-baseline` — модульный VizEnc pipeline
3. `feature/eval-framework` — метрики, загрузчики Replica/VKITTI2, стратифицированная выборка пар, eval-скрипты для VizEnc и SegMASt3R

### Фаза 2: SegMASt3R training pipeline (март 2026)

**Март 11 — рефакторинг:** миграция на `src/aegis/` package layout, SegMASt3R как git submodule, создание `training/` pipeline с Accelerate, YACS конфигами, Sinkhorn OT matcher.

**Март 14 — критический фикс:** MASt3R decoder использует cross-attention между view0/view1 → дескрипторы для img0 ЗАВИСЯТ от img1 → precompute только per-pair, не per-image. Реализован правильный per-pair precompute. Добавлен SegMASt3RLGv2 (MLP проектор + 4× FFN).

### Фаза 3: VGGT backbone (март 2026, feature/segvggt)

**Март 26:**
- SegVGGT (single-layer) и SegVGGT-DPT (multi-layer DPT fusion)
- VGGT trainer support, gradient accumulation, error decomposition
- Обучены v1 → v2 → v3 конфигурации DPT
- Обучены multiframe и joint варианты

---

## 3. Архитектуры моделей

### 3.1. SegMASt3R + Sinkhorn (baseline)

```
MASt3R ViT-Large Decoder (frozen, 24-dim descriptors)
  → Masked Average Pooling → (B, 24, M)
  → Sinkhorn OT matching → score matrix (M, N)
```

- **Обучаемые параметры:** 0 (baseline без обучения)
- **Особенность:** MASt3R cross-attention → precompute только per-pair
- **Конфиг:** `segmast3r_train.yaml`, `segmast3r_train_precompute.yaml`

### 3.2. SegMASt3R + LightGlue v1

```
MASt3R ViT-Large Decoder (frozen, 24-dim)
  → Masked Average Pooling → (B, 24, M)
  → Linear(24 → 128) [без нелинейности]
  → 3× SegmentAttentionLayer (self + cross, 4 heads, 2× FFN)
  → DoubleSoftmax matcher
```

- **Обучаемые параметры:** ~450K
- **Конфиг:** `segmast3r_train_lg.yaml`

### 3.3. SegMASt3R + LightGlue v2

```
MASt3R ViT-Large Decoder (frozen, 24-dim)
  → Masked Average Pooling → (B, 24, M)
  → MLP Projector: Linear(24→64) + GELU + LN + Linear(64→128)
  → 3× SegmentAttentionLayerV2 (pre-LN self + cross, 4 heads, 4× FFN)
  → DoubleSoftmax matcher (learnable temperature, matchability bias)
```

- **Обучаемые параметры:** ~800K
- **Отличия от v1:** нелинейный проектор, 4× FFN, симметричная кросс-нормализация
- **Конфиг:** `segmast3r_train_lg_v2.yaml`

### 3.4. SegVGGT (single-layer)

```
VGGT Aggregator (frozen, layer 23, 2048-dim tokens)
  → Masked Average Pooling → (B, 2048, M)
  → MLP: Linear(2048→256) + GELU + LN + Linear(256→128)
  → 3× SegmentAttentionLayerV2 (pre-LN, 4 heads, 4× FFN)
  → DoubleSoftmax matcher
```

- **Обучаемые параметры:** ~800K
- **Особенность:** один слой Aggregator (только последний), поддержка precomputed mode
- **Нормализация изображений:** [0, 1] (НЕ [-1, 1] как MASt3R)
- **Конфиг:** `segvggt_train.yaml`

### 3.5. SegVGGT-DPT (multi-layer, основная модель)

```
VGGT Aggregator (frozen, layers [5, 11, 17, 23], 2048-dim каждый)
  → 4× Conv1×1(2048 → 256) проекции
  → DPT-style progressive fusion:
      deep → RCU → (+shallow) → RCU → ... → (B, 256, H_p, W_p)
  → 7× Residual Conv Units (3×3 spatial kernels)
  → Masked Average Pooling → (B, 256, M)
  → Linear(256→128) + LayerNorm
  → 3× SegmentAttentionLayerV2 (pre-LN, 4 heads, 4× FFN)
  → DoubleSoftmax matcher (clamped temperature ≤ 1.0)
```

- **Обучаемые параметры:** ~10.7M (преимущественно DPT fusion)
- **Особенность:** пространственное слияние мультимасштабных фичей → лучше границы объектов
- **Online-only:** DPT требует пространственные фичи до пулинга → precompute невозможен
- **Итерации конфигурации:**
  - **v1** (`segvggt_dpt_train.yaml`): базовый, batch_size=20
  - **v2** (`segvggt_dpt_train_v2.yaml`): +gradient accumulation (эфф. batch=40), matchability_bias=0.4
  - **v3** (`segvggt_dpt_train_v3.yaml`): +temperature clamping, lambda_match=0.3, matchability_bias=0.5, batch_size=48 → **лучший pairwise результат**

### 3.6. SegVGGT-DPT Multiframe (N кадров, pairwise attention)

```
[SegVGGT-DPT backbone и fusion — как 3.5]
  → Для каждой пары (i, j) в tuple:
      → Pairwise SegmentAttentionLayerV2 (self + cross)
      → DoubleSoftmax → score matrix (M_i, M_j)
```

- **N_FRAMES:** 4 (default), CONTEXT_STEP=5
- **Подход:** обрабатывает все пары в N-кадровом окне; внимание per-pair
- **Конфиг:** `segvggt_dpt_train_multiframe.yaml`

### 3.7. SegVGGT-DPT Joint (N кадров, joint attention)

```
[SegVGGT-DPT backbone и fusion — как 3.5]
  → Сегменты всех N кадров конкатенируются:
      segments = [seg_0, seg_1, ..., seg_{N-1}]  → (B, Σ M_v, 128)
  → Learnable frame position embeddings (до max_frames=16)
  → 3× JointSegmentAttention (self-attention по всем сегментам всех кадров)
  → Для запрашиваемой пары (i, j):
      → DoubleSoftmax(seg_i, seg_j) → score matrix (M_i, M_j)
```

- **Обучаемые параметры:** ~11.0M
- **Ключевое отличие от Multiframe:** единый self-attention по ВСЕМ сегментам
- **Транзитивный матчинг:** A↔C через общий контекст B
- **Конфиг:** `segvggt_dpt_train_joint.yaml`

---

## 4. Обучение: детали

### Общие параметры

| Параметр | Значение |
|---|---|
| Датасет | ScanNet++ (masks_resize_mast3r) |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | Cosine с warmup 500 шагов |
| Mixed precision | bf16 |
| Разрешение | 336 × 512 |
| MAX_MASKS | 100 (M=100 в статье) |
| Валидация | 2-10% от train |

### Чекпоинты и training runs

| Run | Чекпоинт | Шагов | Эпох | best_val_ma |
|---|---|---|---|---|
| segmast3r_lg_v2 | `results/segmast3r_lg_v2/best.pth` | ~63K | — | — |
| segvggt (single-layer) | `results/segvggt/best.pth` | ~140K | — | — |
| segvggt_dpt/v3-001 | `results/segvggt_dpt/v3-001/best.pth` | ~140K | 3 | — |
| segvggt_dpt/joint-001 | `results/segvggt_dpt/joint-001/best.pth` | ~8K | 2 | 0.7551 |
| segvggt_dpt/multiframe-001 | `results/segvggt_dpt/multiframe-001/` | ~36.5K | — | — |

### Loss функция

```
L = L_match + λ_match · L_matchability

L_match = -Σ gt[i,j] · log_mutual[i,j] / num_gt_matches    (matched pairs)
L_matchability = BCE(match0, has_match0) + BCE(match1, has_match1)  (per-segment)
```

- `lambda_match` = 0.3 (v3), 1.0 (v1/v2)
- **Error decomposition:** false_dustbin_rate, wrong_match_rate, false_match_rate

---

## 5. Результаты: сводные таблицы

### 5.1. Replica (indoor, 3200 пар, 28022 query сегмента)

| Model | Backbone | 0-45° AUPRC | 45-90° | 90-135° | 135-180° | Overall AUPRC | Overall R@1 | Overall R@5 |
|---|---|---|---|---|---|---|---|---|
| **SegMASt3R + LGv2** | MASt3R | **91.17** | 83.59 | 77.09 | 70.28 | **84.48** | **77.46** | **93.63** |
| **SegVGGT-DPT v3** | VGGT | 81.20 | 80.93 | 78.07 | 73.36 | 79.10 | 72.73 | 89.65 |
| **SegVGGT-DPT Joint N=4** | VGGT | 81.66 | **83.03** | **81.85** | **77.69** | 81.37 | 75.64 | 90.42 |
| **SegVGGT-DPT Joint N=2** | VGGT | 81.71 | 83.43 | 81.23 | 76.44 | 81.07 | 75.21 | 90.44 |
| **SegVGGT-DPT Joint N=6** | VGGT | 81.51 | 83.00 | 81.24 | 77.56 | 81.18 | 75.16 | 90.26 |
| **SegVGGT-DPT Joint N=8** | VGGT | 81.36 | 82.59 | 80.68 | 77.42 | 80.82 | 74.96 | 90.05 |
| **SegVGGT single-layer** | VGGT | 80.53 | 80.14 | 77.33 | 74.05 | 79.15 | 73.29 | 89.61 |

**Наблюдения (Replica):**
- MASt3R + LGv2 доминирует на малых углах (0-45°: 91.17 vs 81.7)
- SegVGGT-DPT Joint превосходит MASt3R на больших углах (135-180°: 77.69 vs 70.28)
- Joint-модель показывает стабильное качество при увеличении угла (меньшее падение)
- Увеличение N (кадров) от 2 до 8 почти не влияет на Replica (слабый temporal context)

### 5.2. Virtual KITTI 2 (outdoor, 3200 пар, 10308 query сегментов)

| Model | Backbone | 0-20° AUPRC | 20-40° | 40-60° | 60-90° | Overall AUPRC | Overall R@1 | Overall R@5 |
|---|---|---|---|---|---|---|---|---|
| **SegMASt3R + LGv2** | MASt3R | **77.16** | **89.68** | **89.42** | 78.73 | **81.22** | **72.74** | **92.75** |
| **SegVGGT-DPT v3** | VGGT | 46.25 | 58.40 | 68.86 | 70.63 | 51.92 | 42.93 | 67.61 |
| **SegVGGT-DPT Joint N=2** | VGGT | 47.35 | 57.33 | 66.24 | 71.05 | 52.04 | 41.93 | 66.98 |
| **SegVGGT single-layer** | VGGT | 45.38 | 58.00 | 69.65 | **76.38** | 51.52 | 42.11 | 66.31 |

**Наблюдения (VKITTI2):**
- MASt3R + LGv2 радикально превосходит все VGGT варианты (~81% vs ~52% AUPRC)
- VGGT модели показывают аномально низкие результаты на 0-20° бине (45-47%)
- Интересно: VGGT single-layer лучше DPT на 60-90° (76.38 vs 70.63)
- Joint N=2 на VKITTI2 — единственный полный прогон (N=4/6/8 VKITTI2 не завершились)

### 5.3. SegVGGT-DPT Epoch Sweep (Replica)

| Checkpoint | 0-45° | 45-90° | 90-135° | 135-180° |
|---|---|---|---|---|
| epoch_001 | 81.17 | 81.29 | 78.42 | 73.30 |
| epoch_002 | 80.99 | 80.91 | 78.75 | 73.33 |
| epoch_003 | 81.25 | 81.07 | **78.93** | **74.06** |
| best | 81.20 | 80.93 | 78.07 | 73.36 |

**Наблюдение:** Epoch 003 лучше best на больших углах — best checkpoint выбран по val MA, что не обязательно коррелирует с AUPRC на всех бинах.

### 5.4. SegVGGT-DPT Epoch Sweep (VKITTI2)

| Checkpoint | 0-20° | 20-40° | 40-60° | 60-90° |
|---|---|---|---|---|
| epoch_001 | 46.33 | **58.75** | 67.76 | 71.18 |
| epoch_002 | 46.40 | 57.68 | 66.49 | 63.76 |
| epoch_003 | **46.60** | 58.69 | **68.72** | **73.29** |
| best | 46.25 | 58.40 | 68.86 | 70.63 |

**Наблюдение:** Epoch 003 наиболее стабилен. Epoch 002 деградирует на 60-90° (63.76 vs 73.29).

---

## 6. Модули и их статус

### 6.1. AEGIS Pipeline (`src/aegis/`)

| Модуль | Статус | Описание |
|---|---|---|
| `encoders/` | Готов | DINOv2, Florence-2, NaRadIO энкодеры |
| `segmentation/` | Готов | SAM 1/2 инициализация |
| `graph/` | Готов | Knowledge graph: nodes, edges, exporters |
| `pipeline/` | Готов | Orchestrator + chunk processor |
| `tracking/` | Готов | Anchor tracking, matching, bbox utils |
| `metrics/` | Готов | Unsupervised quality metrics |
| `visualization/` | Готов | Mask/anchor/graph visualization |

### 6.2. Training Pipeline (`training/`)

| Модуль | Статус | Описание |
|---|---|---|
| `models/sinkhorn.py` | Готов | SegMASt3R + Sinkhorn OT baseline |
| `models/lightglue.py` | Готов | SegMASt3R + LightGlue v1 |
| `models/lightglue_v2.py` | Готов | SegMASt3R + LightGlue v2 (MLP + 4×FFN) |
| `models/vggt_lightglue.py` | Готов | SegVGGT single-layer + LightGlue |
| `models/vggt_dpt_lg.py` | Готов | SegVGGT-DPT + SegVGGTDPTJoint |
| `engine/trainer.py` | Готов | Accelerate train loop + multiframe support |
| `engine/validator.py` | Готов | Validation during training |
| `data/dataset.py` | Готов | ScanNetPP loader + multi-frame tuples |
| `config/` | Готов | 10 YAML конфигураций |

### 6.3. Evaluation Pipeline (`evaluation/`)

| Модуль | Статус | Описание |
|---|---|---|
| `core/model_infer.py` | Готов | 5 inference wrappers (все модели) |
| `core/eval_metrics.py` | Готов | AUPRC, R@1, R@5, Table 2 format |
| `core/ground_truth_generator.py` | Готов | Instance ID → GT correspondence matrix |
| `core/vis_utils.py` | Готов | Match visualization + heatmaps |
| `datasets/replica_dataset.py` | Готов | Replica loader (8 сцен, multi-frame) |
| `datasets/vkitti2_dataset.py` | Готов | VKITTI2 loader (5 сцен × 4 варианта) |
| `scripts/eval_replica_table2.py` | Готов | Universal Replica eval |
| `scripts/eval_segvggt_vkitti2.py` | Готов | Universal VKITTI2 eval |
| `scripts/run_eval_segvggt_joint_all.sh` | Готов | N-frame sweep (N=2,4,6,8) |

### 6.4. Third-party (`third_party/`)

| Dependency | Статус | Назначение |
|---|---|---|
| `vggt/` | Submodule | VGGT-1B Aggregator backbone |
| `segmast3r/` | Submodule | MASt3R backbone + helpers |
| `segment-anything/` | Submodule | SAM v1 |
| `sam2/` | Submodule | SAM v2 |
| `RayFronts/` | Submodule | NaRadIO encoder |
| `vggt_weights.pt` | Локальный файл | VGGT-1B веса |

---

## 7. Ключевые проектные решения

1. **Frozen backbone** — ни MASt3R ни VGGT не дообучаются; тренируется только голова матчинга (~0.8–11M параметров из ~1B).

2. **DPT-style fusion** — вместо использования одного слоя Aggregator, извлекаются фичи с 4 уровней глубины и сливаются пространственно через convolutions + residual units. Это даёт лучшие границы сегментов.

3. **Joint attention для multi-frame** — вместо попарного внимания, все сегменты всех кадров обрабатываются в одном self-attention pass. Это позволяет транзитивное соответствие (A↔C через общий контекст B).

4. **DoubleSoftmax matcher** — `log(softmax_row · softmax_col)` + learnable temperature (clamped ≤ 1.0) + matchability bias. Более гибкий чем Sinkhorn, меньше гиперпараметров.

5. **Lambda_match = 0.3** (v3) — сниженный вес matchability loss относительно match loss, чтобы модель больше фокусировалась на правильности соответствий, а не на определении «матчабельности».

6. **Per-pair precompute** (MASt3R) vs **online** (VGGT-DPT) — MASt3R cross-attention вынуждает precompute per-pair; VGGT-DPT требует online из-за spatial fusion до pooling.

---

## 8. Незавершённые эксперименты

| Эксперимент | Статус | Блокер |
|---|---|---|
| Joint sweep: N=4/6/8 на VKITTI2 | Не завершён | Ошибка в eval скрипте (исправлена, нужен rerun) |
| Multiframe-002 | Частично обучен | Прерван |
| SegMASt3R + Sinkhorn eval | Нет отдельных результатов | Базовый baseline без обучения, не прогнан отдельно |

---

## 9. Доступные артефакты для представления

### Количественные результаты
- Полные таблицы AUPRC/R@1/R@5 по бинам для 6+ конфигураций на 2 датасетах
- Epoch sweep (4 чекпоинта) × 2 датасета
- N-frame ablation (N=2,4,6,8) на Replica (полный)
- TensorBoard training curves для всех runs

### Визуализации
- Match visualization (пары с цветовой маркировкой соответствий)
- Score heatmaps (матрицы схожести сегментов)
- Доступны в `evaluation/results/*/visualizations/`

### Чекпоинты
- Все основные модели сохранены с model_state, optimizer_state, scheduler_state
- Возможен inference и дообучение

### Код
- Полностью воспроизводимый pipeline: train → eval → visualization
- Modular design: замена backbone / головы / matcher — через YAML config
