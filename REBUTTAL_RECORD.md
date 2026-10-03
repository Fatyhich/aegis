# MuViSeg rebuttal record

ACCV 2026, submission #1391. Everything measured between 30 August and 2 September 2026,
stated without interpretation. Numbers are as produced by the scripts named in each section.

- Closed-loop: 102 episodes / 36 scenes
- Pairwise: ScanNet++ validation (in-domain) + 28 022 (Replica) and 10 308 (VKITTI2) transfer queries
- 9 navigation rows, 36 paired tests

---

## 1. Protocol

| | |
|---|---|
| dataset | HM3D v0.2, `hm3d_iin_val`, split `val`, task `original` |
| episodes | 108 across 36 scenes; 6 removed by the upstream `configs/defaults.yaml` exclusion list → **n = 102** |
| start | `max_start_distance: hard` (~5 m geodesic from goal) |
| success | geodesic distance to goal ≤ `1.0 m`; `max_steps: 250` |
| controller | `method: robohop`, `goal_source: topological`, `infer_depth: false` |
| segmenter | FastSAM online and at map-build time; graphs `nodes_fast_sam_lightglue.pickle` (prebuilt, downloaded) |
| localiser | `loc_radius: 8` (window 16), `subsample_ref: 1`, `min_num_matches: 0`, `reloc_rad_add: 2`, `reloc_rad_max: 15`, `edge_weight_str: margin` |
| SPL | `path_length / max(shortest, path_length)`; failures contribute 0; averaged over all n |
| oracle row | `goal_source: gt_metric`, `use_gt_localization: true` — ground-truth metric localisation, same controller |

Every row uses the identical episode list and identical localiser settings. Only
`goal_gen.matcher_name` differs, except the oracle row, which differs in `goal_source`.

---

## 2. Table 3 — closed-loop results

| Row | n | SR % | SPL | Soft-SPL |
|---|---:|---:|---:|---:|
| Oracle (GT metric localisation) | 102 | 78.43 | 78.33 | 84.29 |
| SegMASt3R + Sinkhorn | 102 | 55.88 | 55.81 | 67.21 |
| SegVGGT-DPT Joint N=2 | 101 | 45.54 | 45.43 | 53.60 |
| LightGlue (RoboHop baseline) | 102 | 44.12 | 44.10 | 53.16 |
| SegVGGT-DPT Joint N=4 | 102 | 43.14 | 43.03 | 53.45 |
| SegVGGT-DPT pairwise | 102 | 42.16 | 42.05 | 50.17 |
| SegMASt3R + LGv2 | 102 | 41.18 | 41.17 | 51.89 |
| SegVGGT-DPT Joint N=4, matchability gate on | 90 | 35.56 | 35.56 | 49.90 |
| RoMa → segment vote | 102 | 27.45 | 27.45 | 37.40 |

Submitted Table 3, for reference (n = 10 over 4 scenes):
LightGlue 40.00 / 39.87 · Sinkhorn 50.00 / 45.70 · SegVGGT-DPT 50.00 / 50.99 ·
LGv2 60.00 / 59.14 · Joint N=2 70.00 / 49.04.

---

## 3. Paired tests, all 36 comparisons

Exact McNemar on success (b/c = episodes only A solved / only B solved), paired bootstrap on
per-episode SPL differences, 10 000 resamples. Uncorrected; 36 comparisons were made.

| A vs B | n | b/c | p | ΔSPL [95% CI] |
|---|---:|---:|---:|---|
| Sinkhorn vs LightGlue | 102 | 22/10 | **0.050** | +11.71 [+0.93, +22.50] |
| Sinkhorn vs LGv2 | 102 | 23/8 | **0.011** | +14.64 [+4.63, +25.29] |
| Sinkhorn vs DPT pairwise | 102 | 20/6 | **0.009** | +13.76 [+3.99, +23.55] |
| Sinkhorn vs Joint N=4 | 102 | 18/5 | **0.011** | +12.78 [+3.96, +21.60] |
| Sinkhorn vs Joint N=2 | 101 | 15/5 | **0.041** | +9.94 [+1.98, +18.81] |
| Sinkhorn vs RoMa | 102 | 34/5 | **<0.001** | +28.36 [+17.63, +39.08] |
| Sinkhorn vs gate-on | 89 | 24/8 | **0.007** | +17.89 [+5.62, +30.09] |
| Oracle vs Sinkhorn | 102 | 36/13 | **0.001** | +22.52 [+9.79, +35.24] |
| Oracle vs LightGlue | 102 | 43/8 | **<0.001** | +34.22 [+22.34, +45.99] |
| Oracle vs Joint N=2 | 101 | 43/10 | **<0.001** | +32.68 [+19.86, +45.36] |
| Oracle vs Joint N=4 | 102 | 45/9 | **<0.001** | +35.30 [+22.61, +47.13] |
| Oracle vs DPT pairwise | 102 | 45/8 | **<0.001** | +36.28 [+23.86, +48.00] |
| Oracle vs LGv2 | 102 | 46/8 | **<0.001** | +37.16 [+25.35, +48.94] |
| Oracle vs gate-on | 89 | 41/5 | **<0.001** | +40.39 [+28.04, +52.74] |
| Oracle vs RoMa | 102 | 57/5 | **<0.001** | +50.87 [+39.17, +61.69] |
| LightGlue vs RoMa | 102 | 27/10 | **0.008** | +16.65 [+5.86, +28.39] |
| Joint N=4 vs RoMa | 102 | 23/7 | **0.005** | +15.58 [+5.77, +25.48] |
| DPT pairwise vs RoMa | 102 | 22/7 | **0.008** | +14.60 [+4.86, +24.49] |
| LGv2 vs RoMa | 102 | 18/4 | **0.004** | +13.72 [+4.90, +22.54] |
| Joint N=2 vs RoMa | 101 | 23/5 | **0.001** | +17.71 [+7.91, +27.52] |
| LGv2 vs LightGlue | 102 | 16/19 | 0.736 | −2.93 [−14.70, +8.82] |
| LGv2 vs Joint N=4 | 102 | 12/14 | 0.845 | −1.86 [−11.66, +7.94] |
| LGv2 vs DPT pairwise | 102 | 13/14 | 1.000 | −0.88 [−10.78, +8.91] |
| LGv2 vs Joint N=2 | 101 | 11/16 | 0.442 | −4.85 [−14.85, +5.05] |
| LGv2 vs gate-on | 89 | 19/14 | 0.487 | +5.61 [−6.75, +17.97] |
| LightGlue vs Joint N=4 | 102 | 18/17 | 1.000 | +1.07 [−9.82, +12.74] |
| LightGlue vs DPT pairwise | 102 | 18/16 | 0.864 | +2.05 [−9.53, +12.92] |
| LightGlue vs Joint N=2 | 101 | 15/17 | 0.860 | −1.89 [−12.78, +9.10] |
| LightGlue vs gate-on | 89 | 16/8 | 0.152 | +8.98 [−1.14, +20.20] |
| DPT pairwise vs Joint N=4 | 102 | 9/10 | 1.000 | −0.98 [−9.76, +6.90] |
| DPT pairwise vs Joint N=2 | 101 | 9/13 | 0.523 | −3.96 [−12.88, +4.95] |
| DPT pairwise vs gate-on | 89 | 18/13 | 0.473 | +5.50 [−6.75, +17.75] |
| Joint N=2 vs Joint N=4 | 101 | 9/6 | 0.607 | +2.97 [−4.95, +10.89] |
| Joint N=2 vs gate-on | 89 | 19/9 | 0.087 | +11.11 [−0.12, +22.45] |
| Joint N=4 vs gate-on | 89 | 19/12 | 0.281 | +7.74 [−4.51, +19.86] |
| RoMa vs gate-on | 89 | 12/18 | 0.362 | −6.74 [−19.10, +5.62] |

---

## 4. Oracle decomposition, n = 102

| | count |
|---|---:|
| oracle solves | 80 |
| baseline solves | 45 |
| both solve | 37 |
| oracle only | 43 |
| baseline only | 8 |
| neither | 14 |
| oracle fails | 22 of 102 (21.6 %) |

---

## 5. Pairwise segment matching

ScanNet++ is the **training** domain (`EXPERIMENT_HISTORY.md:14`). Replica and VKITTI2 are
transfer sets. §5.1 is in-domain validation, §5.2 is transfer.

### 5.1 ScanNet++ validation (in-domain)

Recorded during training, read from `val/*` scalars in the TensorBoard logs under each run's `tb/`
directory; the same figures are embedded in the `metrics` dict of each `best.pth`.

| | SegMASt3R+LGv2 | SegVGGT-DPT pairwise (v3-001) | SegVGGT-DPT Joint (joint-001) |
|---|---:|---:|---:|
| AUPRC | 0.9103 | 0.8859 | 0.7794 |
| matching accuracy / R@1 | 0.8445 | 0.8220 | 0.7551 |
| R@5 | — (R@3 = 0.9748) | 0.9758 | 0.9303 |
| mean GT logprob | −1.1666 | −1.4065 | −2.4882 |
| dustbin rate | 0.7413 | 0.7302 | 0.5029 |
| false dustbin rate | — | 0.0384 | 0.0535 |
| wrong match rate | — | 0.0328 | 0.0486 |
| false match rate | — | 0.0313 | 0.2719 |
| loss | 1.4470 | 1.3160 | 2.5472 |
| best step | 55 000–60 000 | 35 000 | 8 000 |
| epoch at checkpoint | 0 | 3 | 2 |

Comparability notes:

- LGv2 logged **R@3**, not R@5, and has no error decomposition (`false_dustbin_rate`,
  `wrong_match_rate`, `false_match_rate`). It was trained with an earlier
  `training/engine/validator.py` that did not compute them.
- Joint was trained for 8 000 steps against 35 000 (pairwise) and 60 000 (LGv2), and its checkpoint
  is from epoch 2. Its `false_match_rate` is 0.2719 against 0.0313 for pairwise, and its
  `dustbin_rate` is 0.5029 against 0.7302. These data do not separate architecture from
  under-training.
- No evaluation of ScanNet++ as a *test* set exists. The only config for it,
  `evaluation/configs/config_eval_spp_resz.yaml`, points at `/scratch/public_scratch/...` on another
  machine and is marked dead in the release notes. The data is present on this machine at
  `/mnt/vol1/datasets/ScanNet++` including prebuilt `scannetpp_pairs`.

### 5.2 Tables 1-2 — transfer to Replica and VKITTI2

Query-count-weighted across all viewpoint bins. Replica 28 022 queries, VKITTI2 10 308 queries,
four bins each. Replica from `pairs_replica_3200.json` (3200 pairs), VKITTI2 from
`pairs_vkitti2_4000.json` (2400 pairs).

| Matcher | Replica AUPRC | R@1 | R@5 | VKITTI2 AUPRC | R@1 | R@5 |
|---|---:|---:|---:|---:|---:|---:|
| RoMa → segment vote (certainty) | 84.73 | 81.55 | 90.53 | 62.41 | 59.28 | 75.11 |
| RoMa → segment vote (raw votes) | 84.67 | 81.59 | 90.46 | 62.38 | 59.35 | 75.07 |
| SegMASt3R + LGv2 | — | — | — | 81.22 | 72.74 | 92.75 |
| SegVGGT-DPT Joint N=2 | 81.45 | 76.23 | 90.69 | 52.16 | 42.33 | 66.73 |
| SegVGGT-DPT Joint N=4 | 81.57 | 76.47 | 90.58 | 52.85 | 43.22 | 66.75 |
| SegVGGT-DPT Joint N=6 | 81.38 | 76.30 | 90.39 | 53.08 | 43.42 | 66.74 |
| SegVGGT-DPT Joint N=8 | 81.10 | 76.07 | 90.28 | 52.89 | 43.22 | 66.78 |
| SegVGGT-DPT pairwise (v3-001) | 79.70 | 74.10 | 90.11 | 51.92 | 42.93 | 67.61 |
| SegVGGT single-layer | 79.15 | 73.29 | 89.61 | 51.52 | 42.11 | 66.31 |

These were previously read by taking the last bin as an overall figure, which overstated every row.
The values above are weighted by query count. `results/segvggt_dpt_vkitti2/joint-001` (AUPRC 61.96)
is a 40-query single-bin smoke run, not an evaluation.

---

## 6. Diagnostic measurements

### 6.1 Localisation error and match volume

Distance between the map image the localiser selected and the agent's true pose, per step,
pooled over 107 episodes.

| Row | matches / step | median error | p90 |
|---|---:|---:|---:|
| LightGlue | 106.4 | 1.77 m | 5.49 |
| SegVGGT-DPT pairwise | 155.6 | 2.19 m | 6.92 |
| SegVGGT-DPT Joint N=4 | 142.5 | 2.39 m | 6.60 |
| SegMASt3R + LGv2 | 151.9 | 2.39 m | 7.34 |
| RoMa | 196.5 | 4.40 m | 9.67 |

Zero-match steps: 0.0 % for every row above. With the matchability gate enabled, Joint N=4 gives
22.2 matches/step, **43.5 % zero-match steps** (11 443 of 26 287) and 5 856 relocalisation events;
without the gate the same model has 0 of either.

### 6.2 Localiser start window

Computed offline from the navmesh with `scripts/start_window_analysis.py`, reusing
`task_setup.closest_state`. The localiser cold-starts at map image 0, so its first search window
is `[0, 2·loc_radius)`.

| Split | window | start inside | median start waypoint | median teach length |
|---|---|---|---:|---:|
| val (108 episodes) | [0, 16) | 45 / 108 (42 %) | 23 | 60 |
| val (108 episodes) | [0, 32) | 65 / 108 (60 %) | 23 | 60 |
| minival (12 episodes) | [0, 16) | 6 / 12 | — | — |
| minival (12 episodes) | [0, 32) | 10 / 12 | — | — |

### 6.3 Run-to-run variance

`viz_*` (29 Apr) and `viz2_*` (24 May) are the same matcher, episode and config; frozen `args.yaml`
differ only in paths and timestamps.

- 16 repeated pairs: **11 produced different trajectories**; 1 flipped outcome
  (`lightglue_ep8`: exceeded_steps at 249 steps / 6.67 m → success at 190 steps / 0.99 m).
- Sinkhorn reproduced bitwise in 4 of 4; LightGlue, LGv2 and Joint did not.
- Three repeats each of LGv2 and LightGlue on minival (n = 10, fixed code):
  LGv2 SR 50 / 40 / 50, mean SPL 46.63 (sd 5.77); LightGlue SR 50 / 50 / 40, mean SPL 46.46 (sd 5.67).
  2 of 10 episodes were unstable across repeats.

### 6.4 Statistical power

Simulated McNemar power for the discordance observed at n = 10 (30 % / 10 %), α = 0.05.

| n episodes | 10 | 12 | 24 | 36 | 54 | 72 | 90 | 108 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| power | 0.03 | 0.05 | 0.23 | 0.39 | 0.59 | 0.72 | 0.83 | 0.91 |

---

## 7. Defects found and fixed

| Location | Defect | Change |
|---|---|---|
| `libs/localizer/loc_topo.py:117` | `refImgInds` passed positionally into the `qryPosition` slot; raised `ValueError` in `get_closest_map_image_index` and silently disabled `refImgIndsHistory` de-duplication | pass by keyword |
| `libs/localizer/loc_topo.py:82` | localiser cold-starts at image 0 while `hard` start places the agent mid-trajectory | `init_localization`: `global` (default, match frame 0 against the whole map), `pose`, `zero` (legacy) |
| `libs/control/robohop.py:26`, `task_setup.py:701` | `pls < 99` with `pls = None` when the segmenter returns no masks | None guard mirroring the `learnt` branch |
| `main.py:159,163` | `close(step)` raised `UnboundLocalError` if the exception preceded the loop; error strings reported the outermost frame, so every failure pointed at `main.py` | `step` initialised before `try`; innermost traceback frame reported; full traceback logged |
| `scripts/evaluate_objecreact.py:19-20` | blacklist read from `configs/defaults.yaml`, not the run's own config; the two disagreed and moved reported SR by 10 points | read from the run's `args.yaml`; all run configs declare the upstream list explicitly |
| `scripts/evaluate_objecreact.py:10,100` | episode universe hardcoded to `hm3d_iin_val`; `len(status) == ""` compares int to str | derived from the run's split; comparison fixed |
| `evaluation/pairs_vkitti2_3200.json` | byte-identical to `pairs_replica_3200.json` (md5 `ee66193a…`); `config_eval_vkitti2.yaml` pointed at it | quarantined; config repointed at `pairs_vkitti2_4000.json` |
| `evaluation/core/model_infer.py:97` | `if dataset_type == "mapfree" or "hm3d"` is always true, so the branch applied to every dataset including Replica and VKITTI2 | made unconditional with a comment; **not** "fixed", because all published Table 1-2 numbers were produced with the branch taken |
| 9 of 11 `evaluation/configs/*.yaml` | `DATA_ROOT` pointed at paths from other machines | repointed at the mounted datasets |
| `libs/experiments/task_setup.py:1161` | matcher rebuilt from disk every episode: 59 s × 108 ≈ 1.75 h per run, and the cause of contention between concurrent runs | built once per run in `preload_models`; verified bitwise-identical matches across episodes |
| `libs/matcher/segvggt_dpt_joint.py` | hardcoded 2-frame forward; the model is N-agnostic and was trained at `N_FRAMES: 4` | `joint_n_frames`; verified bitwise-identical to the previous implementation at N = 2 |

---

## 8. Data and environment

- Scene assets downloaded 1 Sep: `hm3d-val-habitat-v0.2.tar` (3.3 GB),
  `hm3d-val-semantic-annots-v0.2.tar` (2.0 GB), `hm3d-val-semantic-configs-v0.2.tar` (40 KB)
  → 100 scene dirs, 36 with semantics. All 108 episodes resolve to a `*basis.glb`.
- Map graphs for all 108 episodes were already present (downloaded from HuggingFace, mtimes
  2025-09-11 09:06:38–39), 229 269 nodes and 973 823 edges in total.
- Pairwise datasets: `semantic-replica` (10 scenes, 7.7 GB), `virtual-KITTI-2` (5 scenes, 17 GB).
- Environment `miniforge3/envs/nav`, Python 3.10, habitat-sim 0.3.3 headless_bullet,
  habitat-lab 0.2.4, torch 2.11.0+cu128. Added: `romatch` 0.1.2, `fused-local-corr` 0.3.211,
  `scikit-learn` 1.7.2, `timm`, `loguru`, `tensorboard` 2.21.0. torch unchanged at 2.11.0+cu128 throughout.
- `nabo-aegis/.venv/bin/python` is a broken symlink into another user's home and does not run.
- Hardware: one RTX PRO 6000 Blackwell (97 GB) and one RTX 5090 (32 GB, drives the display);
  64 cores, 251 GB RAM.
- Measured cost per episode, single run: LightGlue 1.0 min, oracle 4.2, RoMa 5.8, LGv2 8.3,
  Sinkhorn 9.6, DPT pairwise 9.2, Joint N=2 11.5, Joint N=4 ≈ 7.6.
- Code, configs and analysis scripts committed on branch `rebuttal/table3`; per-episode metrics
  written to `per_episode_metrics.csv` beside every run.

---

## 9. Not done

- **DA3 (Sequential).** No implementation exists in the repository — no class, config, results or
  deletion in git history. The Tables 1-2 values 54.29 (Replica) and 40.67 (VKITTI2) appear only as
  literals in `evaluation/scripts/make_thesis_figures.py:61,70`.
- **Segmenter sensitivity (SAM vs FastSAM).** `libs/segmentor/` contains only `fast_sam_module.py`;
  `sam.py`, `sam2_seg.py` and `sam21.py` are absent, and no SAM checkpoint exists on the filesystem.
  213 prebuilt `nodes_sam21_lightglue_depth_inferred.pickle` graphs are present in the data.
- **Repeats at n = 102.** All val rows are single runs. Run-to-run variance is carried over from the
  minival repeats in §6.3.
- **Sinkhorn and Joint N=2 parity.** The matcher adapters were never under version control. The
  adapters on this machine are known to differ from those that produced the submitted table;
  LGv2 and LightGlue match, Joint and Sinkhorn do not.
- **vizEnc / DINOv2.** `evaluation/core/vizenc_inference.py:36` hardcodes a SAM checkpoint path that
  does not exist and builds SAM unconditionally.

---

## 10. Data provenance notes

- Episode `q5QZSEeHe5g_0000000_plant_32_` has no resolvable goal node in its map graph. It fails
  identically for every topological row and is counted as a failure; the oracle row does not resolve
  a goal node from the graph and records `exceeded_steps` instead.
- The submitted VKITTI2 pair count is 3200; `pairs_vkitti2_4000.json` contains 2400 pairs, giving
  10 308 query segments, which is the figure in `EXPERIMENT_HISTORY.md`.
- `make_thesis_figures.py` hardcodes Replica AUPRC 79.70 for SegVGGT-DPT v3; the value on disk is
  79.70. `EXPERIMENT_HISTORY.md` records 79.10.
- VKITTI2 figures label angle bins 0–45/45–90/90–135/135–180°; the data uses 0–20/20–40/40–60/60–90°.
- `Dockerfile:9-10` in `nabo-aegis` contains live proxy credentials and `Dockerfile:6` a personal
  username, both in git history.
- Sinkhorn was run twice in April through the same config 18 hours apart, giving 25.00 % and 41.67 %
  SR. `segmast3r_sinkhorn.py` has an mtime between the two runs and a comment recording that its
  score threshold default was changed from 0.05 to 0.0.

---

Sources: `third_party/object-rel-nav/scripts/analyze_table3.py` (paired tests),
`scripts/evaluate_objecreact.py` (per-episode metrics), `scripts/start_window_analysis.py`
(start windows), `evaluation/scripts/eval_replica_table2.py` and `eval_segvggt_vkitti2.py`
(Tables 1-2); ScanNet++ validation read from `val/*` TensorBoard scalars under each run's `tb/`.

Gate-on row was at 96 of 108 episodes when this record was written; its 61- and 90-episode
evaluations gave SPL 31.15 and 35.56.
