# Project Structure

## vizEnc (src/)

Anchor-based object tracking pipeline для построения Knowledge Graph из последовательности изображений.

```
src/
├── all-in-one.ipynb          # Main notebook - unified pipeline
├── processing.py             # Mask processing + filtering
├── encoders/
│   ├── dinov2.py             # DINOv2 visual encoder
│   ├── naradio.py            # NaRadIO encoder (RADIO + language alignment)
│   └── florence.py           # Florence-2 captioning
├── segmentation/
│   └── sam.py                # SAM 1/2 initialization
├── utils/
│   ├── anchors.py            # Anchor DB: create, update, export
│   ├── matching.py           # Greedy & Hungarian matching
│   ├── metrics.py            # Unsupervised quality metrics
│   ├── tracking.py           # Track ID assignment
│   └── visualization.py      # Visualizations
└── output/                   # Saved anchor_db.pkl, mask_db.pkl
```

### Pipeline Flow
```
Image → SAM (masks) → Visual Encoder (embeddings) → Florence (captions)
                                ↓
                    Filter (category + size)
                                ↓
                    Matching (Hungarian/Greedy)
                                ↓
                    Anchor DB (accumulate observations)
                                ↓
                    Export → Knowledge Graph JSON
```

---

## vl-kgp

Vision-Language Knowledge Graph Pipeline с chunk-based обработкой через Gemini.

```
vl-kgp/
├── main.py                   # Entry point
├── config/
│   └── default.yaml          # Configuration (chunk_size, models, paths)
├── src/
│   ├── core/
│   │   ├── chunk_processing_pipeline.py    # Orchestrator - splits frames into chunks
│   │   ├── efficient_chunk_object_detection.py  # Gemini API - detects objects in chunk
│   │   └── cross_chunk_data_associator.py  # Merges objects across chunks via LLM
│   └── utils/
│       ├── api_provider.py   # Gemini/OpenRouter providers
│       ├── config_manager.py # Config loading
│       └── yaml_handler.py   # YAML parsing
└── experiments/
    ├── retrieval-based/      # GraphRAG with Neo4j
    ├── full_knowledge_graph/ # Full KG baseline
    └── chunkwise_retrieval/  # Per-chunk evaluation
```

### Pipeline Flow
```
Frames → Split into chunks (8 frames default)
              ↓
    Gemini API (per chunk):
    - objects: id, name, description, frames[], bbox
    - spatial_relationships: subject, relation, object
              ↓
    CrossChunkDataAssociator:
    - LLM fixes local IDs based on global summary
    - Programmatic merge into accumulated KG
              ↓
    Final knowledge_graph.json
```

### Key Difference from vizEnc
- **vizEnc**: Local visual encoders (DINOv2/NaRadIO) + embedding matching
- **vl-kgp**: Cloud LLM (Gemini) does detection + tracking in one call per chunk

---

## Integration Ideas

1. **vizEnc anchors → vl-kgp format**: Export anchors as vl-kgp compatible `knowledge_graph.json`
2. **Hybrid**: Use vizEnc for reliable embeddings, vl-kgp for spatial relationships via LLM
3. **Chunk-based processing**: Apply vl-kgp's chunk strategy to vizEnc for better scalability
