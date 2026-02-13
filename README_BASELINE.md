# VizEnc Baseline - Topological Graph Pipeline

Modular Python baseline refactored from `all-in-one.ipynb` with topological graph support.

## Key Features

### 1. Mask-based Cropping with Transparent Background
- Uses actual mask shape with RGBA transparency instead of simple bbox cropping
- Provides better semantic representation for encoders
- Implemented in `src/vizenc_baseline/preprocessing/mask_crop.py`

### 2. Topological Graph Structure
- **Every detected mask is a node** (not just anchors)
- Special `is_anchor` field for tracked objects
- **Two types of edges**:
  - `inner`: Spatial relations within a frame
  - `inter`: Temporal tracking between frames

### 3. YAML Configuration
- Separated configuration from code
- Three config files:
  - `configs/pipeline.yaml` - data, filtering, matching, anchors
  - `configs/models.yaml` - SAM, encoder, Florence, zero-shot
  - `configs/graph.yaml` - inner/inter edge parameters

### 4. Multi-format Export
- **Pickle**: Fast binary serialization
- **JSON**: Human-readable, portable
- **Neo4j Cypher**: Direct import to graph database

### 5. Production-Ready
- Modular architecture
- Pydantic validation
- CLI scripts
- Web visualization

## Architecture

```
vizenc_baseline/
├── configs/                    # YAML configurations
│   ├── pipeline.yaml
│   ├── models.yaml
│   └── graph.yaml
├── src/vizenc_baseline/
│   ├── config/                # Config loader & schemas
│   ├── preprocessing/         # Mask-based cropping (CRITICAL!)
│   ├── graph/                 # Topological graph
│   │   ├── node.py           # MaskNode
│   │   ├── edge.py           # InnerEdge, InterEdge
│   │   ├── spatial.py        # Spatial relations
│   │   ├── temporal.py       # Temporal tracking
│   │   ├── graph_builder.py  # ChunkGraph
│   │   └── exporters/        # Neo4j, JSON, Pickle
│   ├── pipeline/              # Main pipeline
│   │   ├── chunk_processor.py
│   │   └── pipeline.py
│   └── storage/               # Storage management
├── scripts/                   # CLI tools
│   ├── run_pipeline.py
│   ├── export_graph.py
│   └── visualize_chunk.py
└── visualization/             # Gradio web app
    └── app.py
```

## Installation

### Dependencies

Install required packages:
```bash
pip install pyyaml pydantic pillow numpy scikit-learn torch transformers gradio
```

### Optional: Neo4j

For graph database visualization:
```bash
# Docker
docker run -p 7474:7474 -p 7687:7687 neo4j:latest

# Or install Neo4j Desktop
```

## Usage

### 1. Configure Pipeline

Edit `configs/pipeline.yaml`:
```yaml
data:
  dataset_name: "egowalk"
  frames_dir: "/path/to/frames"
  start_frame: 4
  max_frames: 8
  chunk_size: 8

filtering:
  enabled: true
  excluded_categories:
    - "person"
    - "shadow"
  min_mask_ratio: 0.1

anchors:
  enabled: true
  averaging_method: "mean"
  threshold: 0.7
```

Edit `configs/models.yaml`:
```yaml
sam:
  version: "sam1"
  checkpoint_path: "/path/to/sam_vit_h_4b8939.pth"

encoder:
  type: "naradio"  # or "dinov2"
  naradio:
    version: "radio_v2.5-b"
    resolution: [512, 512]

florence:
  enabled: true

zero_shot:
  enabled: true
  labels: ["building", "tree", "sky", "ground"]
```

### 2. Run Pipeline

```bash
# Basic usage
python scripts/run_pipeline.py --config-dir configs/

# Custom project directory
python scripts/run_pipeline.py \
    --config-dir configs/ \
    --project-dir /path/to/project
```

### 3. Visualize Results

#### CLI Visualization
```bash
# View single chunk
python scripts/visualize_chunk.py output/chunks/egowalk_0004_chunk8_*.pkl

# View all chunks
python scripts/visualize_chunk.py output/chunks/*.pkl --detailed
```

#### Web Visualization
```bash
python visualization/app.py --chunks-dir output/chunks --port 7860
```

Then open http://localhost:7860

### 4. Export to Neo4j

```bash
# Export to Cypher script
python scripts/export_graph.py \
    output/chunks/egowalk_0004_chunk8_*.pkl \
    --format neo4j \
    --output graph.cypher

# Import to Neo4j
cat graph.cypher | cypher-shell -u neo4j -p password

# Or use Neo4j Browser
# http://localhost:7474
```

### 5. Export to JSON

```bash
# Export with embeddings
python scripts/export_graph.py \
    output/chunks/egowalk_0004_chunk8_*.pkl \
    --format json \
    --output graph.json

# Export without embeddings (smaller file)
python scripts/export_graph.py \
    output/chunks/egowalk_0004_chunk8_*.pkl \
    --format json \
    --output graph_lite.json \
    --no-embedding
```

## Graph Structure

### Node (MaskNode)
```python
{
    'node_id': 'chunk_f0001_m003',      # Unique ID
    'frame_idx': 1,                     # Frame number
    'mask_idx': 3,                      # Mask index in frame
    'chunk_id': 'chunk_0000_0008',
    'is_anchor': True,                  # Tracked object?
    'track_id': 5,                      # Track ID (for anchors)
    'bbox': [x, y, w, h],
    'segmentation': np.array(...),      # Binary mask
    'embedding': np.array(...),         # DINOv2/NaRadIO
    'category': 'building',
    'description': 'Red brick building',
    'confidence': 0.95
}
```

### Inner Edge (Spatial)
```python
{
    'source_id': 'chunk_f0001_m003',
    'target_id': 'chunk_f0001_m007',
    'edge_type': 'inner',
    'relation': 'spatial',
    'frame_idx': 1,
    'distance': 120.5                   # Euclidean distance
}
```

### Inter Edge (Temporal)
```python
{
    'source_id': 'chunk_f0001_m003',
    'target_id': 'chunk_f0002_m005',
    'edge_type': 'inter',
    'source_frame': 1,
    'target_frame': 2,
    'similarity': 0.87,                 # Cosine similarity
    'track_id': 5
}
```

## File Naming Convention

Format: `{dataset}_{start:04d}_chunk{size}_{YYYYMMDD-HHMM}.{ext}`

Examples:
- `egowalk_0004_chunk8_20260204-1430.pkl`
- `egowalk_0004_chunk8_20260204-1430.json`
- `egowalk_0004_chunk8_20260204-1430.cypher`

## Neo4j Queries

### View all anchors
```cypher
MATCH (n:Anchor)
RETURN n
LIMIT 50
```

### Find spatial relations
```cypher
MATCH (a:Mask)-[r:SPATIAL]->(b:Mask)
WHERE r.distance < 100
RETURN a, r, b
LIMIT 50
```

### Track object across frames
```cypher
MATCH path = (start:Anchor {track_id: 0})-[:TRACKED_FROM*]->(end)
RETURN path
```

### Find objects by category
```cypher
MATCH (n:Mask {category: 'building'})
RETURN n
LIMIT 20
```

## Programmatic Usage

```python
from pathlib import Path
from vizenc_baseline import (
    load_all_configs,
    VizEncPipeline,
    ChunkGraphStorage,
    PickleExporter
)

# Load configs
pipeline_cfg, models_cfg, graph_cfg = load_all_configs('configs')

# Run pipeline
pipeline = VizEncPipeline(pipeline_cfg, models_cfg, graph_cfg)
results = pipeline.run()

# Access graphs
for result in results:
    graph = result['graph']
    print(f"Chunk: {graph.chunk_id}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Inner edges: {len(graph.inner_edges)}")
    print(f"Inter edges: {len(graph.inter_edges)}")

# Load saved graph
storage = ChunkGraphStorage('output/chunks', 'egowalk')
chunks = storage.list_chunks()
graph = storage.load_graph(chunks[0])

# Export to Neo4j
from vizenc_baseline.graph.exporters import Neo4jExporter
exporter = Neo4jExporter()
cypher = exporter.export_graph(graph)
print(cypher)
```

## Key Differences from Notebook

| Feature | Notebook | Baseline |
|---------|----------|----------|
| Configuration | Inline cells | YAML files |
| Cropping | BBox only | Mask-based RGBA |
| Graph nodes | Anchors only | ALL masks |
| Edge types | N/A | Inner + Inter |
| Export | Pickle only | Pickle + JSON + Neo4j |
| Visualization | Matplotlib | Gradio web app |
| Modularity | Monolithic | Fully modular |

## Critical Implementation: Mask-based Cropping

The most important change is mask-based cropping:

**Before (Notebook)**:
```python
x, y, w, h = mask['bbox']
crop = image.crop((x, y, x + w, y + h))
embedding = encoder(crop)
```

**After (Baseline)**:
```python
from vizenc_baseline.preprocessing import MaskCropper

crop = MaskCropper.prepare_for_encoder(
    image=image,
    mask=mask['segmentation'],  # Binary mask
    bbox=mask['bbox'],
    encoder_type='naradio',
    padding=0
)
# crop is now RGBA with transparent background!
embedding = encoder(crop)  # Automatically converts RGBA→RGB
```

## Performance Considerations

### Memory
- Storing all segmentation masks can be memory-intensive
- Use `include_segmentation=False` in JSON export to save space
- Embeddings are always included in pickle format

### Speed
- Building inner edges for all pairs can be slow
- Use `proximity_threshold` in `configs/graph.yaml` to limit edges
- Batch processing is supported for embeddings

### Disk Space
- Pickle files are smallest (~10-50 MB per chunk)
- JSON without segmentation (~20-100 MB per chunk)
- JSON with segmentation (~200-500 MB per chunk)
- Cypher scripts are small (~1-5 MB per chunk)

## Troubleshooting

### "MaskCropper not found"
Make sure `src/vizenc_baseline/preprocessing/` is in Python path.

### "RGBA conversion failed"
Check that encoders properly handle RGBA images. All encoders should convert RGBA→RGB automatically.

### "No frames found"
Check `frames_dir` in `configs/pipeline.yaml` and verify glob patterns.

### "Out of memory"
Reduce chunk_size or use batch processing with smaller batch_size.

## Future Extensions

### Advanced Spatial Relations
Extend `AdvancedSpatialRelations` in `graph/spatial.py`:
- Directional: above, below, left_of, right_of
- Topological: overlaps, contains, inside
- Metric: near, far

### Multi-chunk Graphs
Link chunks together for full dataset graph.

### Graph Neural Networks
Use the topological graph for GNN training.

### Real-time Visualization
Stream processing with live Neo4j updates.

## Citation

If you use this baseline, please cite:
```bibtex
@software{vizenc_baseline,
  title = {VizEnc Baseline: Topological Graph Pipeline},
  year = {2026},
  url = {https://github.com/your-repo/vizenc}
}
```

## License

[Your License Here]
