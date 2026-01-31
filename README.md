# AEGIS: Adaptive Environment Graph Identification System

![aegis_logo](data/aegis.gif)

**Adaptive Environment Graph Hierarchies via Dynamic Anchor Selection**

> **Master's Thesis Project**  
> Denis Fatykhoph • Skoltech  
> Advisor: Prof. Gonzalo Ferrer

---

## Overview

AHEAD addresses the fundamental limitation of current hierarchical scene graph methods: their reliance on fixed, environment-specific schemas (e.g., floor→room→object). We propose a framework for **dynamic anchor object identification** that enables adaptive hierarchical knowledge graph construction across heterogeneous environments.

### Key Contributions

- **Dynamic Anchor Selection**: Three comparative strategies (VLM-based, frequency-based, visual clustering) for identifying organizational anchor objects
- **Adaptive Hierarchies**: Data-driven spatial organization that adapts to environment characteristics rather than imposing fixed schemas
- **Cross-Domain Generalization**: Framework operates across indoor and outdoor environments without manual reconfiguration
- **RGB-Only Processing**: Sufficient scene understanding using vision-language models and RGB observations

---

## Architecture

<p align="center">
  <img src="assets/pipeline.png" alt="AHEAD Pipeline" width="800"/>
</p>

Our pipeline integrates VL-KnG baseline for knowledge graph construction with visual feature extraction (SAM + RADIO-2 + Florence-2) to identify anchor objects and assemble adaptive hierarchies.

---

## Method

### Anchor Selection Strategies

1. **VLM-Based Selection**: Leverages vision-language models to reason about organizational importance through natural language prompts
2. **Frequency-Based Selection**: Statistical analysis treating persistent objects as stable organizational anchors
3. **Visual Feature Clustering**: Identifies anchors through unsupervised clustering in learned embedding space (RADIO)

### Adaptive Hierarchy Assembly

Once anchors are identified, we construct a three-level hierarchical knowledge graph:
- **Level 1**: Anchor nodes representing primary organizational structure
- **Level 2**: Associated objects with strong spatial/semantic relationships to anchors
- **Level 3**: Atomic objects instantiated when fine-grained detail is required

---

## Datasets

We evaluate on two complementary datasets:

- **[SCAND](https://www.cs.utexas.edu/~xiao/SCAND/)**: 8.7 hours, 138 trajectories of socially compliant navigation across indoor/outdoor environments
- **[EgoWalk](https://arxiv.org/abs/2505.21282)**: First-person city-walking videos with diverse outdoor navigation scenarios

---

## Benchmarks

Downstream task evaluation on:

- **[HM-EQA/Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa)**: 500 questions across 267 HM3D scenes for Embodied Question Answering
- **Task-driven object navigation**: Cross-environment navigation efficiency

---

## Research Hypotheses

**H1. Adaptive Hierarchy Superiority**: Dynamic anchor-based construction outperforms fixed schemas on downstream tasks

**H2. Cross-Domain Generalization**: Consistent performance across indoor and outdoor environments

**H3. RGB Sufficiency**: RGB + VLMs provide sufficient information for effective scene graphs, reducing hardware requirements

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/AHEAD.git
cd AHEAD

# Create conda environment
conda create -n ahead python=3.10
conda activate ahead

# Install dependencies
pip install -r requirements.txt

# Install VL-KnG baseline
git clone https://github.com/VL-KnG/VL-KnG.git
cd VL-KnG && pip install -e .
```

---

## Quick Start

```python
from ahead import AnchorIdentifier, HierarchyBuilder

# Initialize anchor identifier with your preferred strategy
identifier = AnchorIdentifier(strategy='vlm')  # or 'frequency', 'visual'

# Process RGB observations
anchors = identifier.identify_anchors(rgb_frames)

# Build adaptive hierarchy
hierarchy_builder = HierarchyBuilder()
knowledge_graph = hierarchy_builder.build(anchors, objects)

# Use for downstream tasks
result = knowledge_graph.query("Find the red chair near the table")
```

---

## Project Structure

```
AHEAD/
├── ahead/
│   ├── anchor_selection/      # Anchor identification strategies
│   ├── hierarchy_assembly/    # Adaptive hierarchy construction
│   ├── baselines/             # VL-KnG integration
│   └── evaluation/            # Benchmark evaluation scripts
├── configs/                   # Configuration files
├── data/                      # Dataset loaders
├── scripts/                   # Training and evaluation scripts
└── notebooks/                 # Jupyter notebooks for visualization
```

---

## Results

### Preliminary Findings

- Visual features contribute **7-15%** of anchor objects, complementing VLM-based identification (63-85%)
- Stable anchor tracking across significant viewpoint changes in outdoor environments
- Multi-modal integration captures organizational structures missed by language-only approaches

Comprehensive evaluation results coming soon.

---

## Citation

```bibtex
@mastersthesis{fatykhoph2025ahead,
  title={Adaptive Environment Graph Hierarchies via Dynamic Anchor Selection},
  author={Fatykhoph, Denis},
  year={2025},
  school={Skolkovo Institute of Science and Technology},
  advisor={Ferrer, Gonzalo}
}
```

---

## Related Work

- **[HOV-SG](https://arxiv.org/abs/XXXX)**: Hierarchical Open-Vocabulary 3D Scene Graphs
- **[ConceptGraphs](https://arxiv.org/abs/2309.16650)**: Open-vocabulary 3D scene graphs
- **[VL-KnG](https://arxiv.org/abs/2510.01483)**: Visual Scene Understanding for Navigation Goal Identification

---

## Acknowledgments

This work is part of the "Semantic-integrated Segmentation and LiDAR Point Clouds for Traversability-Aware Graph Exploration" project at Skoltech's Center for Computational and Data-Intensive Science and Engineering (CDISE).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

Denis Fatykhoph - [denis.fatykhoph@skoltech.ru](mailto:denis.fatykhoph@skoltech.ru)

