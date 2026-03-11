#!/usr/bin/env python3
"""
Debug script to check what's happening in the pipeline.
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# aegis is installed via: uv pip install -e .

# Load the chunk
chunk_file = Path('output/chunks').glob('*.pkl')
chunk_file = list(chunk_file)[0]

print(f"Loading: {chunk_file}")

with open(chunk_file, 'rb') as f:
    graph = pickle.load(f)

print("\n" + "="*60)
print("DETAILED DEBUG")
print("="*60)

# Check nodes by frame
from collections import defaultdict
nodes_by_frame = defaultdict(list)
for node in graph.nodes:
    nodes_by_frame[node.frame_idx].append(node)

print("\n1. NODES PER FRAME:")
for frame_idx in sorted(nodes_by_frame.keys()):
    nodes = nodes_by_frame[frame_idx]
    print(f"\n  Frame {frame_idx}: {len(nodes)} nodes")

    # Check embeddings
    has_emb = sum(1 for n in nodes if n.embedding is not None and len(n.embedding) > 0)
    print(f"    • With embeddings: {has_emb}/{len(nodes)}")

    # Check categories
    cats = [n.category for n in nodes]
    print(f"    • Categories: {set(cats)}")

    # Check descriptions
    has_desc = sum(1 for n in nodes if n.description and n.description != '')
    print(f"    • With descriptions: {has_desc}/{len(nodes)}")

    # Check track IDs
    track_ids = [n.track_id for n in nodes]
    print(f"    • Track IDs: {set(track_ids)}")

    # Sample embedding size
    if has_emb > 0:
        sample_node = [n for n in nodes if n.embedding is not None][0]
        print(f"    • Embedding size: {len(sample_node.embedding)}")

print("\n2. INTER EDGES CHECK:")
print(f"  Total inter edges: {len(graph.inter_edges)}")

if len(graph.inter_edges) > 0:
    print("\n  Sample inter edges:")
    for i, edge in enumerate(graph.inter_edges[:5]):
        print(f"    {i+1}. {edge}")
else:
    print("\n  ❌ NO INTER EDGES FOUND!")
    print("\n  Debugging why...")

    # Check if we have embeddings to match
    frame0_nodes = nodes_by_frame[0]
    frame1_nodes = nodes_by_frame[1] if 1 in nodes_by_frame else []

    if frame0_nodes and frame1_nodes:
        # Get embeddings
        emb0 = [n.embedding for n in frame0_nodes if n.embedding is not None]
        emb1 = [n.embedding for n in frame1_nodes if n.embedding is not None]

        print(f"\n  Frame 0: {len(emb0)} embeddings")
        print(f"  Frame 1: {len(emb1)} embeddings")

        if len(emb0) > 0 and len(emb1) > 0:
            # Test matching manually
            from sklearn.metrics.pairwise import cosine_similarity
            emb0_array = np.array(emb0)
            emb1_array = np.array(emb1)

            sim_matrix = cosine_similarity(emb0_array, emb1_array)
            max_sim = sim_matrix.max()
            mean_sim = sim_matrix.mean()

            print(f"\n  Similarity matrix shape: {sim_matrix.shape}")
            print(f"  Max similarity: {max_sim:.3f}")
            print(f"  Mean similarity: {mean_sim:.3f}")
            print(f"  Threshold used: 0.7")

            if max_sim < 0.7:
                print(f"\n  ❌ PROBLEM: Max similarity ({max_sim:.3f}) < threshold (0.7)")
                print(f"     Embeddings are too different between frames!")
                print(f"     This means objects are not being tracked.")
        else:
            print("\n  ❌ PROBLEM: No embeddings in one or both frames!")
    else:
        print("\n  ❌ PROBLEM: Not enough frames to match!")

print("\n3. CATEGORIES CHECK:")
all_cats = [n.category for n in graph.nodes]
from collections import Counter
cat_counts = Counter(all_cats)
for cat, count in cat_counts.most_common():
    print(f"  • {cat}: {count}")

if 'unknown' in cat_counts and cat_counts['unknown'] == len(graph.nodes):
    print("\n  ❌ PROBLEM: All categories are 'unknown'")
    print("     Florence-2 and zero-shot classification didn't run!")

print("\n" + "="*60)
