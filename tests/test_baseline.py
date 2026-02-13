#!/usr/bin/env python3
"""
Test script for VizEnc baseline implementation.

Tests core functionality without running full pipeline.

Usage:
    python tests/test_baseline.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST: Configuration")
    print("="*60)

    try:
        from vizenc_baseline.config import load_all_configs

        config_dir = Path(__file__).parent.parent / "configs"
        pipeline_cfg, models_cfg, graph_cfg = load_all_configs(config_dir)

        print(f"✓ Pipeline config loaded: {pipeline_cfg.data.dataset_name}")
        print(f"✓ Models config loaded: {models_cfg.encoder.type}")
        print(f"✓ Graph config loaded: proximity={graph_cfg.inner.proximity_threshold}")

        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_mask_cropper():
    """Test mask-based cropping."""
    print("\n" + "="*60)
    print("TEST: Mask-based Cropping")
    print("="*60)

    try:
        from vizenc_baseline.preprocessing import MaskCropper

        # Create test image
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))

        # Create test mask (circle)
        mask = np.zeros((100, 100), dtype=bool)
        for y in range(100):
            for x in range(100):
                if (x - 50)**2 + (y - 50)**2 < 25**2:
                    mask[y, x] = True

        bbox = [25, 25, 50, 50]

        # Test cropping
        crop = MaskCropper.crop_by_mask(image, mask, bbox, target_size=(64, 64))

        assert crop.mode == 'RGBA', f"Expected RGBA, got {crop.mode}"
        assert crop.size == (64, 64), f"Expected (64, 64), got {crop.size}"

        print(f"✓ Mask cropping works")
        print(f"  Output mode: {crop.mode}")
        print(f"  Output size: {crop.size}")

        # Test RGBA to RGB conversion
        rgb = MaskCropper.rgba_to_rgb(crop)
        assert rgb.mode == 'RGB', f"Expected RGB, got {rgb.mode}"
        print(f"✓ RGBA to RGB conversion works")

        return True
    except Exception as e:
        print(f"✗ Mask cropping failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_structure():
    """Test graph construction."""
    print("\n" + "="*60)
    print("TEST: Graph Structure")
    print("="*60)

    try:
        from vizenc_baseline.graph import (
            MaskNode, InnerEdge, InterEdge, ChunkGraph
        )

        # Create test nodes
        node1 = MaskNode(
            node_id='test_f0001_m001',
            frame_idx=1,
            mask_idx=1,
            chunk_id='test_chunk',
            bbox=[10, 10, 20, 20],
            embedding=np.random.rand(768),
            category='test',
            description='test object',
            confidence=0.9,
            is_anchor=True,
            track_id=0
        )

        node2 = MaskNode(
            node_id='test_f0001_m002',
            frame_idx=1,
            mask_idx=2,
            chunk_id='test_chunk',
            bbox=[50, 50, 20, 20],
            embedding=np.random.rand(768),
            category='test',
            description='another test',
            confidence=0.85
        )

        print(f"✓ Created nodes: {node1}, {node2}")

        # Create test edges
        inner_edge = InnerEdge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            frame_idx=1,
            distance=50.0
        )

        inter_edge = InterEdge(
            source_id=node1.node_id,
            target_id='test_f0002_m001',
            source_frame=1,
            target_frame=2,
            similarity=0.87,
            track_id=0
        )

        print(f"✓ Created edges: {inner_edge}, {inter_edge}")

        # Create graph
        graph = ChunkGraph('test_chunk', {'inner': {'proximity_threshold': 200}})
        graph.nodes = [node1, node2]
        graph.inner_edges = [inner_edge]
        graph.inter_edges = [inter_edge]

        stats = graph.get_statistics()
        print(f"✓ Graph created: {graph}")
        print(f"  Statistics: {stats}")

        assert stats['total_nodes'] == 2
        assert stats['anchor_nodes'] == 1
        assert stats['inner_edges'] == 1
        assert stats['inter_edges'] == 1

        return True
    except Exception as e:
        print(f"✗ Graph structure failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exporters():
    """Test graph exporters."""
    print("\n" + "="*60)
    print("TEST: Graph Exporters")
    print("="*60)

    try:
        from vizenc_baseline.graph import ChunkGraph, MaskNode
        from vizenc_baseline.graph.exporters import (
            Neo4jExporter, JSONExporter, PickleExporter
        )
        import tempfile

        # Create simple graph
        graph = ChunkGraph('test_chunk', {})
        node = MaskNode(
            node_id='test_f0001_m001',
            frame_idx=1,
            mask_idx=1,
            chunk_id='test_chunk',
            bbox=[10, 10, 20, 20],
            embedding=np.random.rand(768),
            category='test',
            description='test',
            confidence=0.9
        )
        graph.nodes = [node]

        # Test Neo4j export
        neo4j_exp = Neo4jExporter()
        cypher = neo4j_exp.export_graph(graph)
        assert 'CREATE' in cypher
        print(f"✓ Neo4j export works ({len(cypher)} chars)")

        # Test JSON export
        json_exp = JSONExporter()
        json_str = json_exp.export_graph(graph, include_embedding=True)
        assert 'test_chunk' in json_str
        print(f"✓ JSON export works ({len(json_str)} chars)")

        # Test Pickle export (save/load)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle_path = Path(f.name)

        pickle_exp = PickleExporter()
        pickle_exp.save_graph(graph, pickle_path)
        loaded_graph = pickle_exp.load_graph(pickle_path)

        assert loaded_graph.chunk_id == graph.chunk_id
        assert len(loaded_graph.nodes) == len(graph.nodes)
        print(f"✓ Pickle save/load works")

        # Cleanup
        pickle_path.unlink()

        return True
    except Exception as e:
        print(f"✗ Exporters failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_storage():
    """Test storage management."""
    print("\n" + "="*60)
    print("TEST: Storage")
    print("="*60)

    try:
        from vizenc_baseline.storage import (
            ChunkGraphStorage,
            generate_chunk_filename,
            parse_chunk_filename
        )

        # Test filename generation
        filename = generate_chunk_filename(
            'testset', 4, 8, 'pkl', '20260204-1430'
        )
        assert filename == 'testset_0004_chunk8_20260204-1430.pkl'
        print(f"✓ Filename generation: {filename}")

        # Test filename parsing
        parsed = parse_chunk_filename(filename)
        assert parsed['dataset_name'] == 'testset'
        assert parsed['start_frame'] == 4
        assert parsed['chunk_size'] == 8
        print(f"✓ Filename parsing: {parsed}")

        return True
    except Exception as e:
        print(f"✗ Storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VizEnc Baseline Test Suite")
    print("="*60)

    tests = [
        ('Configuration', test_config),
        ('Mask Cropping', test_mask_cropper),
        ('Graph Structure', test_graph_structure),
        ('Exporters', test_exporters),
        ('Storage', test_storage),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
