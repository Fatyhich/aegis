"""
Neo4j Cypher script exporter for chunk graphs.
"""

from datetime import datetime
from typing import List
from ..graph_builder import ChunkGraph


class Neo4jExporter:
    """Export ChunkGraph to Neo4j Cypher script."""

    def export_graph(self, graph: ChunkGraph) -> str:
        """
        Export graph to Cypher script format.

        Args:
            graph: ChunkGraph instance

        Returns:
            Cypher script as string
        """
        lines = []

        # Header
        lines.append(f"// ===== CHUNK GRAPH: {graph.chunk_id} =====")
        lines.append(f"// Generated: {datetime.now().isoformat()}")
        stats = graph.get_statistics()
        lines.append(f"// Nodes: {stats['total_nodes']} "
                    f"(Anchors: {stats['anchor_nodes']})")
        lines.append(f"// Inner Edges: {stats['inner_edges']}")
        lines.append(f"// Inter Edges: {stats['inter_edges']}")
        lines.append("")

        # Nodes
        lines.append("// ===== NODES =====")
        lines.append("// Creating mask nodes...")
        for node in graph.nodes:
            labels = "Mask:Anchor" if node.is_anchor else "Mask"
            props = {
                'node_id': node.node_id,
                'frame_idx': node.frame_idx,
                'mask_idx': node.mask_idx,
                'is_anchor': node.is_anchor,
                'category': node.category,
                'description': self._escape_string(node.description),
                'bbox_x': node.bbox[0],
                'bbox_y': node.bbox[1],
                'bbox_w': node.bbox[2],
                'bbox_h': node.bbox[3],
                'confidence': round(node.confidence, 3)
            }

            if node.track_id is not None:
                props['track_id'] = node.track_id

            lines.append(f"CREATE (:{labels} {self._format_props(props)});")

        lines.append("")

        # Inner Edges
        lines.append("// ===== INNER EDGES (Spatial Relations) =====")
        lines.append(f"// Creating {len(graph.inner_edges)} spatial edges...")
        for edge in graph.inner_edges:
            lines.append(
                f"MATCH (a:Mask {{node_id: '{edge.source_id}'}}), "
                f"(b:Mask {{node_id: '{edge.target_id}'}}) "
                f"CREATE (a)-[:SPATIAL {{distance: {edge.distance:.2f}, "
                f"relation: '{edge.relation}'}}]->(b);"
            )

        lines.append("")

        # Inter Edges
        lines.append("// ===== INTER EDGES (Temporal Tracking) =====")
        lines.append(f"// Creating {len(graph.inter_edges)} tracking edges...")
        for edge in graph.inter_edges:
            props_str = f"similarity: {edge.similarity:.3f}"
            if edge.track_id is not None:
                props_str += f", track_id: {edge.track_id}"

            lines.append(
                f"MATCH (a:Mask {{node_id: '{edge.source_id}'}}), "
                f"(b:Mask {{node_id: '{edge.target_id}'}}) "
                f"CREATE (a)-[:TRACKED_FROM {{{props_str}}}]->(b);"
            )

        lines.append("")
        lines.append("// ===== DONE =====")
        lines.append(f"// Total commands: {len(graph.nodes) + len(graph.inner_edges) + len(graph.inter_edges)}")

        return "\n".join(lines)

    @staticmethod
    def _format_props(props: dict) -> str:
        """Format properties for Cypher."""
        parts = []
        for key, value in props.items():
            if isinstance(value, str):
                parts.append(f"{key}: '{value}'")
            elif isinstance(value, bool):
                parts.append(f"{key}: {str(value).lower()}")
            else:
                parts.append(f"{key}: {value}")
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _escape_string(s: str) -> str:
        """Escape single quotes in string for Cypher."""
        return s.replace("'", "\\'")

    def export_batch(self, graphs: List[ChunkGraph]) -> str:
        """
        Export multiple graphs to a single Cypher script.

        Args:
            graphs: List of ChunkGraph instances

        Returns:
            Combined Cypher script
        """
        scripts = []

        scripts.append("// ===== MULTI-CHUNK EXPORT =====")
        scripts.append(f"// Total chunks: {len(graphs)}")
        scripts.append(f"// Generated: {datetime.now().isoformat()}")
        scripts.append("")

        for i, graph in enumerate(graphs):
            scripts.append(f"\n// ===== CHUNK {i + 1}/{len(graphs)} =====")
            scripts.append(self.export_graph(graph))

        return "\n".join(scripts)
