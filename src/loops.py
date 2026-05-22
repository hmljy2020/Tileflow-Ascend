from __future__ import annotations

from typing import Any

from .ir import ArchGraph, LoopInfo
from .parsers import _parse_factor_map


def _concept_child_count(graph: ArchGraph, parent: str, target: str) -> int | None:
    parent_node = graph.concept_nodes.get(parent)
    if parent_node is None:
        return None
    for child in parent_node.contains:
        if child.ref == target:
            return child.count
    return None


def compile_mapping_loops(map_yaml: dict[str, Any], graph: ArchGraph) -> LoopInfo:
    mapping = map_yaml.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("map YAML must contain a 'mapping' mapping")

    def node_label(node: Any) -> str:
        if not isinstance(node, dict):
            return "node"
        return str(node.get("name") or node.get("target") or node.get("node-type") or "node")

    def visit(
        node: Any,
        path: str,
        concept_scope: str,
        instance_path: str,
    ) -> LoopInfo:
        if not isinstance(node, dict):
            raise ValueError(f"mapping node at {path} must be a mapping")

        node_type = node.get("node-type")
        if node_type == "Tile":
            target = node.get("target")
            if not isinstance(target, str) or not target:
                raise ValueError(f"Tile at {path} must have a non-empty target")
            tile_type = node.get("type", "temporal")
            if not isinstance(tile_type, str):
                raise ValueError(f"Tile at {path} type must be a string")
            kind = tile_type if tile_type in {"spatial", "temporal"} else "tile"

            next_concept_scope = concept_scope
            extent: int | None = None
            current_instance_path = f"{instance_path}/{target}"
            if target in graph.concept_nodes:
                if kind == "spatial":
                    extent = _concept_child_count(graph, concept_scope, target)
                    if extent is None:
                        raise ValueError(
                            f"Spatial target '{target}' at {path} is not contained by "
                            f"concept scope '{concept_scope}'"
                        )
                next_concept_scope = target
            elif target not in graph.resource_nodes:
                raise ValueError(f"Tile target '{target}' at {path} is not in arch graph")

            loop = LoopInfo(
                kind=kind,
                target=target,
                factors=_parse_factor_map(node.get("factors"), f"{path}.factors"),
                receive_tile=_parse_factor_map(node.get("receive_tile"), f"{path}.receive_tile"),
                extent=extent,
                instance_path=current_instance_path,
                children=[],
                ops=[],
            )
            fill_loop_children(loop, node, path, next_concept_scope, current_instance_path)
            return loop

        if node_type == "Scope":
            target = node_label(node)
            scope_type = node.get("type", "")
            if not isinstance(scope_type, str):
                raise ValueError(f"Scope at {path} type must be a string")
            loop = LoopInfo(
                kind="scope",
                target=target,
                factors={},
                receive_tile={},
                extent=None,
                instance_path=f"{instance_path}/{target}",
                scope_type=scope_type,
                children=[],
                ops=[],
            )
            fill_loop_children(loop, node, path, concept_scope, loop.instance_path)
            return loop

        if node_type == "Op":
            op_name = node.get("name")
            if not isinstance(op_name, str) or not op_name:
                raise ValueError(f"Op at {path} must have a non-empty name")
            return LoopInfo(
                kind="op",
                target=op_name,
                factors={},
                receive_tile={},
                extent=None,
                instance_path=f"{instance_path}/{op_name}",
                ops=[op_name],
            )

        raise ValueError(f"Unsupported mapping node-type '{node_type}' at {path}")

    def fill_loop_children(
        loop: LoopInfo,
        node: dict[str, Any],
        path: str,
        concept_scope: str,
        instance_path: str,
    ) -> None:
        children = loop.children
        ops = loop.ops
        if children is None or ops is None:
            raise ValueError("internal error: mutable loop fields were not initialized")

        subtree = node.get("subtree", [])
        if subtree is None:
            return
        if not isinstance(subtree, list):
            raise ValueError(f"subtree at {path} must be a list")
        for index, child in enumerate(subtree):
            child_path = f"{path}/subtree[{index}]/{node_label(child)}"
            if isinstance(child, dict) and child.get("node-type") == "Op":
                op_name = child.get("name")
                if not isinstance(op_name, str) or not op_name:
                    raise ValueError(f"Op at {child_path} must have a non-empty name")
                ops.append(op_name)
                continue
            children.append(visit(child, child_path, concept_scope, instance_path))

    return visit(mapping, "mapping", graph.root_concept, graph.root_concept)
