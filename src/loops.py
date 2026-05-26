from __future__ import annotations

from typing import Any

from .ir import (
    ArchGraph,
    LoopBlock,
    LoopDescriptor,
    LoopProgram,
    ProblemInfo,
)
from .parsers import _parse_factor_map


def _node_label(node: Any) -> str:
    if not isinstance(node, dict):
        return "node"
    return str(node.get("name") or node.get("target") or node.get("node-type") or "node")


def _ordered_dimensions(
    preferred: list[str],
    *shapes: dict[str, Any],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for dimension in preferred:
        if dimension not in seen:
            result.append(dimension)
            seen.add(dimension)
    for shape in shapes:
        for dimension in shape:
            if dimension not in seen:
                result.append(dimension)
                seen.add(dimension)
    return result


def _parse_tile_shape(value: Any, field_name: str) -> dict[str, Any]:
    shape = _parse_factor_map(value, field_name)
    for dimension, bound in shape.items():
        if not isinstance(dimension, str) or not dimension:
            raise ValueError(f"{field_name} dimensions must be non-empty strings")
        if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
            raise ValueError(f"{field_name}.{dimension} must be a positive integer")
    return shape


def _is_nontrivial_loop(descriptor: LoopDescriptor) -> bool:
    if not isinstance(descriptor.end, int) or not isinstance(descriptor.step, int):
        return True
    return descriptor.start + descriptor.step < descriptor.end


def compile_loop_program(
    map_yaml: dict[str, Any],
    problem: ProblemInfo,
    graph: ArchGraph,
) -> LoopProgram:
    mapping = map_yaml.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("map YAML must contain a 'mapping' mapping")

    var_counts: dict[str, int] = {}

    def make_var(dimension: str) -> str:
        index = var_counts.get(dimension, 0)
        var_counts[dimension] = index + 1
        return f"{dimension.lower()}{index}"

    def make_descriptors(
        parent_shape: dict[str, Any],
        step_shape: dict[str, Any],
        target: str,
        phase: str,
        spacetime: str,
    ) -> list[LoopDescriptor]:
        descriptors: list[LoopDescriptor] = []
        for dimension in _ordered_dimensions(problem.dimensions, parent_shape, step_shape):
            if dimension not in step_shape:
                continue
            end = parent_shape.get(dimension, step_shape[dimension])
            descriptors.append(
                LoopDescriptor(
                    dimension=dimension,
                    start=0,
                    end=end,
                    step=step_shape[dimension],
                    var=make_var(dimension),
                    spacetime=spacetime,
                    phase=phase,
                    target=target,
                )
            )
        return descriptors

    def visit(
        node: Any,
        path: str,
        parent_shape: dict[str, Any],
    ) -> LoopBlock:
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
            tile_kind = tile_type if tile_type in {"spatial", "temporal"} else "tile"

            if target not in graph.nodes:
                raise ValueError(f"Tile target '{target}' at {path} is not in arch graph")

            factors = _parse_tile_shape(node.get("factors"), f"{path}.factors")
            receive_tile = _parse_tile_shape(node.get("receive_tile"), f"{path}.receive_tile")

            loops: list[LoopDescriptor] = []
            if receive_tile:
                loops.extend(
                    make_descriptors(
                        parent_shape=parent_shape,
                        step_shape=receive_tile,
                        target=target,
                        phase="receive",
                        spacetime="temporal",
                    )
                )
                local_shape = receive_tile
            else:
                local_shape = parent_shape

            dispatch_shape = factors or local_shape
            loops.extend(
                make_descriptors(
                    parent_shape=local_shape,
                    step_shape=dispatch_shape,
                    target=target,
                    phase="dispatch",
                    spacetime=tile_kind,
                )
            )

            children = collect_children(
                node,
                path,
                dispatch_shape,
            )
            return LoopBlock(
                kind="tile",
                target=target,
                tile_kind=tile_kind,
                loops=loops,
                children=children,
            )

        if node_type == "Scope":
            scope_type = node.get("type", "")
            if not isinstance(scope_type, str):
                raise ValueError(f"Scope at {path} type must be a string")
            children = collect_children(node, path, parent_shape)
            return LoopBlock(
                kind="scope",
                target=_node_label(node),
                scope_type=scope_type,
                loops=[],
                children=children,
            )

        if node_type == "Op":
            op_name = node.get("name")
            if not isinstance(op_name, str) or not op_name:
                raise ValueError(f"Op at {path} must have a non-empty name")
            return LoopBlock(kind="op", target=op_name, loops=[], children=[])

        raise ValueError(f"Unsupported mapping node-type '{node_type}' at {path}")

    def collect_children(
        node: dict[str, Any],
        path: str,
        child_shape: dict[str, Any],
    ) -> list[LoopBlock]:
        subtree = node.get("subtree", [])
        if subtree is None:
            return []
        if not isinstance(subtree, list):
            raise ValueError(f"subtree at {path} must be a list")
        children: list[LoopBlock] = []
        for index, child in enumerate(subtree):
            child_path = f"{path}/subtree[{index}]/{_node_label(child)}"
            children.append(visit(child, child_path, child_shape))
        return children

    root = visit(
        mapping,
        "mapping",
        dict(problem.instance),
    )
    return LoopProgram(
        root=root,
        dimensions=list(problem.dimensions),
        instance=dict(problem.instance),
    )


def render_loop_pseudocode(program: LoopProgram) -> str:
    def indent_line(level: int, text: str) -> str:
        return f"{'    ' * level}{text}"

    def shift(lines: list[str]) -> list[str]:
        return [f"    {line}" if line else line for line in lines]

    def shape_text(descriptors: list[LoopDescriptor]) -> str:
        parts = [f"{loop.dimension}={loop.step}" for loop in descriptors]
        return ", ".join(parts)

    def loop_header(loop: LoopDescriptor) -> str:
        prefix = "parallel_for" if loop.spacetime == "spatial" else "for"
        comment = f"{loop.phase} {loop.target}"
        return (
            f"{prefix} {loop.var} in range({loop.start}, {loop.end}, {loop.step}):"
            f"  # {comment}"
        )

    def wrap_phase(
        block: LoopBlock,
        phase: str,
        lines: list[str],
        level: int,
    ) -> list[str]:
        descriptors = [loop for loop in block.loops if loop.phase == phase]
        if not descriptors:
            return lines

        rendered = [loop for loop in descriptors if _is_nontrivial_loop(loop)]
        if not rendered:
            tile_shape = shape_text(descriptors)
            suffix = f"({tile_shape})" if tile_shape else "()"
            return [indent_line(level, f"{phase} {block.target}{suffix}:")] + shift(lines)

        result = lines
        for loop in reversed(rendered):
            result = [indent_line(level, loop_header(loop))] + shift(result)
        return result

    def emit(block: LoopBlock, level: int) -> list[str]:
        if block.kind == "op":
            return [indent_line(level, f"{block.target}()")]

        if block.kind == "scope":
            scope_type = block.scope_type or "Scope"
            lines = [indent_line(level, f"with {scope_type} {block.target}:")]
            child_lines: list[str] = []
            for child in block.children or []:
                child_lines.extend(emit(child, level + 1))
            if not child_lines:
                child_lines = [indent_line(level + 1, "pass")]
            return lines + child_lines

        if block.kind == "tile":
            lines: list[str] = []
            for child in block.children or []:
                lines.extend(emit(child, level))
            if not lines:
                lines = [indent_line(level, "pass")]
            lines = wrap_phase(block, "dispatch", lines, level)
            lines = wrap_phase(block, "receive", lines, level)
            return lines

        raise ValueError(f"Unsupported loop block kind '{block.kind}'")

    return "\n".join(emit(program.root, 0))
