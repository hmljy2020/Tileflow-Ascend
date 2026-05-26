from __future__ import annotations

from collections import deque
from typing import Any

from .ir import (
    ArchGraph,
    ArchNode,
    ConceptChild,
    OpInfo,
    ProblemInfo,
    RoutedTensorEdge,
    TensorEdge,
    TensorInfo,
    TransferEdge,
)


def _as_positive_int(value: Any, field_name: str) -> int:
    if value is None:
        return 1
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"{field_name} must be a positive integer")


def _as_required_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"{field_name} must be a positive integer")


def _validate_buffer_size(attributes: dict[str, Any], field_name: str) -> None:
    if "word-bits" in attributes:
        raise ValueError(f"{field_name}.word-bits is not supported; use size only")
    if "size" not in attributes:
        raise ValueError(f"{field_name}.size is required for buffer nodes")
    size = attributes["size"]
    if isinstance(size, bool) or not isinstance(size, (int, float, str)) or size == "":
        raise ValueError(f"{field_name}.size must be a scalar value")


def _parse_contains(
    contains_spec: Any,
    field_name: str,
) -> list[ConceptChild]:
    if contains_spec is None:
        contains_spec = []
    if not isinstance(contains_spec, list):
        raise ValueError(f"{field_name} must be a list")

    contains: list[ConceptChild] = []
    for child_index, child_spec in enumerate(contains_spec):
        child_field = f"{field_name}[{child_index}]"
        if not isinstance(child_spec, dict):
            raise ValueError(f"{child_field} must be a mapping")
        ref = child_spec.get("ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{child_field}.ref must be a non-empty string")
        count = _as_positive_int(child_spec.get("count"), f"{child_field}.count")
        contains.append(ConceptChild(ref=ref, count=count))
    return contains


def build_arch_graph(arch_yaml: dict[str, Any]) -> ArchGraph:
    architecture = arch_yaml.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("arch YAML must contain an 'architecture' mapping")

    node_specs = architecture.get("nodes")
    if not isinstance(node_specs, list):
        raise ValueError("architecture.nodes must be a list")
    concept_specs = architecture.get("concepts")
    if not isinstance(concept_specs, list):
        raise ValueError("architecture.concepts must be a list")

    nodes: dict[str, ArchNode] = {}
    out_nodes: list[str] = []
    for index, spec in enumerate(node_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"architecture.nodes[{index}] must be a mapping")

        name = spec.get("name")
        kind = spec.get("kind")
        class_name = spec.get("class", "")
        role = spec.get("role", "")
        count = spec.get("count")
        attributes = spec.get("attributes", {})
        if not isinstance(name, str) or not name:
            raise ValueError(f"architecture.nodes[{index}].name must be a non-empty string")
        if name in nodes:
            raise ValueError(f"architecture.nodes contains duplicate node '{name}'")
        if not isinstance(kind, str) or kind not in {"buffer", "compute"}:
            raise ValueError(
                f"architecture.nodes[{index}].kind must be 'buffer' or 'compute'"
            )
        if not isinstance(class_name, str):
            raise ValueError(f"architecture.nodes[{index}].class must be a string")
        if not isinstance(role, str):
            raise ValueError(f"architecture.nodes[{index}].role must be a string")
        count = _as_required_positive_int(count, f"architecture.nodes[{index}].count")
        if not isinstance(attributes, dict):
            raise ValueError(f"architecture.nodes[{index}].attributes must be a mapping")
        if spec.get("contains"):
            raise ValueError(f"architecture.nodes[{index}].contains is only valid on concept nodes")
        if "roles" in spec:
            raise ValueError("architecture.nodes roles are not supported")
        if kind == "buffer":
            _validate_buffer_size(attributes, f"architecture.nodes[{index}].attributes")

        nodes[name] = ArchNode(
            name=name,
            kind=kind,
            class_name=class_name,
            role=role,
            count=count,
            attributes=dict(attributes),
            contains=[],
        )
        if role == "out":
            out_nodes.append(name)

    if len(out_nodes) != 1:
        raise ValueError("architecture.nodes must contain exactly one node with role: out")
    if nodes[out_nodes[0]].kind != "buffer":
        raise ValueError("architecture role: out node must be a buffer")

    for index, spec in enumerate(concept_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"architecture.concepts[{index}] must be a mapping")

        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"architecture.concepts[{index}].name must be a non-empty string")
        if name in nodes:
            raise ValueError(f"architecture contains duplicate node '{name}'")
        if "roles" in spec:
            raise ValueError("architecture.concepts roles are not supported")
        contains = _parse_contains(
            spec.get("contains", []),
            f"architecture.concepts[{index}].contains",
        )

        nodes[name] = ArchNode(
            name=name,
            kind="concept",
            class_name="",
            role="",
            count=1,
            attributes={},
            contains=contains,
        )

    referenced_by_concept: set[str] = set()
    for node in nodes.values():
        if node.kind != "concept":
            continue
        for child in node.contains:
            if child.ref not in nodes:
                raise ValueError(f"Concept node '{node.name}' contains unknown node '{child.ref}'")
            if nodes[child.ref].kind == "concept":
                referenced_by_concept.add(child.ref)

    resource_nodes = {
        name: node for name, node in nodes.items() if node.kind in {"buffer", "compute"}
    }
    concept_nodes = {
        name: node for name, node in nodes.items() if node.kind == "concept"
    }
    root_concepts = [
        name for name in concept_nodes if name not in referenced_by_concept
    ]
    if len(root_concepts) != 1:
        raise ValueError("architecture.concepts must contain exactly one root concept node")

    graph = ArchGraph(
        nodes=nodes,
        resource_nodes=resource_nodes,
        concept_nodes=concept_nodes,
        adj={},
        edges=[],
        out_node=out_nodes[0],
        root_concept=root_concepts[0],
    )

    edge_specs = architecture.get("edges")
    if not isinstance(edge_specs, list):
        raise ValueError("architecture.edges must be a list")
    for index, spec in enumerate(edge_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"architecture.edges[{index}] must be a mapping")

        src = spec.get("from")
        dst = spec.get("to")
        kind = spec.get("kind", "transfer")
        name = spec.get("name")
        bandwidth = spec.get("bandwidth")
        converts = spec.get("converts", [])
        attributes = spec.get("attributes", {})
        if not isinstance(src, str) or src not in graph.resource_nodes:
            raise ValueError(f"architecture.edges[{index}].from must name an existing resource node")
        if not isinstance(dst, str) or dst not in graph.resource_nodes:
            raise ValueError(f"architecture.edges[{index}].to must name an existing resource node")
        if not isinstance(kind, str):
            raise ValueError(f"architecture.edges[{index}].kind must be a string")
        if name is None:
            name = f"{src}->{dst}"
        if not isinstance(name, str):
            raise ValueError(f"architecture.edges[{index}].name must be a string")
        if isinstance(converts, str):
            converts = [converts]
        if not isinstance(converts, list) or not all(isinstance(item, str) for item in converts):
            raise ValueError(f"architecture.edges[{index}].converts must be a string list")
        if not isinstance(attributes, dict):
            raise ValueError(f"architecture.edges[{index}].attributes must be a mapping")

        edge = TransferEdge(
            src=src,
            dst=dst,
            kind=kind,
            name=name,
            bandwidth=bandwidth,
            converts=list(converts),
            attributes=dict(attributes),
        )
        graph.edges.append(edge)
        graph.adj.setdefault(src, []).append(edge)

    return graph


def build_producer_table(problem: ProblemInfo) -> dict[str, TensorInfo]:
    producers: dict[str, TensorInfo] = {}

    for tensor in problem.inputs:
        producers[tensor] = TensorInfo(
            name=tensor, producer_kind="global", producer_name="input"
        )

    for op in problem.ops.values():
        producers[op.output] = TensorInfo(
            name=op.output, producer_kind="op", producer_name=op.name
        )

    return producers


def build_edges(
    problem: ProblemInfo,
    mapping_ops: list[OpInfo],
    producer_table: dict[str, TensorInfo],
) -> list[TensorEdge]:
    edges: list[TensorEdge] = []

    for op in mapping_ops:
        for tensor in op.inputs:
            producer = producer_table.get(tensor)
            if producer is None:
                raise ValueError(f"No producer found for input tensor '{tensor}'")
            edges.append(
                TensorEdge(
                    tensor=tensor,
                    producer_kind=producer.producer_kind,
                    producer_name=producer.producer_name,
                    consumer_kind="op",
                    consumer_name=op.name,
                )
            )

    for tensor in problem.outputs:
        producer = producer_table.get(tensor)
        if producer is None:
            raise ValueError(f"No producer found for output tensor '{tensor}'")
        edges.append(
            TensorEdge(
                tensor=tensor,
                producer_kind=producer.producer_kind,
                producer_name=producer.producer_name,
                consumer_kind="global",
                consumer_name="output",
            )
        )

    return edges


def route_shortest_path(
    graph: ArchGraph, src: str, dst: str
) -> list[TransferEdge]:
    if src not in graph.resource_nodes:
        raise ValueError(f"Route source '{src}' is not a resource node in arch graph")
    if dst not in graph.resource_nodes:
        raise ValueError(f"Route destination '{dst}' is not a resource node in arch graph")
    if src == dst:
        return []

    queue: deque[str] = deque([src])
    parents: dict[str, tuple[str, TransferEdge] | None] = {src: None}

    while queue:
        node = queue.popleft()
        for edge in graph.adj.get(node, []):
            if edge.dst in parents:
                continue
            parents[edge.dst] = (node, edge)
            if edge.dst == dst:
                path_edges: list[TransferEdge] = []
                current = dst
                while parents[current] is not None:
                    previous, path_edge = parents[current]
                    path_edges.append(path_edge)
                    current = previous
                path_edges.reverse()
                return path_edges
            queue.append(edge.dst)

    raise ValueError(f"No route found from '{src}' to '{dst}'")


def route_edges_to_paths(
    mapping_ops: list[OpInfo],
    tensor_edges: list[TensorEdge],
    graph: ArchGraph,
) -> list[RoutedTensorEdge]:
    op_locations = {op.name: op.placement for op in mapping_ops}

    def endpoint_location(kind: str, name: str) -> str:
        if kind == "global":
            return graph.out_node
        if kind == "op":
            location = op_locations.get(name, "")
            if not location:
                raise ValueError(f"No placement found for op '{name}'")
            node = graph.nodes.get(location)
            if node is None:
                raise ValueError(f"Placement '{location}' for op '{name}' is not in arch graph")
            if node.kind != "buffer":
                raise ValueError(f"Placement '{location}' for op '{name}' must be a buffer")
            return location
        raise ValueError(f"Unsupported endpoint kind '{kind}'")

    routes: list[RoutedTensorEdge] = []
    for edge in tensor_edges:
        src = endpoint_location(edge.producer_kind, edge.producer_name)
        dst = endpoint_location(edge.consumer_kind, edge.consumer_name)
        transfers = route_shortest_path(graph, src, dst)
        path = [src]
        path.extend(transfer.dst for transfer in transfers)
        routes.append(
            RoutedTensorEdge(
                edge=edge,
                path=path,
                transfers=[transfer.name for transfer in transfers],
            )
        )

    return routes
