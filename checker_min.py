#!/usr/bin/env python3
"""Minimal tensor dependency checker for TileFlow-Ascend semantic YAML."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TensorInfo:
    name: str
    producer_kind: str
    producer_name: str


@dataclass(frozen=True)
class OpInfo:
    name: str
    inputs: list[str]
    output: str
    map_path: str
    placement: str = ""


@dataclass(frozen=True)
class TensorEdge:
    tensor: str
    producer_kind: str
    producer_name: str
    consumer_kind: str
    consumer_name: str


@dataclass(frozen=True)
class ProblemInfo:
    inputs: list[str]
    outputs: list[str]
    ops: dict[str, OpInfo]


@dataclass(frozen=True)
class ArchNode:
    name: str
    kind: str
    class_name: str
    role: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class TransferEdge:
    src: str
    dst: str
    kind: str
    name: str
    bandwidth: Any = None
    converts: list[str] | None = None
    attributes: dict[str, Any] | None = None


@dataclass
class ArchGraph:
    nodes: dict[str, ArchNode]
    adj: dict[str, list[TransferEdge]]
    edges: list[TransferEdge]
    out_node: str


@dataclass(frozen=True)
class RoutedTensorEdge:
    edge: TensorEdge
    path: list[str]
    transfers: list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _as_name_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"{field_name} must be a comma-separated string or string list")


def parse_problem(prob_yaml: dict[str, Any]) -> ProblemInfo:
    problem = prob_yaml.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("prob YAML must contain a 'problem' mapping")

    io = problem.get("io", {})
    if not isinstance(io, dict):
        raise ValueError("problem.io must be a mapping")

    inputs = _as_name_list(io.get("ins"), "problem.io.ins")
    outputs = _as_name_list(io.get("outs"), "problem.io.outs")

    op_specs = problem.get("ops", [])
    if not isinstance(op_specs, list):
        raise ValueError("problem.ops must be a list")

    ops: dict[str, OpInfo] = {}
    for index, spec in enumerate(op_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"problem.ops[{index}] must be a mapping")

        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"problem.ops[{index}].name must be a non-empty string")

        op_inputs = _as_name_list(spec.get("ins"), f"problem.ops[{index}].ins")
        op_outputs = _as_name_list(spec.get("out"), f"problem.ops[{index}].out")
        if len(op_outputs) != 1:
            raise ValueError(f"problem.ops[{index}].out must name exactly one output")

        ops[name] = OpInfo(name=name, inputs=op_inputs, output=op_outputs[0], map_path="")

    return ProblemInfo(inputs=inputs, outputs=outputs, ops=ops)


def collect_mapping_ops(map_yaml: dict[str, Any], problem: ProblemInfo) -> list[OpInfo]:
    mapping = map_yaml.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("map YAML must contain a 'mapping' mapping")

    collected: list[OpInfo] = []

    def visit(node: Any, path: str, placement: str) -> None:
        if not isinstance(node, dict):
            return

        node_type = node.get("node-type")
        node_name = node.get("name") or node.get("target") or node_type or "node"
        current_path = f"{path}/{node_name}" if path else str(node_name)
        current_placement = placement
        target = node.get("target")
        if isinstance(target, str) and target:
            current_placement = target

        if node_type == "Op":
            op_name = node.get("name")
            if not isinstance(op_name, str) or not op_name:
                raise ValueError(f"Op at {current_path} must have a non-empty name")
            spec = problem.ops.get(op_name)
            if spec is None:
                raise ValueError(f"Op '{op_name}' appears in mapping but not problem")
            collected.append(
                OpInfo(
                    name=spec.name,
                    inputs=list(spec.inputs),
                    output=spec.output,
                    map_path=current_path,
                    placement=current_placement,
                )
            )

        subtree = node.get("subtree", [])
        if subtree is None:
            return
        if not isinstance(subtree, list):
            raise ValueError(f"subtree at {current_path} must be a list")
        for index, child in enumerate(subtree):
            visit(child, f"{current_path}/subtree[{index}]", current_placement)

    visit(mapping, "mapping", "")
    return collected


def build_arch_graph(arch_yaml: dict[str, Any]) -> ArchGraph:
    architecture = arch_yaml.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("arch YAML must contain an 'architecture' mapping")

    node_specs = architecture.get("nodes")
    if not isinstance(node_specs, list):
        raise ValueError("architecture.nodes must be a list")

    nodes: dict[str, ArchNode] = {}
    out_nodes: list[str] = []
    for index, spec in enumerate(node_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"architecture.nodes[{index}] must be a mapping")

        name = spec.get("name")
        kind = spec.get("kind")
        class_name = spec.get("class", "")
        role = spec.get("role", "")
        attributes = spec.get("attributes", {})
        if not isinstance(name, str) or not name:
            raise ValueError(f"architecture.nodes[{index}].name must be a non-empty string")
        if name in nodes:
            raise ValueError(f"architecture.nodes contains duplicate node '{name}'")
        if not isinstance(kind, str) or kind not in {"buffer", "compute"}:
            raise ValueError(f"architecture.nodes[{index}].kind must be 'buffer' or 'compute'")
        if not isinstance(class_name, str):
            raise ValueError(f"architecture.nodes[{index}].class must be a string")
        if not isinstance(role, str):
            raise ValueError(f"architecture.nodes[{index}].role must be a string")
        if not isinstance(attributes, dict):
            raise ValueError(f"architecture.nodes[{index}].attributes must be a mapping")

        nodes[name] = ArchNode(
            name=name,
            kind=kind,
            class_name=class_name,
            role=role,
            attributes=dict(attributes),
        )
        if role == "out":
            out_nodes.append(name)

    if len(out_nodes) != 1:
        raise ValueError("architecture.nodes must contain exactly one node with role: out")
    if nodes[out_nodes[0]].kind != "buffer":
        raise ValueError("architecture role: out node must be a buffer")

    graph = ArchGraph(nodes=nodes, adj={}, edges=[], out_node=out_nodes[0])

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
        if not isinstance(src, str) or src not in graph.nodes:
            raise ValueError(f"architecture.edges[{index}].from must name an existing node")
        if not isinstance(dst, str) or dst not in graph.nodes:
            raise ValueError(f"architecture.edges[{index}].to must name an existing node")
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
    if src not in graph.nodes:
        raise ValueError(f"Route source '{src}' is not in arch graph")
    if dst not in graph.nodes:
        raise ValueError(f"Route destination '{dst}' is not in arch graph")
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


def _producer_json(producer: TensorInfo) -> dict[str, str]:
    return {"kind": producer.producer_kind, "name": producer.producer_name}


def _edge_json(edge: TensorEdge) -> dict[str, str]:
    return {
        "tensor": edge.tensor,
        "producer": f"{edge.producer_kind}:{edge.producer_name}",
        "consumer": f"{edge.consumer_kind}:{edge.consumer_name}",
    }


def _route_json(route: RoutedTensorEdge) -> dict[str, Any]:
    edge = route.edge
    return {
        "tensor": edge.tensor,
        "producer": f"{edge.producer_kind}:{edge.producer_name}",
        "consumer": f"{edge.consumer_kind}:{edge.consumer_name}",
        "path": route.path,
        "transfers": route.transfers,
    }


def _arch_graph_json(graph: ArchGraph) -> dict[str, Any]:
    return {
        "out_node": graph.out_node,
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "class": node.class_name,
                "role": node.role,
                "attributes": node.attributes,
            }
            for node in graph.nodes.values()
        ],
        "edges": [
            {
                "from": edge.src,
                "to": edge.dst,
                "kind": edge.kind,
                "name": edge.name,
                "bandwidth": edge.bandwidth,
                "converts": edge.converts or [],
                "attributes": edge.attributes or {},
            }
            for edge in graph.edges
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob", required=True, help="Path to prob.yaml")
    parser.add_argument("--map", required=True, help="Path to map.yaml")
    parser.add_argument("--arch", help="Path to arch.yaml")
    args = parser.parse_args()

    problem = parse_problem(load_yaml(args.prob))
    mapping_ops = collect_mapping_ops(load_yaml(args.map), problem)
    producer_table = build_producer_table(problem)
    edges = build_edges(problem, mapping_ops, producer_table)

    result = {
        "ops": [
            {
                "name": op.name,
                "inputs": op.inputs,
                "output": op.output,
                "map_path": op.map_path,
                "placement": op.placement,
            }
            for op in mapping_ops
        ],
        "producers": {
            name: _producer_json(producer)
            for name, producer in producer_table.items()
        },
        "edges": [_edge_json(edge) for edge in edges],
    }
    if args.arch:
        arch_graph = build_arch_graph(load_yaml(args.arch))
        routes = route_edges_to_paths(mapping_ops, edges, arch_graph)
        result["arch_graph"] = _arch_graph_json(arch_graph)
        result["routes"] = [_route_json(route) for route in routes]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
