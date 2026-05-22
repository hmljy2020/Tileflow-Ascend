"""Minimal tensor dependency checker for TileFlow-Ascend semantic YAML."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .arch import (
    build_arch_graph,
    build_edges,
    build_producer_table,
    route_edges_to_paths,
)
from .ir import ArchGraph, LoopInfo, RoutedTensorEdge, TensorEdge, TensorInfo
from .loops import compile_mapping_loops
from .parsers import collect_mapping_ops, load_yaml, parse_problem


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


def _loop_json(loop: LoopInfo) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": loop.kind,
        "target": loop.target,
        "factors": loop.factors,
        "receive_tile": loop.receive_tile,
        "extent": loop.extent,
        "instance_path": loop.instance_path,
    }
    if loop.scope_type:
        result["scope_type"] = loop.scope_type
    if loop.children:
        result["children"] = [_loop_json(child) for child in loop.children]
    if loop.ops:
        result["ops"] = loop.ops
    return result


def _arch_graph_json(graph: ArchGraph) -> dict[str, Any]:
    return {
        "out_node": graph.out_node,
        "root_concept": graph.root_concept,
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "class": node.class_name,
                "role": node.role,
                "attributes": node.attributes,
                "contains": [
                    {"ref": child.ref, "count": child.count}
                    for child in node.contains
                ],
                "roles": node.roles,
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
    map_yaml = load_yaml(args.map)
    mapping_ops = collect_mapping_ops(map_yaml, problem)
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
        result["loops"] = _loop_json(compile_mapping_loops(map_yaml, arch_graph))
        result["routes"] = [_route_json(route) for route in routes]
    print(json.dumps(result, indent=2))
