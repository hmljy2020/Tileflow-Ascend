from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class ConceptChild:
    ref: str
    count: int = 1


@dataclass(frozen=True)
class ArchNode:
    name: str
    kind: str
    class_name: str
    role: str
    attributes: dict[str, Any]
    contains: list[ConceptChild]
    roles: dict[str, str]


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
    resource_nodes: dict[str, ArchNode]
    concept_nodes: dict[str, ArchNode]
    adj: dict[str, list[TransferEdge]]
    edges: list[TransferEdge]
    out_node: str
    root_concept: str


@dataclass(frozen=True)
class RoutedTensorEdge:
    edge: TensorEdge
    path: list[str]
    transfers: list[str]


@dataclass(frozen=True)
class LoopInfo:
    kind: str
    target: str
    factors: dict[str, Any]
    receive_tile: dict[str, Any]
    extent: int | None
    instance_path: str
    scope_type: str = ""
    children: list["LoopInfo"] | None = None
    ops: list[str] | None = None
