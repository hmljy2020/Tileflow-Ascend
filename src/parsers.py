from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .ir import OpInfo, ProblemInfo


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


def _parse_scalar(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_factor_map(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        result: dict[str, Any] = {}
        for item in value.split():
            if "=" not in item:
                raise ValueError(f"{field_name} item '{item}' must use name=value")
            name, raw_value = item.split("=", 1)
            if not name or not raw_value:
                raise ValueError(f"{field_name} item '{item}' must use name=value")
            result[name] = _parse_scalar(raw_value)
        return result
    raise ValueError(f"{field_name} must be a mapping or name=value string")


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
