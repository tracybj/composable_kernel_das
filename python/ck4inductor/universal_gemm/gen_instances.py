# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from .._template_parser import TemplateInstance, find_template_instances
from ..util import library_path
from .op import CKGemmOperation

log = logging.getLogger(__name__)

_TEMPLATE_NAME = "DeviceGemm_Xdl_CShuffleV3"
_EXPECTED_ARGS_WITH_EXPLICIT_TAIL = 42
_CONCRETE_DTYPES = {"F16", "BF16", "F32", "F64", "I8", "int8_t"}
_SCHEDULERS = [
    "BlockGemmPipelineScheduler::Intrawave",
    "BlockGemmPipelineScheduler::Interwave",
]
_GEMM_SPECS = [
    "GemmSpecialization::Default",
    "GemmSpecialization::MPadding",
    "GemmSpecialization::NPadding",
    "GemmSpecialization::KPadding",
    "GemmSpecialization::MNPadding",
    "GemmSpecialization::MKPadding",
    "GemmSpecialization::NKPadding",
    "GemmSpecialization::MNKPadding",
]


def _ck_library_dir():
    gemm_instances_path = (
        Path(library_path())
        / "src"
        / "tensor_operation_instance_hcu"
        / "gpu"
        / "gemm_universal"
    )
    if not gemm_instances_path.exists():
        log.error("CK library path %s does not exist", gemm_instances_path)
        return None
    return gemm_instances_path


def _iter_xdl_cshuffle_v3_instances(root: Path) -> Iterable[TemplateInstance]:
    for instance in find_template_instances(root, _TEMPLATE_NAME, suffixes=(".hpp", ".cpp")):
        if instance.path.name.startswith("device_gemm_xdl_universal_"):
            yield instance


def _convert_token(token: str):
    token = token.strip()
    if token in {"Tuple<>", "ck::Tuple<>"}:
        return tuple()
    if token.startswith("S<") and token.endswith(">"):
        inner = token[2:-1].strip()
        if not inner:
            return tuple()
        return tuple(int(part.strip()) for part in inner.split(","))
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    try:
        return int(token)
    except ValueError:
        return token


def _bool_token(token: str):
    value = _convert_token(token)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return value


def _op_from_instance(instance: TemplateInstance) -> CKGemmOperation:
    args = list(instance.args)
    if len(args) != _EXPECTED_ARGS_WITH_EXPLICIT_TAIL:
        raise ValueError(
            f"{instance.path}: expected {_EXPECTED_ARGS_WITH_EXPLICIT_TAIL} args for {_TEMPLATE_NAME}, got {len(args)}"
        )

    if any(args[index] not in _CONCRETE_DTYPES for index in (3, 4, 5)):
        raise ValueError(f"{instance.path}: skipping non-concrete dtype args {args[3:6]}")
    values = [_convert_token(arg) for arg in args]
    values.insert(2, tuple())
    values.insert(6, tuple())
    values[30] = _bool_token(args[28])
    values[37] = _bool_token(args[35])
    return CKGemmOperation(*values)


def parse_instances(instances: Iterable[TemplateInstance]) -> List[CKGemmOperation]:
    ops = []
    for instance in instances:
        try:
            ops.append(_op_from_instance(instance))
        except (TypeError, ValueError) as exc:
            log.debug("%s when parsing %s", exc, instance.path)
    return ops


@lru_cache(None)
def gen_ops_library() -> List[CKGemmOperation]:
    ck_library_dir = _ck_library_dir()
    if not ck_library_dir:
        return []
    op_instances = parse_instances(_iter_xdl_cshuffle_v3_instances(ck_library_dir))
    log.debug("ck gemm instances from library: %d", len(op_instances))
    substitute_instances = []
    for instance in op_instances:
        schedulers = (
            _SCHEDULERS
            if instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
            else [instance.block_gemm_pipeline_scheduler]
        )
        gemm_specs = (
            _GEMM_SPECS
            if instance.gemm_specialization == "GemmSpec"
            else [instance.gemm_specialization]
        )
        for scheduler in schedulers:
            for spec in gemm_specs:
                substitute_instances.append(
                    replace(
                        instance,
                        block_gemm_pipeline_scheduler=scheduler,
                        gemm_specialization=spec,
                    )
                )
    return substitute_instances


@lru_cache(None)
def gen_ops_preselected() -> List[CKGemmOperation]:
    """DCU preselected GEMM instances are intentionally empty until benchmarked."""

    return []


if __name__ == "__main__":
    for op in gen_ops_library():
        print(op.name())
