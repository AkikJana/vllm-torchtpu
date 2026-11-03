# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker process for PCP8 streaming RPA A/B microbenchmarks."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import NoReturn

import jax
import numpy as np
import torch
from test_pcp8_160k_chunk_benchmark import (
    AB_PARITY_RTOL,
    AB_TRIALS,
    AB_WORKLOAD_SPECS,
    HEAD_DIM,
    KV_PACKING,
    MAX_NUM_SEQS,
    WARMUP_STEPS,
    WORLD_SIZE,
    _build_arm_attention,
    _build_mesh,
    _build_pcp8_workload_host_metadata,
    _gather_global_device_ids,
    _profile_attention,
    _selected_kv_layout_name,
)
from torch import distributed as dist
from torch_tpu._internal import sync
from torch_tpu._internal.distributed import tpu_distributed

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    kernel as pcp_kernel,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as pcp_stage_plan,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
    make_pcp_streaming_rpa_kernel,
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME,
    TP_AXIS_NAME,
)


def _raise_error(exc_type: type[Exception], message: str) -> NoReturn:
    """Raise an error with message."""
    raise exc_type(message)


@dataclasses.dataclass(frozen=True)
class _WorkerRuntimeContext:
    """Per-rank runtime objects for A/B workload execution."""

    rank: int
    cpu_group: object
    mesh: jax.sharding.Mesh
    kv_layout: KVLayout
    result_dir: Path


def _build_workload_caches_and_args(
    spec: dict[str, object],
    meta: dict[str, object],
    ctx: _WorkerRuntimeContext,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Allocate per-arm FP8 KV caches and TPU input tensors for one workload."""
    num_q_heads = int(spec["num_q_heads"])
    num_kv_heads = int(spec["num_kv_heads"])
    page_size = int(spec["page_size"])
    local_q = int(spec["local_query_tokens"])
    max_pages = int(spec["max_local_pages_per_seq"])
    packed_groups = math.ceil(2 * num_kv_heads / KV_PACKING)

    query = torch.zeros(
        (local_q, num_q_heads, HEAD_DIM), dtype=torch.bfloat16, device="tpu"
    )
    key = torch.zeros(
        (local_q, num_kv_heads, HEAD_DIM), dtype=torch.bfloat16, device="tpu"
    )
    val_col = torch.from_numpy(meta["value_fill_by_rank"][ctx.rank]).to(
        device="tpu", dtype=torch.bfloat16
    )
    value = val_col[:, None, None].expand(local_q, num_kv_heads, HEAD_DIM).contiguous()
    if ctx.kv_layout.name == "SEQ_ALONG_LANE":
        cache_shape = (
            max_pages,
            2 * num_kv_heads,
            HEAD_DIM // KV_PACKING,
            KV_PACKING,
            page_size,
        )
    else:
        cache_shape = (
            max_pages,
            page_size,
            packed_groups,
            KV_PACKING,
            HEAD_DIM,
        )
    caches = {
        "base_bkv512_rungs5": torch.zeros(
            cache_shape, dtype=torch.float8_e4m3fn, device="tpu"
        ),
        "cand_bkv1024_rungs6": torch.zeros(
            cache_shape, dtype=torch.float8_e4m3fn, device="tpu"
        ),
    }
    args = (
        query,
        key,
        value,
        torch.from_numpy(meta["seq_lens"]).to(device="tpu"),
        torch.from_numpy(meta["block_tables"]).to(device="tpu"),
        torch.from_numpy(meta["query_start_loc"]).to(device="tpu"),
        torch.from_numpy(meta["request_distribution"]).to(device="tpu"),
    )
    sync.synchronize([*caches.values(), *args], wait=True)
    return caches, args


def _verify_arms_and_collect_errors(
    wl_name: str,
    ops: dict[str, object],
    caches: dict[str, torch.Tensor],
    args: tuple[torch.Tensor, ...],
    meta: dict[str, object],
) -> tuple[dict[str, float], dict[str, float]]:
    """Run warmup steps and verify numeric parity against exact causal reference."""
    rank = int(os.environ["RANK"])
    expected_np = meta["expected_by_rank"][rank]
    valid_np = meta["valid_mask_by_rank"][rank]
    exp_valid = expected_np[valid_np]
    max_abs_by_arm: dict[str, float] = {}
    max_rel_by_arm: dict[str, float] = {}
    for arm_key, arm_op in ops.items():
        out_t = None
        for _ in range(WARMUP_STEPS):
            caches[arm_key], out_t = arm_op(caches[arm_key], *args)
            sync.synchronize([caches[arm_key], out_t], wait=True)
        if out_t is None:
            _raise_error(AssertionError, "Warmup produced no output tensor.")
        got_np = out_t[:, 0, 0].cpu().float().numpy()
        diff = np.abs(got_np[valid_np] - exp_valid)
        max_err = float(np.max(diff)) if diff.size else 0.0
        rel_diff = diff / np.maximum(np.abs(exp_valid), 1.0e-12)
        max_rel_err = float(np.max(rel_diff)) if rel_diff.size else 0.0
        if max_rel_err > AB_PARITY_RTOL:
            msg = (
                f"{wl_name} {arm_key} rank={rank} "
                f"max_rel={max_rel_err:.6e} > {AB_PARITY_RTOL:.6e} "
                f"(max_abs={max_err:.6e})"
            )
            _raise_error(AssertionError, msg)
        max_abs_by_arm[arm_key] = max_err
        max_rel_by_arm[arm_key] = max_rel_err
    return max_abs_by_arm, max_rel_by_arm


def _run_single_workload(
    wl_name: str,
    spec: dict[str, object],
    ctx: _WorkerRuntimeContext,
) -> dict[str, object]:
    """Execute interleaved A/B benchmark and XProf capture for one workload spec."""
    requests = spec["requests"]
    page_size = int(spec["page_size"])
    interleave_size = int(spec["interleave_size"])
    local_q = int(spec["local_query_tokens"])
    max_pages = int(spec["max_local_pages_per_seq"])

    meta = _build_pcp8_workload_host_metadata(
        requests,
        max_num_seqs=MAX_NUM_SEQS,
        max_local_pages_per_seq=max_pages,
        page_size=page_size,
        interleave_size=interleave_size,
        local_query_tokens=local_q,
        pcp_size=WORLD_SIZE,
    )
    caches, args = _build_workload_caches_and_args(spec, meta, ctx)
    ops = {
        "base_bkv512_rungs5": _build_arm_attention(
            jax,
            pcp_stage_plan,
            pcp_kernel,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            mesh=ctx.mesh,
            kv_layout=ctx.kv_layout,
            op_tag=f"{wl_name}_b512",
            interleave_size=interleave_size,
            rungs_override=(32, 64, 128, 256, 512),
            force_bkv_512=True,
        ),
        "cand_bkv1024_rungs6": _build_arm_attention(
            jax,
            pcp_stage_plan,
            pcp_kernel,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            mesh=ctx.mesh,
            kv_layout=ctx.kv_layout,
            op_tag=f"{wl_name}_c1024",
            interleave_size=interleave_size,
            rungs_override=(32, 64, 96, 128, 256, 512),
            force_bkv_512=False,
        ),
    }

    max_abs_by_arm, max_rel_by_arm = _verify_arms_and_collect_errors(
        wl_name, ops, caches, args, meta
    )

    profile_samples_by_arm: dict[str, list[float]] = {}
    if bool(spec["profile_xprof"]):
        for arm_key, arm_op in ops.items():
            prof_dir = ctx.result_dir / f"xprof_{wl_name}_{arm_key}"
            os.environ["TPU_PCP8_160K_PROFILE_DIR"] = str(prof_dir)
            os.environ["TPU_PCP8_PROFILE_SESSION_KEY"] = f"ab_{wl_name}_{arm_key}"
            try:
                caches[arm_key], _, p_samples = _profile_attention(
                    torch,
                    dist,
                    ctx.cpu_group,
                    sync,
                    arm_op,
                    caches[arm_key],
                    args,
                    ctx.rank,
                )
                profile_samples_by_arm[arm_key] = p_samples
            finally:
                os.environ.pop("TPU_PCP8_160K_PROFILE_DIR", None)
                os.environ.pop("TPU_PCP8_PROFILE_SESSION_KEY", None)

    wall_samples: dict[str, list[float]] = {
        "base_bkv512_rungs5": [],
        "cand_bkv1024_rungs6": [],
    }
    orders = (
        ("base_bkv512_rungs5", "cand_bkv1024_rungs6"),
        ("cand_bkv1024_rungs6", "base_bkv512_rungs5"),
    )
    for trial in range(AB_TRIALS):
        for arm_key in orders[trial % 2]:
            dist.barrier(group=ctx.cpu_group)
            t0 = time.perf_counter()
            caches[arm_key], out_t = ops[arm_key](caches[arm_key], *args)
            sync.synchronize([caches[arm_key], out_t], wait=True)
            wall_samples[arm_key].append((time.perf_counter() - t0) * 1e3)

    wl_payload: dict[str, object] = {
        "rank": ctx.rank,
        "workload": wl_name,
        "active_vpages": int(meta["active_vpages"]),
        "max_abs": max_abs_by_arm,
        "max_rel": max_rel_by_arm,
        "wall_samples_ms": wall_samples,
        "profile_samples_ms": profile_samples_by_arm,
    }
    (ctx.result_dir / f"ab_rank_{ctx.rank}_{wl_name}.json").write_text(
        json.dumps(wl_payload, indent=2), encoding="utf-8"
    )
    return wl_payload


def _run_ab_worker(result_dir: Path) -> None:
    """Execute one rank for the A/B microbenchmark."""
    importlib.import_module("torch_tpu")
    rank = int(os.environ["RANK"])
    rank_path = result_dir / f"ab_rank_{rank}.json"
    initialized = False
    payload: dict[str, object] | None = None
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        cpu_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
        torch.empty((1,), device="tpu").cpu()

        global_device_ids = _gather_global_device_ids(torch, dist, tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids, PCP_AXIS_NAME, TP_AXIS_NAME)
        kv_layout = KVLayout[_selected_kv_layout_name()]
        ctx = _WorkerRuntimeContext(
            rank=rank,
            cpu_group=cpu_group,
            mesh=mesh,
            kv_layout=kv_layout,
            result_dir=result_dir,
        )

        per_workload: dict[str, dict[str, object]] = {}
        for wl_name, spec in AB_WORKLOAD_SPECS.items():
            per_workload[wl_name] = _run_single_workload(wl_name, spec, ctx)
            gc.collect()

        payload = {"rank": rank, "workloads": per_workload}
    except (
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        AssertionError,
        OSError,
    ) as exc:
        traceback.print_exc()
        payload = {"rank": rank, "error": repr(exc)}
    finally:
        if payload is not None:
            rank_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except (RuntimeError, OSError):
                traceback.print_exc()
    if payload is not None and "error" in payload:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ab-worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> None:
    """Run the main worker entry point."""
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    _run_ab_worker(args.result_dir)


if __name__ == "__main__":
    _main()
