# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-397B PCP8 long-chunk RPA layout benchmark.

This opt-in TPU benchmark reproduces one full-attention layer for a single
request with 128K cached tokens followed by a 32K prefill chunk.  PCP8 assigns
4K query rows and 16K history rows to every rank.

The serving geometry is intentionally copied from
``tpu_benchmark_daily/scripts/start_prefill_server.sh --config pcp8`` and its
latest startup log:

* DP1 / PCP8 / TP1;
* 32 Q heads, 2 KV heads and head dimension 256 (Qwen3.5-397B);
* FP8 KV cache using HEAD_ALONG_SUBLANE by default, with an opt-in
  SEQ_ALONG_LANE comparison;
* 32,768 max batched tokens and 64 request metadata slots;
* manager block size 4,352, split into 256-token CUSTOM kernel pages;
* 256-token PCP KV-cache interleave; and
* the unified PCP Splash tiling: block_q=128, block_kv=1024, block_kv_compute=256.

The manager block size is recorded in the result but is not directly visible
to RPA: vLLM expands manager blocks into 256-token kernel pages before building
attention metadata.

Run with:

  TPU_RUN_PCP8_160K_BENCHMARK=1 pytest -s \
    tests/kernels/experimental/pcp_streaming_rpa/\
test_pcp8_160k_chunk_benchmark.py

Set ``TPU_PCP8_160K_RESULT_DIR`` to retain per-rank samples and ``summary.json``
outside pytest's temporary directory.

Set ``TPU_PCP8_160K_PROFILE_DIR`` to capture three already-compiled attention
steps from all eight ranks into one merged XProf/TensorBoard run.

Set TPU_PCP8_160K_KV_LAYOUT=SEQ_ALONG_LANE to profile the new layout; the
default remains HEAD_ALONG_SUBLANE.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
import types
import typing
from pathlib import Path

import jax
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_kernel as splash_kernel_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    KVLayout,
)

pytestmark = [pytest.mark.multichip, pytest.mark.nightly]

WORLD_SIZE = 8
MAX_NUM_BATCHED_TOKENS = 32 * 1024
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 256 * 1024
HISTORY_TOKENS = 128 * 1024
CHUNK_TOKENS = 32 * 1024
TOTAL_KV_TOKENS = HISTORY_TOKENS + CHUNK_TOKENS
LOCAL_QUERY_TOKENS = CHUNK_TOKENS // WORLD_SIZE
LOCAL_HISTORY_TOKENS = HISTORY_TOKENS // WORLD_SIZE

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 256
KV_PACKING = 4
PACKED_KV_GROUPS = math.ceil(2 * NUM_KV_HEADS / KV_PACKING)

MANAGER_BLOCK_SIZE = 4352
KERNEL_PAGE_SIZE = 256
PCP_INTERLEAVE_SIZE = 256

MAX_LOCAL_PAGES_PER_SEQ = MAX_MODEL_LEN // (WORLD_SIZE * KERNEL_PAGE_SIZE)
ACTIVE_LOCAL_PAGES = TOTAL_KV_TOKENS // (WORLD_SIZE * KERNEL_PAGE_SIZE)


def _raise_error(exc_type: type[Exception], message: str) -> typing.NoReturn:
    """Raise an error with message."""
    raise exc_type(message)


WARMUP_STEPS = 2
BENCHMARK_STEPS = 10
PROFILE_STEPS = 3
LAUNCH_TIMEOUT_SECONDS = 900
AB_LAUNCH_TIMEOUT_SECONDS = 2400
KV_LAYOUT_ENV = "TPU_PCP8_160K_KV_LAYOUT"
WORKLOAD_ENV = "TPU_PCP8_160K_WORKLOAD"
OVERFLOW_ACTIVE_REQS = 20
OVERFLOW_HISTORY_TOKENS = 1
OVERFLOW_QUERY_TOKENS = 257


def _selected_kv_layout_name() -> str:
    layout = os.getenv(KV_LAYOUT_ENV, "HEAD_ALONG_SUBLANE").strip().upper()
    if layout not in ("HEAD_ALONG_SUBLANE", "SEQ_ALONG_LANE"):
        raise ValueError(f"Unsupported {KV_LAYOUT_ENV}={layout!r}.")
    return layout


def _selected_workload_name() -> str:
    """Return the name of the selected benchmark workload."""
    workload = os.getenv(WORKLOAD_ENV, "aligned").strip().lower()
    if workload not in ("aligned", "misaligned_overflow"):
        msg = f"Unsupported {WORKLOAD_ENV}={workload!r}."
        _raise_error(ValueError, msg)
    return workload


def _require_benchmark() -> None:
    if os.getenv("TPU_RUN_PCP8_160K_BENCHMARK") != "1":
        pytest.skip("set TPU_RUN_PCP8_160K_BENCHMARK=1 to run the PCP8 benchmark")


def _worker_command(result_dir: Path, worker_flag: str = "--worker") -> list[str]:
    """Build command to launch worker process."""
    target = (
        Path(__file__).with_name("_ab_worker.py")
        if worker_flag == "--ab-worker"
        else Path(__file__)
    )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(target.resolve()),
        worker_flag,
        "--result-dir",
        str(result_dir),
    ]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")

    prepared_keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    original_values = {key: os.environ.get(key) for key in prepared_keys}
    try:
        for key in prepared_keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Match the TorchTPU execution controls used by the PCP service launcher.
    env.setdefault("PJRT_DEVICE", "TPU")
    env.setdefault("VLLM_TARGET_DEVICE", "tpu")
    env.setdefault("VLLM_PLUGINS", "torchtpu")
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("TPU_PARALLEL_PRECOMPILE", "1")
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "disabled")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    if env.get("TPU_PCP8_160K_PROFILE_DIR"):
        env["TPU_PCP8_PROFILE_SESSION_KEY"] = (
            f"pcp8_160k_{os.getpid()}_{time.time_ns()}"
        )
    return env


def _run_worker_group(
    result_dir: Path,
    worker_flag: str = "--worker",
    timeout_seconds: int = LAUNCH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run worker group subprocess."""
    process = subprocess.Popen(
        _worker_command(result_dir, worker_flag=worker_flag),
        cwd=Path(__file__).resolve().parents[4],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        raise AssertionError(
            f"PCP8 benchmark workers timed out. Output tail:\n{output[-12000:]}"
        ) from exc
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _gather_global_device_ids(torch, dist, tpu_distributed) -> tuple[int, ...]:
    local_id = torch.tensor(
        [int(tpu_distributed.global_device_id())],
        dtype=torch.int32,
        device="tpu",
    )
    gathered = [torch.empty_like(local_id) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local_id)
    return tuple(int(tensor.cpu().item()) for tensor in gathered)


def _build_mesh(
    jax: object,
    np: object,
    global_device_ids: tuple[int, ...],
    pcp_axis_name: str,
    tp_axis_name: str,
) -> object:
    """Build mesh for the device topology."""
    devices_by_id = {int(device.id): device for device in jax.devices()}
    missing = [
        device_id for device_id in global_device_ids if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            f"JAX does not expose worker TPU ids {missing}; "
            f"available={sorted(devices_by_id)}"
        )
    devices = np.asarray(
        [devices_by_id[device_id] for device_id in global_device_ids]
    ).reshape(WORLD_SIZE, 1)
    return jax.sharding.Mesh(devices, axis_names=(pcp_axis_name, tp_axis_name))


def _make_inputs(torch):
    kv_layout = _selected_kv_layout_name()
    query = torch.zeros(
        (LOCAL_QUERY_TOKENS, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="tpu",
    )
    key = torch.zeros(
        (LOCAL_QUERY_TOKENS, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="tpu",
    )
    value = torch.ones_like(key)
    if kv_layout == "SEQ_ALONG_LANE":
        kv_cache_shape = (
            MAX_LOCAL_PAGES_PER_SEQ,
            2 * NUM_KV_HEADS,
            HEAD_DIM // KV_PACKING,
            KV_PACKING,
            KERNEL_PAGE_SIZE,
        )
    else:
        kv_cache_shape = (
            MAX_LOCAL_PAGES_PER_SEQ,
            KERNEL_PAGE_SIZE,
            PACKED_KV_GROUPS,
            KV_PACKING,
            HEAD_DIM,
        )
    kv_cache = torch.zeros(kv_cache_shape, dtype=torch.float8_e4m3fn, device="tpu")

    workload = _selected_workload_name()
    seq_lens = torch.zeros((MAX_NUM_SEQS,), dtype=torch.int32, device="tpu")
    block_tables = torch.zeros(
        (MAX_NUM_SEQS, MAX_LOCAL_PAGES_PER_SEQ),
        dtype=torch.int32,
        device="tpu",
    )
    if workload == "misaligned_overflow":
        # Each of 20 requests has 1 cached token (V=0) + 257 new tokens (V=1),
        # concentrating 20 * 255 = 5,100 > 4,096 writeback rows onto cache rank 0.
        seq_lens[:OVERFLOW_ACTIVE_REQS] = (
            OVERFLOW_HISTORY_TOKENS + OVERFLOW_QUERY_TOKENS
        )
        total_overflow_q = OVERFLOW_ACTIVE_REQS * OVERFLOW_QUERY_TOKENS
        query_start_loc = torch.full(
            (MAX_NUM_SEQS + 1,), total_overflow_q, dtype=torch.int32, device="tpu"
        )
        query_start_loc[:OVERFLOW_ACTIVE_REQS] = (
            torch.arange(OVERFLOW_ACTIVE_REQS, dtype=torch.int32, device="tpu")
            * OVERFLOW_QUERY_TOKENS
        )
        request_distribution = torch.tensor(
            [0, 0, OVERFLOW_ACTIVE_REQS], dtype=torch.int32, device="tpu"
        )
        block_tables[:OVERFLOW_ACTIVE_REQS, 0] = torch.arange(
            OVERFLOW_ACTIVE_REQS, dtype=torch.int32, device="tpu"
        )
    else:
        # Production pads request metadata to max_num_seqs=64 even though only the
        # first request is live in this benchmark.
        seq_lens[0] = TOTAL_KV_TOKENS
        query_start_loc = torch.full(
            (MAX_NUM_SEQS + 1,), CHUNK_TOKENS, dtype=torch.int32, device="tpu"
        )
        query_start_loc[0] = 0
        request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device="tpu")
        block_tables[0] = torch.arange(
            MAX_LOCAL_PAGES_PER_SEQ, dtype=torch.int32, device="tpu"
        )
    return kv_cache, (
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    )


def _expected_local_output(torch, rank: int):
    cycle = WORLD_SIZE * PCP_INTERLEAVE_SIZE
    if _selected_workload_name() == "misaligned_overflow":
        expected_vals: list[float] = []
        for req_idx in range(OVERFLOW_ACTIVE_REQS):
            owner_start = req_idx * OVERFLOW_QUERY_TOKENS
            owner_end = owner_start + OVERFLOW_QUERY_TOKENS
            chunk_pos = (owner_start // cycle) * cycle + rank * PCP_INTERLEAVE_SIZE
            if chunk_pos + PCP_INTERLEAVE_SIZE <= owner_start:
                chunk_pos += cycle
            while chunk_pos < owner_end:
                seg_start = max(chunk_pos, owner_start)
                seg_end = min(chunk_pos + PCP_INTERLEAVE_SIZE, owner_end)
                for owner_pos in range(seg_start, seg_end):
                    abs_pos = OVERFLOW_HISTORY_TOKENS + (owner_pos - owner_start)
                    expected_vals.append(float(abs_pos) / float(abs_pos + 1))
                chunk_pos += cycle
        expected_vals.extend([0.0] * (LOCAL_QUERY_TOKENS - len(expected_vals)))
        return torch.tensor(expected_vals, dtype=torch.float32)

    # Query ownership is rank-major in 256-token chunks. Q=K=0 makes all
    # attention logits equal, history V is zero and current V is one, so row i
    # is exactly (new tokens visible so far) / (all tokens visible so far).
    offsets = []
    chunk_start = rank * PCP_INTERLEAVE_SIZE
    while chunk_start < CHUNK_TOKENS:
        chunk_end = min(chunk_start + PCP_INTERLEAVE_SIZE, CHUNK_TOKENS)
        offsets.extend(range(chunk_start, chunk_end))
        chunk_start += cycle
    assert len(offsets) == LOCAL_QUERY_TOKENS
    offsets_t = torch.tensor(offsets, dtype=torch.float32)
    return (offsets_t + 1.0) / (HISTORY_TOKENS + offsets_t + 1.0)


def _profile_attention(
    torch, dist, cpu_group, sync, attention, kv_cache, args, rank: int
):
    configured = os.getenv("TPU_PCP8_160K_PROFILE_DIR", "").strip()
    if not configured:
        return kv_cache, None, []

    from torch_tpu._internal.profiler import TpuProfilerConfig

    from vllm_torchtpu import profiler_trace

    profile_dir = str(Path(configured).resolve())
    session_key = os.environ["TPU_PCP8_PROFILE_SESSION_KEY"]
    capture_dir = profiler_trace.rank_capture_dir(profile_dir, rank)
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    canonical_ts = profiler_trace.resolve_canonical_dst_ts(
        profile_dir, rank, session_key=session_key
    )
    Path(capture_dir).mkdir(parents=True, exist_ok=True)

    handler = torch.profiler.tensorboard_trace_handler(
        dir_name=capture_dir, use_gzip=True
    )
    config = TpuProfilerConfig(
        run_dir=capture_dir,
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=1,
        experimental_options={
            "tpu_trace_mode": "TRACE_COMPUTE",
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        },
    )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=handler,
        experimental_config=config,
    )

    profile_samples_ms = []
    output = None
    dist.barrier(group=cpu_group)
    profiler.__enter__()
    try:
        # Start only after every rank has armed its profiler. Warmup and
        # compilation completed before this helper was called.
        dist.barrier(group=cpu_group)
        for step in range(PROFILE_STEPS):
            start = time.perf_counter()
            with torch.profiler.record_function(f"pcp8_160k_attention_step_{step}"):
                kv_cache, output = attention(kv_cache, *args)
                sync.synchronize([kv_cache, output], wait=True)
            profile_samples_ms.append((time.perf_counter() - start) * 1e3)
    finally:
        profiler.__exit__(None, None, None)

    # All captures must be complete before ranks concurrently merge their
    # uniquely prefixed files into the shared TensorBoard run.
    dist.barrier(group=cpu_group)
    profiler_trace.merge_rank_capture(
        capture_dir,
        profile_dir,
        canonical_ts,
        rank,
        world_size=WORLD_SIZE,
    )
    dist.barrier(group=cpu_group)
    if rank == 0:
        profiler_trace.clear_canonical_ts_marker(profile_dir, session_key)
    return kv_cache, output, profile_samples_ms


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist
    from torch_tpu._internal import sync
    from torch_tpu._internal.distributed import tpu_distributed

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    initialized = False
    result: dict[str, object]
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        cpu_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")

        # Match vLLM startup: initialize TorchTPU before querying JAX topology.
        torch.empty((1,), device="tpu").cpu()
        import jax

        from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
        )
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
            PCP_AXIS_NAME,
            TP_AXIS_NAME,
        )

        global_device_ids = _gather_global_device_ids(torch, dist, tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids, PCP_AXIS_NAME, TP_AXIS_NAME)
        kv_cache, args = _make_inputs(torch)
        kv_layout = KVLayout[_selected_kv_layout_name()]
        sync.synchronize([kv_cache, *args], wait=True)

        entry = make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=1.0,
            v_scale=1.0,
            mesh=mesh,
            sliding_window=None,
            sm_scale=1.0 / math.sqrt(HEAD_DIM),
            soft_cap=None,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=PCP_INTERLEAVE_SIZE,
            kv_layout=kv_layout,
        )
        attention = pcp_streaming_jax_op(
            f"pcp8_160k_benchmark::qwen35_397b_h128k_q32k_{kv_layout.name.lower()}",
            entry,
            donate_argnums=(0,),
            mesh=mesh,
            input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        )

        output = None
        for _ in range(WARMUP_STEPS):
            kv_cache, output = attention(kv_cache, *args)
            sync.synchronize([kv_cache, output], wait=True)

        kv_cache, profiled_output, profile_samples_ms = _profile_attention(
            torch, dist, cpu_group, sync, attention, kv_cache, args, rank
        )
        if profiled_output is not None:
            output = profiled_output

        samples_ms = []
        for _ in range(BENCHMARK_STEPS):
            dist.barrier(group=cpu_group)
            start = time.perf_counter()
            kv_cache, output = attention(kv_cache, *args)
            sync.synchronize([kv_cache, output], wait=True)
            samples_ms.append((time.perf_counter() - start) * 1e3)

        assert output is not None
        actual = output[:, 0, 0].cpu().float()
        expected = _expected_local_output(torch, rank)
        output_max_abs = float((actual - expected).abs().max().item())
        if output_max_abs > 0.01:
            raise AssertionError(f"rank={rank} output_max_abs={output_max_abs}")

        result = {
            "rank": rank,
            "global_device_id": int(tpu_distributed.global_device_id()),
            "mesh_device_ids": global_device_ids,
            "samples_ms": samples_ms,
            "median_ms": statistics.median(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "output_max_abs": output_max_abs,
            "profile_samples_ms": profile_samples_ms,
        }
    except BaseException as exc:  # noqa: BLE001 - report worker failures
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001 - best-effort worker cleanup
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _resolve_result_dir(tmp_path: Path) -> Path:
    configured = os.getenv("TPU_PCP8_160K_RESULT_DIR", "").strip()
    layout = _selected_kv_layout_name().lower()
    workload = _selected_workload_name()
    suffix = layout if workload == "aligned" else f"{layout}_{workload}"
    result_dir = (
        Path(configured).resolve()
        if configured
        else (tmp_path / f"pcp8_qwen35_397b_h128k_q32k_{suffix}")
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


HBM_5D_DIM_COUNT = 5


def _has_5d_hbm_float_shape(type_text: str) -> bool:
    """Check whether type text matches 5D HBM float layout."""
    raw_text = type_text.strip()
    shape_match = re.match(r"^(?:f8\w+|bf16|f16|f32|f64)\[([0-9,]+)\]", raw_text)
    if shape_match is None or bool(re.search(r"S\([1-9]\d*\)", raw_text)):
        return False
    dims = [int(dim) for dim in shape_match.group(1).split(",") if dim]
    return len(dims) == HBM_5D_DIM_COUNT and dims[3] in (2, 4)


def _is_donated_kv_cache_hbm_copy_event(
    event_name: str,
    event_stats: tuple[tuple[str, object], ...] | list[tuple[str, object]] = (),
) -> bool:
    """True for synchronous 5D HBM floating-point KV-cache kCopy events."""
    lhs, sep, rhs = event_name.partition(" = ")
    if not re.fullmatch(r"%?copy(?:\.[a-zA-Z0-9_]+)?", lhs.strip()):
        return False
    if sep:
        return _has_5d_hbm_float_shape(rhs.split(" copy(", 1)[0])
    for stat_name, stat_val in event_stats:
        if stat_name == "shape_with_layout" and _has_5d_hbm_float_shape(str(stat_val)):
            return True
    return False


def _collect_plane_events(
    plane: object,
    legacy_prefixes: dict[str, str],
    unified_prefixes: dict[str, str],
) -> tuple[dict[str, list[object]], list[tuple[int, str, object]], list[object]]:
    """Collect legacy, unified, and copy events from an XLA Ops plane."""
    legacy_events: dict[str, list[object]] = {key: [] for key in legacy_prefixes}
    unified_events: list[tuple[int, str, object]] = []
    copy_events: list[object] = []
    lines = getattr(plane, "lines", [])
    for line in lines:
        if getattr(line, "name", "") != "XLA Ops":
            continue
        for event in getattr(line, "events", []):
            ev_name = getattr(event, "name", "")
            for key, prefix in legacy_prefixes.items():
                if ev_name.startswith(prefix):
                    legacy_events[key].append(event)
            for kind, prefix in unified_prefixes.items():
                if ev_name.startswith(prefix):
                    unified_events.append((getattr(event, "start_ns", 0), kind, event))
            if _is_donated_kv_cache_hbm_copy_event(
                ev_name, getattr(event, "stats", [])
            ):
                copy_events.append(event)
    return legacy_events, unified_events, copy_events


def _process_unified_step_buckets(
    plane_name: str,
    xplane_name: str,
    unified_events: list[tuple[int, str, object]],
    copy_events: list[object],
) -> list[dict[str, object]]:
    """Process unified step buckets into sample dicts."""
    unified_events.sort(key=lambda item: item[0])
    step_buckets: list[dict[str, list[object]]] = []
    for _, kind, event in unified_events:
        if kind == "writeback":
            if not step_buckets or step_buckets[-1]["splash"]:
                step_buckets.append({"writeback": [event], "splash": []})
            else:
                step_buckets[-1]["writeback"].append(event)
        else:
            if not step_buckets:
                ev_name = getattr(event, "name", "")
                msg = f"splash event before writeback on {plane_name}: {ev_name}"
                _raise_error(AssertionError, msg)
            step_buckets[-1]["splash"].append(event)
    if len(step_buckets) != PROFILE_STEPS:
        msg = (
            f"expected {PROFILE_STEPS} unified step buckets on "
            f"{plane_name}, found {len(step_buckets)}"
        )
        _raise_error(AssertionError, msg)
    samples: list[dict[str, object]] = []
    for step, bucket in enumerate(step_buckets):
        if not bucket["writeback"] or not bucket["splash"]:
            _raise_error(AssertionError, "Empty writeback or splash bucket.")
        step_start = min(getattr(ev, "start_ns", 0) for ev in bucket["writeback"])
        step_end = max(
            getattr(ev, "start_ns", 0) + getattr(ev, "duration_ns", 0)
            for ev in bucket["writeback"] + bucket["splash"]
        )
        step_copies = [
            ev
            for ev in copy_events
            if getattr(ev, "start_ns", 0) < step_end
            and getattr(ev, "start_ns", 0) + getattr(ev, "duration_ns", 0) > step_start
        ]
        current_ms = (
            sum(getattr(ev, "duration_ns", 0) for ev in bucket["writeback"]) / 1e6
        )
        history_ms = sum(getattr(ev, "duration_ns", 0) for ev in bucket["splash"]) / 1e6
        copy_ms = sum(getattr(ev, "duration_ns", 0) for ev in step_copies) / 1e6
        samples.append(
            {
                "xplane": xplane_name,
                "device_plane": plane_name,
                "step": step,
                "current_state_ms": current_ms,
                "history_output_ms": history_ms,
                "copy_ms": copy_ms,
                "kernel_sum_ms": current_ms + history_ms + copy_ms,
            }
        )
    return samples


def _process_legacy_step_samples(
    plane_name: str,
    xplane_name: str,
    legacy_events: dict[str, list[object]],
) -> list[dict[str, object]]:
    """Process legacy events into step samples."""
    for values in legacy_events.values():
        values.sort(key=lambda event: getattr(event, "start_ns", 0))
    if (
        len(legacy_events["current"]) != PROFILE_STEPS
        or len(legacy_events["history"]) != PROFILE_STEPS
    ):
        _raise_error(AssertionError, "Mismatched legacy step count.")
    samples: list[dict[str, object]] = []
    for step, (current, history) in enumerate(
        zip(legacy_events["current"], legacy_events["history"])
    ):
        current_ms = getattr(current, "duration_ns", 0) / 1e6
        history_ms = getattr(history, "duration_ns", 0) / 1e6
        samples.append(
            {
                "xplane": xplane_name,
                "device_plane": plane_name,
                "step": step,
                "current_state_ms": current_ms,
                "history_output_ms": history_ms,
                "copy_ms": 0.0,
                "kernel_sum_ms": current_ms + history_ms,
            }
        )
    return samples


def _extract_profiled_kernel_summary(
    profile_dir_override: Path | None = None,
) -> dict[str, object]:
    """Extract profiled kernel summary metrics from XPlane protobufs."""
    configured = (
        str(profile_dir_override)
        if profile_dir_override is not None
        else os.getenv("TPU_PCP8_160K_PROFILE_DIR", "").strip()
    )
    if not configured:
        return {}

    import jax

    legacy_prefixes = {
        "current": "%pcp_streaming_attention_current_state_page_groups_multi_head.",
        "history": "%pcp_streaming_attention_history_output_page_groups_multi_head.",
    }
    unified_prefixes = {
        "writeback": "%pcp_write_captured_kv_to_local_cache",
        "splash": "%splash_mha_fwd_segmented",
    }
    core_step_samples: list[dict[str, object]] = []
    xplane_paths = sorted(Path(configured).rglob("*.xplane.pb"))
    if not xplane_paths:
        return {}
    for xplane_path in xplane_paths:
        profile = jax.profiler.ProfileData.from_file(str(xplane_path))
        for plane in getattr(profile, "planes", []):
            pl_name = getattr(plane, "name", "")
            if not pl_name.startswith("/device:TPU:") or "SparseCore" in pl_name:
                continue
            leg, uni, cop = _collect_plane_events(
                plane, legacy_prefixes, unified_prefixes
            )
            if any(kind == "splash" for _, kind, _ in uni):
                core_step_samples.extend(
                    _process_unified_step_buckets(pl_name, xplane_path.name, uni, cop)
                )
            elif leg["current"] or leg["history"]:
                core_step_samples.extend(
                    _process_legacy_step_samples(pl_name, xplane_path.name, leg)
                )

    expected_samples = WORLD_SIZE * PROFILE_STEPS
    if len(core_step_samples) != expected_samples:
        msg = (
            f"expected {expected_samples} profiled core-steps, found "
            f"{len(core_step_samples)}"
        )
        _raise_error(AssertionError, msg)
    current_samples = [
        float(sample["current_state_ms"]) for sample in core_step_samples
    ]
    history_samples = [
        float(sample["history_output_ms"]) for sample in core_step_samples
    ]
    copy_samples = [float(sample["copy_ms"]) for sample in core_step_samples]
    sum_samples = [float(sample["kernel_sum_ms"]) for sample in core_step_samples]
    critical_step_samples = [
        max(
            float(sample["kernel_sum_ms"])
            for sample in core_step_samples
            if sample["step"] == step
        )
        for step in range(PROFILE_STEPS)
    ]
    return {
        "profiled_kernel_current_state_core_step_median_ms": statistics.median(
            current_samples
        ),
        "profiled_kernel_history_output_core_step_median_ms": statistics.median(
            history_samples
        ),
        "profiled_kernel_copy_core_step_median_ms": statistics.median(copy_samples),
        "profiled_kernel_sum_core_step_median_ms": statistics.median(sum_samples),
        "profiled_kernel_critical_step_samples_ms": critical_step_samples,
        "profiled_kernel_critical_step_median_ms": statistics.median(
            critical_step_samples
        ),
        "profiled_kernel_core_step_samples": core_step_samples,
    }


def test_pcp8_qwen35_397b_history128k_chunk32k_layout_benchmark(tmp_path, capsys):
    _require_benchmark()
    result_dir = _resolve_result_dir(tmp_path)
    completed = _run_worker_group(result_dir)

    results = []
    for rank in range(WORLD_SIZE):
        result_path = result_dir / f"rank_{rank}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            results.append({"rank": rank, "error": "missing worker result"})
    errors = [result for result in results if "error" in result]
    assert completed.returncode == 0 and not errors, (
        f"PCP8 benchmark failure: returncode={completed.returncode}, "
        f"results={errors}\nworker output tail:\n{completed.stdout[-12000:]}"
    )

    # A distributed step completes when its slowest rank completes. Aggregate
    # sample-by-sample rank maxima instead of averaging independently timed
    # ranks.
    step_samples_ms = [
        max(float(result["samples_ms"][sample_idx]) for result in results)
        for sample_idx in range(BENCHMARK_STEPS)
    ]
    median_ms = statistics.median(step_samples_ms)
    ordered = sorted(step_samples_ms)
    p90_ms = ordered[math.ceil(0.9 * len(ordered)) - 1]
    summary = {
        "model": "Qwen3.5-397B-A17B-FP8",
        "layout": _selected_kv_layout_name(),
        "workload": _selected_workload_name(),
        "world_size": WORLD_SIZE,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": WORLD_SIZE,
        "tensor_parallel_size": 1,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "kv_cache_dtype": "float8_e4m3fn",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "manager_block_size": MANAGER_BLOCK_SIZE,
        "kernel_page_size": KERNEL_PAGE_SIZE,
        "pcp_interleave_size": PCP_INTERLEAVE_SIZE,
        "history_tokens_global": HISTORY_TOKENS,
        "chunk_tokens_global": CHUNK_TOKENS,
        "history_tokens_per_rank": LOCAL_HISTORY_TOKENS,
        "query_tokens_per_rank": LOCAL_QUERY_TOKENS,
        "active_local_pages": ACTIVE_LOCAL_PAGES,
        "allocated_local_pages": MAX_LOCAL_PAGES_PER_SEQ,
        "warmup_steps": WARMUP_STEPS,
        "benchmark_steps": BENCHMARK_STEPS,
        "profile_steps": PROFILE_STEPS,
        "profile_dir": os.getenv("TPU_PCP8_160K_PROFILE_DIR") or None,
        "profile_step_samples_ms_by_rank": {
            str(result["rank"]): result["profile_samples_ms"] for result in results
        },
        "step_samples_ms": step_samples_ms,
        # These wall-time fields include host dispatch and scheduling. The
        # profiled device-kernel fields below are the primary performance
        # metric when profiling is enabled.
        "median_attention_call_wall_ms": median_ms,
        "min_attention_call_wall_ms": min(step_samples_ms),
        "p90_attention_call_wall_ms": p90_ms,
        "max_output_abs_error": max(
            float(result["output_max_abs"]) for result in results
        ),
    }
    summary.update(_extract_profiled_kernel_summary())
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with capsys.disabled():
        kernel_ms = summary.get("profiled_kernel_critical_step_median_ms")
        kernel_text = (
            f", device_kernel={float(kernel_ms):.3f} ms/layer"
            if kernel_ms is not None
            else ""
        )
        print(
            f"\nPCP8 Qwen3.5-397B {summary['layout']} RPA benchmark: "
            f"history=128K, chunk=32K, local_q=4K{kernel_text}, "
            f"call_wall_median={median_ms:.3f} ms, "
            f"call_wall_p90={p90_ms:.3f} ms, "
            f"samples={step_samples_ms}, result_dir={result_dir}"
        )


# Q=K=0 keeps softmax sums positive; bf16 output cast (2^-8) plus 640-subtile
# f32 online sum and 1/l_acc reciprocal rounding across midpoints (2^-10).
AB_PARITY_RTOL = 2.0**-8 + 2.0**-10
AB_TRIALS = 5
AB_EXTRA_5D_COPY_FLOOR_MS = (
    2.0 * 33_554_432
) / 3.6864e9  # 2 * 32 MiB R+W at 3,686.4 GB/s peak


class _BuildArmKwargs(typing.TypedDict):
    """Typed keyword arguments for _build_arm_attention."""

    mesh: jax.sharding.Mesh
    kv_layout: KVLayout
    op_tag: str
    interleave_size: int
    rungs_override: tuple[int, ...]
    force_bkv_512: bool


class _WorkloadMetaKwargs(typing.TypedDict, total=False):
    """Typed keyword arguments for _build_pcp8_workload_host_metadata."""

    max_num_seqs: typing.Required[int]
    max_local_pages_per_seq: typing.Required[int]
    page_size: typing.Required[int]
    interleave_size: typing.Required[int]
    local_query_tokens: typing.Required[int]
    pcp_size: int


@contextlib.contextmanager
def _override_pcp_splash_arm(
    pcp_stage_plan: types.ModuleType,
    pcp_kernel: types.ModuleType,
    rungs_override: tuple[int, ...],
    *,
    force_bkv_512: bool,
) -> typing.Iterator[None]:
    """Override PCP splash arm rungs and tiling for testing."""
    orig_sp_rungs = pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS
    orig_k_rungs = pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS
    orig_cfg_cls = splash_kernel_mod.SplashConfig

    def _patched_splash_cfg(*args: object, **kwargs: object) -> object:
        """Patch SplashConfig to force block_kv=512 when requested."""
        if force_bkv_512:
            kwargs = dict(kwargs, block_kv=512)
        return orig_cfg_cls(*args, **kwargs)

    pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = rungs_override
    pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = rungs_override
    splash_kernel_mod.SplashConfig = _patched_splash_cfg
    try:
        yield
    finally:
        pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = orig_sp_rungs
        pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = orig_k_rungs
        splash_kernel_mod.SplashConfig = orig_cfg_cls


def _build_arm_attention(
    _jax_mod: types.ModuleType,
    sp_mod: types.ModuleType,
    kernel_mod: types.ModuleType,
    make_kernel_fn: typing.Callable[
        ..., typing.Callable[..., tuple[jax.Array, jax.Array]]
    ],
    jax_op_fn: typing.Callable[..., typing.Callable[..., tuple[jax.Array, jax.Array]]],
    in_specs: tuple[object, ...],
    *,
    mesh: jax.sharding.Mesh,
    kv_layout: KVLayout,
    op_tag: str,
    interleave_size: int,
    rungs_override: tuple[int, ...],
    force_bkv_512: bool,
) -> typing.Callable[..., tuple[jax.Array, jax.Array]]:
    """Build attention arm for PCP8 streaming RPA A/B testing."""
    make_fn = typing.cast(
        "typing.Callable[..., typing.Callable[..., tuple[jax.Array, jax.Array]]]",
        make_kernel_fn,
    )
    base_entry = make_fn(
        q_scale=None,
        k_scale=1.0,
        v_scale=1.0,
        mesh=mesh,
        sliding_window=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=interleave_size,
        kv_layout=kv_layout,
    )

    def _wrapped_entry(
        kv_cache: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Wrap entry function under overridden PCP splash arm rungs."""
        with _override_pcp_splash_arm(
            typing.cast("types.ModuleType", sp_mod),
            typing.cast("types.ModuleType", kernel_mod),
            rungs_override,
            force_bkv_512=force_bkv_512,
        ):
            return base_entry(
                kv_cache,
                query,
                key,
                value,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
            )

    op_fn = typing.cast(
        "typing.Callable[..., typing.Callable[..., tuple[jax.Array, jax.Array]]]",
        jax_op_fn,
    )
    layout_name = kv_layout.name.lower()
    return op_fn(
        f"pcp8_ab::{op_tag}_{layout_name}",
        _wrapped_entry,
        donate_argnums=(0,),
        mesh=mesh,
        input_partition_specs=in_specs,
    )


def _collect_ab_workload_payloads(
    result_dir: Path,
    workload_names: tuple[str, ...],
    world_size: int = WORLD_SIZE,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    """Collect workload JSON results across ranks."""
    errors: list[dict[str, object]] = []
    for rank in range(world_size):
        rp = result_dir / f"ab_rank_{rank}.json"
        if rp.exists():
            r_data = json.loads(rp.read_text(encoding="utf-8"))
            if "error" in r_data:
                errors.append(r_data)
    by_workload: dict[str, list[dict[str, object]]] = {}
    for wl_name in workload_names:
        wl_ranks: list[dict[str, object]] = []
        for rank in range(world_size):
            wp = result_dir / f"ab_rank_{rank}_{wl_name}.json"
            if wp.exists():
                wl_ranks.append(json.loads(wp.read_text(encoding="utf-8")))
            else:
                errors.append(
                    {
                        "rank": rank,
                        "workload": wl_name,
                        "error": "missing ab workload result",
                    }
                )
        by_workload[wl_name] = wl_ranks
    return by_workload, errors


AB_WORKLOAD_SPECS: dict[str, dict[str, object]] = {
    "qwen35_160k_80vp": {
        "requests": ((32768, 131072),),
        "num_q_heads": 32,
        "num_kv_heads": 2,
        "page_size": 256,
        "interleave_size": 256,
        "local_query_tokens": 4096,
        "max_local_pages_per_seq": 128,
        "profile_xprof": True,
    },
    "qwen38_p2_8vp": {
        "requests": ((8192, 0),),
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "page_size": 128,
        "interleave_size": 128,
        "local_query_tokens": 1024,
        "max_local_pages_per_seq": 1089,
        "profile_xprof": False,
    },
    "qwen38_p1_40vp": {
        "requests": ((4096, 4096), (4096, 4096), (8192, 0), (8192, 0), (8192, 0)),
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "page_size": 128,
        "interleave_size": 128,
        "local_query_tokens": 4096,
        "max_local_pages_per_seq": 1089,
        "profile_xprof": False,
    },
    "qwen38_p3b_124vp": {
        "requests": ((24576, 94208), (4096, 4096)),
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "page_size": 128,
        "interleave_size": 128,
        "local_query_tokens": 4096,
        "max_local_pages_per_seq": 1089,
        "profile_xprof": False,
    },
    "qwen38_p4_508vp": {
        "requests": (
            (8192, 122880),
            (8192, 122880),
            (8192, 122880),
            (4096, 122880),
        ),
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "page_size": 128,
        "interleave_size": 128,
        "local_query_tokens": 4096,
        "max_local_pages_per_seq": 1089,
        "profile_xprof": False,
    },
}


def _request_value_scale(req_idx: int) -> float:
    """Return deterministic scale factor for request index."""
    return 1.0 - 0.125 * float(req_idx % 4)


@dataclasses.dataclass(frozen=True)
class _RankFillArrays:
    """Target arrays for rank causal reference fill."""

    expected: np.ndarray
    valid_mask: np.ndarray
    value_fill: np.ndarray


def _fill_pcp8_rank_expected(
    rank: int,
    requests: tuple[tuple[int, int], ...],
    cycle: int,
    interleave_size: int,
    targets: _RankFillArrays,
) -> None:
    """Fill expected causal outputs and masks for one rank."""
    write_pos = 0
    owner_cursor = 0
    for req_idx, (q_len, hist_len) in enumerate(requests):
        owner_start = owner_cursor
        owner_end = owner_cursor + q_len
        owner_cursor = owner_end
        v_scale = _request_value_scale(req_idx)
        chunk_pos = (owner_start // cycle) * cycle + rank * interleave_size
        if chunk_pos + interleave_size <= owner_start:
            chunk_pos += cycle
        while chunk_pos < owner_end:
            seg_start = max(chunk_pos, owner_start)
            seg_end = min(chunk_pos + interleave_size, owner_end)
            for owner_pos in range(seg_start, seg_end):
                q_off = owner_pos - owner_start
                targets.expected[rank, write_pos] = (
                    v_scale * float(q_off + 1) / float(hist_len + q_off + 1)
                )
                targets.valid_mask[rank, write_pos] = True
                targets.value_fill[rank, write_pos] = v_scale
                write_pos += 1
            chunk_pos += cycle


def _build_pcp8_workload_host_metadata(
    requests: tuple[tuple[int, int], ...],
    **kwargs: typing.Unpack[_WorkloadMetaKwargs],
) -> dict[str, object]:
    """Builds host metadata and exact causal reference for multi-request PCP8 shapes."""
    max_num_seqs = int(kwargs["max_num_seqs"])
    max_local_pages_per_seq = int(kwargs["max_local_pages_per_seq"])
    page_size = int(kwargs["page_size"])
    interleave_size = int(kwargs["interleave_size"])
    local_query_tokens = int(kwargs["local_query_tokens"])
    pcp_size = int(kwargs.get("pcp_size", WORLD_SIZE))

    num_reqs = len(requests)
    if num_reqs == 0 or num_reqs > max_num_seqs:
        msg = f"Unsupported request count {num_reqs} for {max_num_seqs=}."
        _raise_error(ValueError, msg)

    virtual_page_size = page_size * pcp_size
    cycle = pcp_size * interleave_size
    seq_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    query_start_loc = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    block_tables = np.zeros((max_num_seqs, max_local_pages_per_seq), dtype=np.int32)

    cursor_q = 0
    cursor_vpage = 0
    for req_idx, (q_len, hist_len) in enumerate(requests):
        total_kv = hist_len + q_len
        req_vpages = (total_kv + virtual_page_size - 1) // virtual_page_size
        if cursor_vpage + req_vpages > max_local_pages_per_seq:
            msg = (
                f"Active vpages {cursor_vpage + req_vpages} exceed "
                f"pool {max_local_pages_per_seq}."
            )
            _raise_error(ValueError, msg)
        seq_lens[req_idx] = total_kv
        query_start_loc[req_idx] = cursor_q
        block_tables[req_idx, :req_vpages] = np.arange(
            cursor_vpage, cursor_vpage + req_vpages, dtype=np.int32
        )
        cursor_q += q_len
        cursor_vpage += req_vpages
    query_start_loc[num_reqs:] = cursor_q
    request_distribution = np.asarray([0, 0, num_reqs], dtype=np.int32)

    expected_by_rank = np.zeros((pcp_size, local_query_tokens), dtype=np.float32)
    valid_mask_by_rank = np.zeros((pcp_size, local_query_tokens), dtype=bool)
    value_fill_by_rank = np.zeros((pcp_size, local_query_tokens), dtype=np.float32)

    targets = _RankFillArrays(
        expected=expected_by_rank,
        valid_mask=valid_mask_by_rank,
        value_fill=value_fill_by_rank,
    )
    for rank in range(pcp_size):
        _fill_pcp8_rank_expected(rank, requests, cycle, interleave_size, targets)

    return {
        "seq_lens": seq_lens,
        "query_start_loc": query_start_loc,
        "request_distribution": request_distribution,
        "block_tables": block_tables,
        "active_vpages": cursor_vpage,
        "expected_by_rank": expected_by_rank,
        "valid_mask_by_rank": valid_mask_by_rank,
        "value_fill_by_rank": value_fill_by_rank,
    }


def _one_sided_welch_non_regression(
    base_samples: list[float], cand_samples: list[float]
) -> tuple[bool, float, float]:
    """One-sided alpha=0.05 Welch non-regression check (cand <= base within spread)."""
    min_welch_samples = 2
    n_b, n_c = len(base_samples), len(cand_samples)
    if n_b < min_welch_samples or n_c < min_welch_samples:
        msg = "Welch check requires at least 2 samples per arm."
        _raise_error(ValueError, msg)
    mean_b = statistics.fmean(base_samples)
    mean_c = statistics.fmean(cand_samples)
    var_b = statistics.variance(base_samples)
    var_c = statistics.variance(cand_samples)
    se_sq = (var_b / n_b) + (var_c / n_c)
    if se_sq <= 0.0:
        return mean_c <= mean_b, 0.0, 1.860
    t_stat = (mean_c - mean_b) / math.sqrt(se_sq)
    num = se_sq * se_sq
    den = ((var_b / n_b) ** 2) / (n_b - 1) + ((var_c / n_c) ** 2) / (n_c - 1)
    df = num / den if den > 0.0 else float(n_b + n_c - 2)
    # Exact alpha=0.05 one-sided critical Student-t quantiles for df >= 2,
    # using Cornish-Fisher expansion from z_0.95 = 1.6448536269514722.
    z = 1.6448536269514722
    g1 = (z**3 + z) / 4.0
    g2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / 96.0
    g3 = (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / 384.0
    t_crit = z + g1 / df + g2 / (df**2) + g3 / (df**3)
    return t_stat <= t_crit, t_stat, t_crit


def test_pcp8_qwen35_160k_and_qwen38_guard_interleaved_ab(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Interleaved PCP8 A/B microbenchmark with Welch non-regression test."""
    if os.getenv("TPU_RUN_PCP8_160K_AB") != "1":
        pytest.skip("set TPU_RUN_PCP8_160K_AB=1 to run the PCP8 interleaved A/B")

    ring_ref_env = os.getenv("TPU_PCP8_RING_CRITICAL_MS", "").strip()
    assert ring_ref_env, (
        "TPU_PCP8_RING_CRITICAL_MS must be set from the 3-slot ring "
        "critical-path median"
    )
    ring_crit = float(ring_ref_env)
    assert ring_crit > 0.0, f"invalid TPU_PCP8_RING_CRITICAL_MS={ring_crit}"

    result_dir = _resolve_result_dir(tmp_path)
    completed = _run_worker_group(
        result_dir,
        worker_flag="--ab-worker",
        timeout_seconds=AB_LAUNCH_TIMEOUT_SECONDS,
    )

    by_workload, errors = _collect_ab_workload_payloads(
        result_dir, tuple(AB_WORKLOAD_SPECS.keys())
    )
    assert completed.returncode == 0, f"PCP8 returncode={completed.returncode}"
    assert not errors, (
        f"PCP8 interleaved A/B errors={errors}\n"
        f"worker output tail:\n{completed.stdout[-12000:]}"
    )

    summary_workloads: dict[str, dict[str, object]] = {}
    for wl_name, spec in AB_WORKLOAD_SPECS.items():
        wl_ranks = by_workload[wl_name]
        step_wall: dict[str, list[float]] = {}
        for arm_key in ("base_bkv512_rungs5", "cand_bkv1024_rungs6"):
            step_wall[arm_key] = [
                max(float(p["wall_samples_ms"][arm_key][idx]) for p in wl_ranks)
                for idx in range(AB_TRIALS)
            ]
        ok_welch, t_stat, t_crit = _one_sided_welch_non_regression(
            step_wall["base_bkv512_rungs5"], step_wall["cand_bkv1024_rungs6"]
        )
        wl_entry: dict[str, object] = {
            "active_vpages": int(wl_ranks[0]["active_vpages"]),
            "base_wall_samples_ms": step_wall["base_bkv512_rungs5"],
            "cand_wall_samples_ms": step_wall["cand_bkv1024_rungs6"],
            "base_wall_median_ms": statistics.median(step_wall["base_bkv512_rungs5"]),
            "cand_wall_median_ms": statistics.median(step_wall["cand_bkv1024_rungs6"]),
            "welch_non_regression_ok": ok_welch,
            "welch_t_stat": t_stat,
            "welch_t_crit_one_sided_0p05": t_crit,
            "max_output_abs_error": {
                arm_key: max(float(p["max_abs"][arm_key]) for p in wl_ranks)
                for arm_key in ("base_bkv512_rungs5", "cand_bkv1024_rungs6")
            },
            "max_output_rel_error": {
                arm_key: max(float(p["max_rel"][arm_key]) for p in wl_ranks)
                for arm_key in ("base_bkv512_rungs5", "cand_bkv1024_rungs6")
            },
        }
        if bool(spec["profile_xprof"]):
            wl_entry["xprof"] = {
                arm_key: _extract_profiled_kernel_summary(
                    result_dir / f"xprof_{wl_name}_{arm_key}"
                )
                for arm_key in ("base_bkv512_rungs5", "cand_bkv1024_rungs6")
            }
        summary_workloads[wl_name] = wl_entry

    ab_summary = {"workloads": summary_workloads}
    (result_dir / "ab_summary.json").write_text(
        json.dumps(ab_summary, indent=2), encoding="utf-8"
    )

    q35 = summary_workloads["qwen35_160k_80vp"]
    q35_xprof = q35["xprof"]
    base_crit = float(
        q35_xprof["base_bkv512_rungs5"]["profiled_kernel_critical_step_median_ms"]
    )
    cand_crit = float(
        q35_xprof["cand_bkv1024_rungs6"]["profiled_kernel_critical_step_median_ms"]
    )
    base_copy = float(
        q35_xprof["base_bkv512_rungs5"]["profiled_kernel_copy_core_step_median_ms"]
    )
    cand_copy = float(
        q35_xprof["cand_bkv1024_rungs6"]["profiled_kernel_copy_core_step_median_ms"]
    )
    with capsys.disabled():
        print(
            f"\nPCP8_AB_SUMMARY qwen35_base_crit_ms={base_crit:.6f} "
            f"qwen35_cand_crit_ms={cand_crit:.6f} "
            f"qwen35_base_copy_ms={base_copy:.6f} "
            f"qwen35_cand_copy_ms={cand_copy:.6f} "
            f"qwen35_base_wall_ms={float(q35['base_wall_median_ms']):.6f} "
            f"qwen35_cand_wall_ms={float(q35['cand_wall_median_ms']):.6f}"
        )
        for wl_name, wl_entry in summary_workloads.items():
            print(
                f"PCP8_AB_GUARD workload={wl_name} "
                f"vpages={wl_entry['active_vpages']} "
                f"base_wall_ms={float(wl_entry['base_wall_median_ms']):.6f} "
                f"cand_wall_ms={float(wl_entry['cand_wall_median_ms']):.6f} "
                f"t={float(wl_entry['welch_t_stat']):.4f} "
                f"t_crit={float(wl_entry['welch_t_crit_one_sided_0p05']):.4f}"
            )

    assert cand_copy <= base_copy + 0.5 * AB_EXTRA_5D_COPY_FLOOR_MS, (
        f"unexpected extra 5D KV-cache copy ms: cand={cand_copy:.6f} "
        f"vs base={base_copy:.6f}"
    )
    assert cand_crit < base_crit, (
        f"qwen35 cand critical {cand_crit:.3f} ms >= base {base_crit:.3f} ms"
    )
    assert cand_crit <= ring_crit, (
        f"qwen35 cand critical {cand_crit:.3f} ms > ring {ring_crit:.3f} ms"
    )
    for wl_name, wl_entry in summary_workloads.items():
        assert bool(wl_entry["welch_non_regression_ok"]), (
            f"non-regression failed on {wl_name}: {wl_entry}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ab-worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> None:
    """Run worker main entry point."""
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _run_worker(args.result_dir)
        return
    msg = "expected --worker"
    _raise_error(ValueError, msg)


if __name__ == "__main__":
    _main()
