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
"""Unit tests for sharded_jax_op integration with torch_tpu JaxCallable."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, PartitionSpec
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed.sharded_jax_op import (
    _build_sharded_callable_cls,
    sharded_jax_op,
)


def test_build_sharded_callable_patches_installed_torch_tpu() -> None:
    """_build_sharded_callable_cls must match installed JaxCallable.__call__."""
    cls = _build_sharded_callable_cls()
    assert cls is not pallas_impl.JaxCallable
    assert issubclass(cls, pallas_impl.JaxCallable)


def test_sharded_jax_op_fake_fn_with_static_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fake_fn normalizes static kwargs and computes local per-rank shard shapes."""
    if len(jax.devices()) < 2:
        assert os.environ.get("_SHARDED_JAX_OP_SUBPROC") != "1", (
            "Expected at least 2 JAX devices under "
            "--xla_force_host_platform_device_count=2"
        )
        env = os.environ.copy()
        xla_flags = env.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in xla_flags:
            env["XLA_FLAGS"] = (
                f"{xla_flags} --xla_force_host_platform_device_count=2".strip()
            )
        env["_SHARDED_JAX_OP_SUBPROC"] = "1"
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_sharded_jax_op_fake_fn_with_static_kwargs",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, (
            f"2-device CPU subprocess failed (rc={res.returncode}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
        return

    devices = np.asarray(jax.devices()[:2])
    assert len(devices) == 2
    mesh = Mesh(devices, ("tp",))

    # Avoid allocating a real TPU tensor inside _placeholder on CPU-only hosts.
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.sharded_jax_op._placeholder",
        lambda aval, spec, m: torch.empty(
            tuple(pallas_impl.get_local_shape(aval.shape, m, spec))
            if spec is not None
            else tuple(aval.shape),
            dtype=pallas_impl.JAX_TO_TORCH_DTYPE_MAP[aval.dtype],
            device="meta",
        ),
    )

    def scale_fn(
        x: jax.Array,
        factor: float = 2.0,
        scales: Sequence[int] = (2, 1),
    ) -> jax.Array:
        if factor > 0.0:
            return (x * jnp.float32(factor * scales[0] + scales[1])).astype(x.dtype)
        return x

    # 1. Replicated input (4, 8) -> sharded output ("tp", None) across 2 ranks:
    #    global output is (4, 8), so per-rank local output must be (2, 8).
    op_shard_out = sharded_jax_op(
        "test_sharded_jax_op::scale_static_kw",
        scale_fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec(None, None),),
        output_partition_specs=PartitionSpec("tp", None),
    )

    # 2. Sharded input (2, 8) along dim 0 -> global (4, 8) -> re-sharded output
    #    (None, "tp") along dim 1: local output (4, 4) differs from both global
    #    (4, 8) and input shard (2, 8).
    op_reshard = sharded_jax_op(
        "test_sharded_jax_op::reshard_static_kw",
        scale_fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec("tp", None),),
        output_partition_specs=PartitionSpec(None, "tp"),
    )

    with FakeTensorMode() as fake_mode:
        x_rep = fake_mode.from_tensor(torch.zeros((4, 8), dtype=torch.float32))
        out = op_shard_out(x_rep, factor=3.0, scales=[3, 1])
        assert out.shape == (2, 8)
        assert out.dtype == torch.float32

        x_shard = fake_mode.from_tensor(torch.zeros((2, 8), dtype=torch.float32))
        out_reshard = op_reshard(x_shard, factor=3.0, scales=[3, 1])
        assert out_reshard.shape == (4, 4)
        assert out_reshard.dtype == torch.float32
