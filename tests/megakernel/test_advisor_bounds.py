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
"""VMEM floorplan, roofline bounds, and verdict tests for TPU advisor."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from tests.megakernel.megakernel_fakes import TINY_TILES, make_fake_model_spec
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import spec_bridge
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles

_QWEN35_08B_CFG = {
    "model_type": "qwen3_5",
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "top_k": 50,
    "quantization_config": {"quant_method": "fp8"},
}
_QWEN35_397B_CFG = {
    "model_type": "qwen3_5_moe",
    "num_hidden_layers": 60,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "quantization_config": {"quant_method": "fp8"},
}
_MIXTRAL_8X7B_CFG = {
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}
_DEEPSEEK_V3_CFG = {
    "model_type": "deepseek_v3",
    "num_hidden_layers": 61,
    "first_k_dense_replace": 3,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "fp8"},
}
_KIMI_K3_PRO_CFG = {
    "model_type": "kimi_linear",
    "num_hidden_layers": 93,
    "first_k_dense_replace": 1,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 6144,
    "routed_expert_hidden_size": 3584,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "linear_attn_config": {
        "kda_layers": list(range(69)),
        "num_heads": 96,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "short_conv_kernel_size": 4,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}

_NUM_SCOPES = 4
_NUM_REGIMES = 12
_KIMI_ROUTED_HIDDEN = 3584
_KIMI_KDA_LAYERS = 69
_MEGACORE_CORES = 2
_TPU7X_FP8_FLOPS = 2.30e15
_TPU7X_FP8_ROW_FLOOR = 32
_V6E_FP8_FLOPS = 9.20e14
_MIXTRAL_LAYERS = 32
_MIXTRAL_MOE_INTER = 14336
_V5E_HBM_CAP_GB = 24.0
_FFN_DIM_2048 = 2048
_DSV3_HEAD_DIM = 128
_DSV3_MOE_LAYERS = 58
_VERIFY_ACTIVE_TOKENS = 32
_MIXED_ACTIVE_TOKENS = 160
_TPU7X_VMEM_MIB = 62.0
_CUSTOM_VMEM_MIB = 58.0
_STAGE_OPS_PER_STEP = 18
_PREFILL_STAGE_FLOOR_MIN_US = 30_000.0
_PREFILL_STAGE_FLOOR_MAX_US = 60_000.0
_MEASURED_COVERAGE = 0.85
_QWEN35_SPLIT_VERIFY_ROWS = 452
_QWEN35_VERIFY_ROWS_MIN = 64
_K3_PREFILL_CALLS = 13
_MULTI_HOST_TP = 16
_MULTI_HOST_TP_HOPS = 8
_EXPECTED_VMEM_BUDGET_BYTES = 60817408
_BYTES_PER_MIB = 1 << 20
_BYTES_PER_GIB = 1 << 30
_TPU7X_CORES_PER_DEVICE = 2


def _preset_workload(
    layout: advisor.ParallelLayout, **changes: object
) -> advisor.WorkloadSpec:
    """Return a WorkloadSpec on a deployment preset layout with `changes` applied."""
    mesh = advisor.WorkloadSpec(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    return dataclasses.replace(mesh, **changes)


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify per-token live VMEM interval math and TP/EP sharding on Kimi-K3.1-Pro."""
    pk3 = advisor.resolve_model_profile("kimi-k3.1-pro")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    expected_grp_tok = parts_grp["residual_buffer_bytes"] + max(
        parts_grp["attn_scratch_bytes"],
        parts_grp["linear_attn_scratch_bytes"],
        parts_grp["moe_scratch_bytes"],
    )
    assert parts_grp["total_per_token_bytes"] == expected_grp_tok
    assert parts_grp["b_tok"] == expected_grp_tok
    rep_grp = advisor.advise(pk3, wl_grp, "tpu7x")
    cross_sc = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in rep_grp.scopes if s.scope == "single_layer")
    assert cross_sc.max_vmem_tokens >= _QWEN35_VERIFY_ROWS_MIN
    assert one_l_sc.max_vmem_tokens > cross_sc.max_vmem_tokens

    wl_msk = dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    parts_msk = advisor.compute_per_token_vmem_bytes(pk3, wl_msk)
    assert parts_grp["total_per_token_bytes"] < parts_msk["total_per_token_bytes"]

    wl_no_alias = dataclasses.replace(wl_grp, vmem_alias=False)
    parts_no_alias = advisor.compute_per_token_vmem_bytes(pk3, wl_no_alias)
    assert parts_no_alias["total_per_token_bytes"] > parts_grp["total_per_token_bytes"]


def test_multi_tpu_vmem_and_fp8_dtype_physics() -> None:
    """Verify TPU FP8 physics, VMEM buffer overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")
    cv5e = advisor.resolve_chip_spec("v5e")
    cv5p = advisor.resolve_chip_spec("v5p")
    cv4 = advisor.resolve_chip_spec("v4")

    assert (c7x.native_ocp_fp8, c7x.peak_flops("fp8"), c7x.lhs_row_floor("fp8")) == (
        True,
        _TPU7X_FP8_FLOPS,
        _TPU7X_FP8_ROW_FLOOR,
    )
    assert (cv6e.native_ocp_fp8, cv6e.peak_flops("fp8"), cv5p.peak_flops("fp8")) == (
        False,
        _V6E_FP8_FLOPS,
        cv5p.bf16_flops_per_s,
    )
    assert (
        c7x.vmem_usable_bytes,
        cv6e.vmem_usable_bytes,
        cv4.vmem_usable_bytes,
    ) == (62 << 20, 126 << 20, 14 << 20)

    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")
    rep_v4 = advisor.advise(
        p397,
        advisor.WorkloadSpec(tp=8, batch=4, tiles=Tiles(nbd=8, nbe=4)),
        cv4,
    )
    v4_cross = next(s for s in rep_v4.scopes if s.scope == "whole_step")
    assert rep_v4.hbm_fits is False
    assert rep_v4.build_megakernel is False
    assert v4_cross.t_vmem_max == 0
    assert v4_cross.reason == "Weight buffers exceed VMEM"
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    pmix = advisor.model_profile_from_hf_config(_MIXTRAL_8X7B_CFG, name="mixtral-8x7b")
    assert pmix.num_moe_layers == _MIXTRAL_LAYERS
    assert pmix.moe_intermediate_size == _MIXTRAL_MOE_INTER
    rep_mix = advisor.advise(pmix, advisor.WorkloadSpec(tp=4, batch=16), cv5e)
    assert rep_mix.hbm_fits is False
    assert rep_mix.hbm_used_gb > _V5E_HBM_CAP_GB
    p_ffn = advisor.model_profile_from_hf_config(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "ffn_dim": 2048,
            "num_attention_heads": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    assert p_ffn.intermediate_size == _FFN_DIM_2048
    assert p_ffn.moe_intermediate_size == _FFN_DIM_2048

    pdsv3 = advisor.model_profile_from_hf_config(_DEEPSEEK_V3_CFG, name="deepseek-v3")
    assert pdsv3.head_dim == _DSV3_HEAD_DIM
    assert pdsv3.shared_expert_intermediate_size == _FFN_DIM_2048
    assert pdsv3.num_moe_layers == _DSV3_MOE_LAYERS
    assert pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3, num_moe_layers=0, num_experts=0, top_k=0, intermediate_size=0
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    nh_loc = pdsv3_dense_only.num_attention_heads // 8
    q_tot = nh_loc * (
        pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.qk_rope_head_dim
    )
    q_dim = nh_loc * pdsv3_dense_only.head_dim
    mla_layer_params = (
        pdsv3_dense_only.q_lora_rank * (pdsv3_dense_only.hidden_size + q_tot)
        + pdsv3_dense_only.hidden_size
        * (pdsv3_dense_only.kv_lora_rank + pdsv3_dense_only.qk_rope_head_dim + q_dim)
        + pdsv3_dense_only.kv_lora_rank
        * nh_loc
        * (pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.head_dim)
        + 3 * pdsv3_dense_only.hidden_size
    )
    expected_mla_w_gb = round((pdsv3_dense_only.num_layers * mla_layer_params) / 1e9, 4)
    assert r_mla_only.scopes[0].w_step_gb == expected_mla_w_gb
    r_mla_pre = advisor.advise(
        pdsv3_dense_only,
        advisor.WorkloadSpec(phase="prefill", tp=8, batch=1, isl=4096, chunk_size=1),
        c7x,
    )
    assert r_mla_only.step_arithmetic_intensity > r_mla_pre.step_arithmetic_intensity
    r_ds1 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, batch=32, context=8192), c7x
    )
    r_ds2 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, kv_split=2, batch=32, context=8192), c7x
    )
    assert r_ds2.hbm_used_gb < r_ds1.hbm_used_gb
    assert next(s for s in r_ds1.scopes if s.scope == "whole_step").feasible is False
    r_ver = advisor.advise(
        pdsv3, advisor.WorkloadSpec(phase="verify", tp=8, batch=8, spec_tokens=3), c7x
    )
    r_mix = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="mixed", tp=8, batch=8, spec_tokens=3, chunk_size=128
        ),
        c7x,
    )
    assert r_ver.active_tokens == _VERIFY_ACTIVE_TOKENS
    assert r_mix.active_tokens == _MIXED_ACTIVE_TOKENS


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
    """Verify `asymptotic_decode_intensity`, `fit_vmem_budget_bytes`, `w_crit_gb`."""
    assert advisor.fit_vmem_budget_bytes() == 62 << 20
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == 62 << 20
    p08 = advisor.model_profile_from_hf_config(
        _QWEN35_08B_CFG, name="qwen35-0.8b", state_dtype="fp32"
    )
    r_b16 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=16, spec_tokens=3, tp=8),
        "tpu7x",
    )
    r_b128 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=128, spec_tokens=3, tp=8),
        "tpu7x",
    )
    assert r_b16.asymptotic_decode_intensity is not None
    assert r_b16.asymptotic_decode_intensity == r_b128.asymptotic_decode_intensity
    assert r_b16.asymptotic_decode_intensity < r_b16.ridge_dense_flop_per_byte
    assert r_b16.step_arithmetic_intensity < r_b16.asymptotic_decode_intensity

    r_alias = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    r_no_alias = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
        ),
        "tpu7x",
    )
    r_no_alias_disp = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    assert r_alias.vmem_budget_mib == _TPU7X_VMEM_MIB
    assert r_no_alias.vmem_budget_mib == advisor.fit_vmem_budget_bytes() / (1 << 20)
    cx_alias = next(s for s in r_alias.scopes if s.scope == "whole_step")
    cx_no_al = next(s for s in r_no_alias.scopes if s.scope == "whole_step")
    cx_disp = next(s for s in r_no_alias_disp.scopes if s.scope == "whole_step")
    assert cx_alias.n_splits == 1
    assert cx_no_al.n_splits > 1
    assert cx_no_al.w_crit_gb is not None
    assert cx_disp.w_crit_gb is not None
    assert cx_disp.w_crit_gb > cx_no_al.w_crit_gb
    assert cx_disp.step_bound_us > cx_no_al.step_bound_us


def test_all_four_fusion_scopes_reachable() -> None:
    """Prove `whole_step`, `single_layer`, `stage`, and `per_op` each win."""
    p08 = advisor.model_profile_from_hf_config(_QWEN35_08B_CFG, name="qwen35-0.8b")
    p397 = advisor.model_profile_from_hf_config(_QWEN35_397B_CFG, name="qwen35-397b")
    pllama = advisor.resolve_model_profile("llama-3.1-405b")

    # 1. `whole_step` (`whole_step`) wins on 397B TP8 decode and 0.8B prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    assert r_08_pre.recommended_scope == "whole_step"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"

    # 2. `single_layer` (`layer`) wins in the B=488 VMEM cliff band on tpu7x,
    # whereas v6e (TP16) and v5p Megacore (TP8) keep `whole_step` in 1 pass.
    r_cliff_7x = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        "tpu7x",
    )
    assert r_cliff_7x.recommended_scope == "single_layer"
    assert r_cliff_7x.recommended_moe_mode == "grouped_on_chip"
    r_cliff_v6e = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=16),
        "v6e",
    )
    assert r_cliff_v6e.recommended_scope == "whole_step"
    v5p_mc = advisor.resolve_chip_spec("v5p", megacore=True)
    r_cliff_v5p_mc = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        v5p_mc,
    )
    assert r_cliff_v5p_mc.recommended_scope == "whole_step"
    r_ep8_sat = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode", batch=600, tp=2, dp=4, ep=8, tiles=Tiles()
        ),
        "tpu7x",
    )
    # Saturated DP4 x TP2 decode with locked wide ring: 150 tok/core overflow
    # the wide whole-step ceiling while single_layer fits in 1 pass.
    assert r_ep8_sat.recommended_scope == "single_layer"
    assert r_ep8_sat.recommended_moe_mode == "grouped_on_chip"

    r_llama_pre = advisor.advise(
        pllama, advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8), "tpu7x"
    )
    assert r_llama_pre.recommended_scope == "per_op"
    assert r_llama_pre.recommended_moe_mode == "dense_mlp"
    assert r_llama_pre.build_megakernel is False

    # 3. `layer_group` wins on PP8 tail and segmented TP8 (`segment_layers=10`).
    r_stg = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.recommended_scope == "layer_group"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.recommended_scope == "layer_group"
    assert (
        next(s for s in r_seg.scopes if s.scope == "layer_group").ops_per_step
        == _STAGE_OPS_PER_STEP
    )
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


def test_vmem_budget_override_and_dma_buffers_triple_buffering() -> None:
    """Verify user-configurable `--vmem-budget-mib` and `--dma-buffers {2,3}`."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_2buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2), "tpu7x"
    )
    rep_3buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=3), "tpu7x"
    )
    rep_custom_vmem = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            tp=8, ep=8, batch=16, dma_buffers=2, vmem_budget_mib=_CUSTOM_VMEM_MIB
        ),
        "tpu7x",
    )
    cx_2 = next(s for s in rep_2buf.scopes if s.scope == "whole_step")
    cx_3 = next(s for s in rep_3buf.scopes if s.scope == "whole_step")
    cx_vmem = next(s for s in rep_custom_vmem.scopes if s.scope == "whole_step")
    assert rep_2buf.vmem_budget_mib == _TPU7X_VMEM_MIB
    assert rep_custom_vmem.vmem_budget_mib == _CUSTOM_VMEM_MIB
    assert cx_vmem.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.vmem_static_mib > cx_2.vmem_static_mib
    assert cx_3.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.step_bound_us >= cx_2.step_bound_us


@pytest.mark.parametrize(
    ("tp", "ep", "pcp", "accepted"),
    [(4, 8, 2, True), (4, 8, 1, False), (4, 16, 2, False)],
)
def test_expert_parallel_spans_context_parallel_ranks(
    tp: int, ep: int, pcp: int, *, accepted: bool
) -> None:
    """Verify `ep` may span `tp * dp * pcp` ranks and no wider."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp)
    if accepted:
        rep = advisor.advise(p397, wl, "tpu7x")
        assert rep.active_tokens == 4096 // pcp
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(p397, wl, "tpu7x")


def test_advise_from_model_spec_and_planner() -> None:
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    rep = advisor.advise_from_spec(ms, ex, chip="tpu7x", moe_mode="grouped_on_chip")
    assert rep.recommended_scope == "whole_step"
    rep_def = advisor.advise_from_spec(
        ms, ExecSpec(B=4, T=1, R=2, ctx=1024), chip="tpu7x", moe_mode="grouped_on_chip"
    )
    assert rep_def.recommended_scope == "whole_step"
    assert rep_def.vmem_budget_mib == _TPU7X_VMEM_MIB

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="grouped_on_chip"
    )
    cross = next(s for s in rep_np.scopes if s.scope == "whole_step")
    assert cross.feasible is False
    assert rep_np.recommended_scope == "single_layer"


def test_spec_bridge_jax_isolation_ast() -> None:
    """Verify spec_bridge.py isolates jax and planner imports inside ast.Try."""
    bridge_path = Path(advisor.__file__).with_name("spec_bridge.py")
    tree = ast.parse(bridge_path.read_text("utf-8"), filename=str(bridge_path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            mod = getattr(node, "module", "") or ""
            assert not any("jax" in n for n in names), f"Unguarded jax import: {names}"
            assert "jax" not in mod, f"Unguarded jax module import: {mod}"
            assert not any("planner" in n for n in names), (
                f"Unguarded planner import: {names}"
            )
    try_nodes = [n for n in tree.body if isinstance(n, ast.Try)]
    assert any(
        any(
            (
                isinstance(h.type, ast.Tuple)
                and any(
                    getattr(e, "id", "") in ("ImportError", "RuntimeError")
                    for e in h.type.elts
                )
            )
            or getattr(h.type, "id", "") in ("ImportError", "RuntimeError")
            for h in t.handlers
        )
        for t in try_nodes
    ), "spec_bridge.py must have ast.Try catching (ImportError, RuntimeError)"


def test_advise_from_spec_raises_when_planner_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify advise_from_spec raises MegakernelConfigError when planner is missing."""
    monkeypatch.setattr(spec_bridge, "planner", None)
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    with pytest.raises(MegakernelConfigError, match="Megakernel planner requires JAX"):
        advisor.advise_from_spec(ms, ex)


def test_qwen35_decode_preset_whole_step_admits_shipped_verify_rows() -> None:
    """Verify the whole_step token tile fits 64 verify rows and the shipped split."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rows_per_seq = dep.decode.spec_tokens + 1
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=_QWEN35_VERIFY_ROWS_MIN // rows_per_seq,
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    rows_cap = rows_per_seq * (whole.t_vmem_max // rows_per_seq)
    assert rows_cap >= _QWEN35_VERIFY_ROWS_MIN
    assert abs(rows_cap - _QWEN35_SPLIT_VERIFY_ROWS) <= rows_per_seq


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_kimi_k3_decode_preset_low_batch_prefers_whole_step(batch: int) -> None:
    """Verify Kimi-K3 decode at batch 1..8 on the TP32 mesh is one call."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.decode,
            phase="decode",
            batch=batch,
        ),
        "tpu7x",
    )
    assert rep.recommended_scope == "whole_step"


@pytest.mark.parametrize("window", [2048, 8192])
def test_kimi_k3_prefill_preset_evaluates_layer_group(window: int) -> None:
    """Verify the shipped 13-call layer grouping is feasible and evaluated."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.prefill,
            phase="prefill",
            batch=1,
            isl=window,
            chunk_size=window,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    group = next(s for s in rep.scopes if s.scope == "layer_group")
    assert group.feasible
    assert group.ops_per_step == _K3_PREFILL_CALLS
    assert rep.active_tokens == window // dep.prefill.pcp


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.advise(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


def test_whole_step_tiles_overflow_and_reads_weights_once() -> None:
    """Verify whole_step tiles tokens past the ceiling instead of re-passing."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=(_QWEN35_SPLIT_VERIFY_ROWS // (dep.decode.spec_tokens + 1)) + 4,
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    tiled = next(s for s in rep.scopes if s.scope == "whole_step")
    one_l = next(s for s in rep.scopes if s.scope == "single_layer")
    assert tiled.n_splits > 1
    assert tiled.feasible
    assert tiled.w_step_gb == one_l.w_step_gb
    assert tiled.step_bound_us < 1.5 * one_l.step_bound_us


def test_fused_scopes_price_the_recommended_moe_mode() -> None:
    """Verify a full row tile per EP rank prices on-chip grouping either way."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    steps = []
    for mode in ("masked_on_chip", "grouped_on_chip"):
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8, moe_mode=mode),
            "tpu7x",
        )
        steps.append(next(s for s in rep.scopes if s.scope == "whole_step"))
        assert rep.recommended_moe_mode == "grouped_on_chip"
    assert steps[0].moe_flop_factor == steps[1].moe_flop_factor


_GROUPED_PHI_CEILING = 5.0
_UNGROUPED_PHI_FLOOR = 10.0
_PREFILL_ISL_4096 = 4096


def test_intra_host_dp_grouped_moe_consistency() -> None:
    """Verify intra-host DPExchange groups MoE while multi-host DCN stays ungrouped."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep_grp = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_grp = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    assert ws_grp.moe_flop_factor < _GROUPED_PHI_CEILING
    assert rep_grp.recommended_scope == "whole_step"
    assert rep_grp.recommended_moe_mode == "grouped_on_chip"
    assert float(rep_grp.metrics["achievable_gain_pct"]["value"]) > 0.0

    rep_msk = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        "tpu7x",
    )
    ws_msk = next(s for s in rep_msk.scopes if s.scope == "whole_step")
    assert ws_msk.moe_flop_factor > _UNGROUPED_PHI_FLOOR
    assert rep_msk.recommended_moe_mode != "grouped_on_chip"

    rep_dcn = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_dcn = next(s for s in rep_dcn.scopes if s.scope == "whole_step")
    assert ws_dcn.moe_flop_factor > _UNGROUPED_PHI_FLOOR
    assert rep_dcn.recommended_moe_mode != "grouped_on_chip"


def test_unchunked_pp_prefill_preserves_window() -> None:
    """Verify unchunked B=1 prefill under PP>1 preserves its full stage window."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep_single = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=_PREFILL_ISL_4096,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
        "tpu7x",
    )
    assert rep_single.active_tokens == _PREFILL_ISL_4096
    assert rep_single.recommended_scope == "per_op"
    assert rep_single.recommended_moe_mode == "grouped_in_hbm"
    assert float(rep_single.metrics["compiler_proof_gain_pct"]["value"]) == 0.0

    rep_microbatched = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=8,
            isl=_PREFILL_ISL_4096,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=8,
        ),
        "tpu7x",
    )
    assert rep_microbatched.active_tokens == _PREFILL_ISL_4096
