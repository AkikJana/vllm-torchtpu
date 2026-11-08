// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Client-side ES2022 roofline and on-chip VMEM physics evaluator.
 */

(function(global) {
  function createAdvisorEngine(seedData) {
    const D = seedData || {};
    const DATA = D;
  const PROF = DATA.model_profile || {}, CHIP = DATA.chip_spec || {};
  const REGIMES = DATA.regimes || [];
  const SUPPORTED_FAM = new Set([
    'gqa', 'mla', 'dsa', 'swa', 'gdn', 'kda', 'dense_mlp', 'moe', 'mtp', 'ple'
  ]);
  const DTYPE_BYTES = {
    fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, int8: 1.0,
    nvfp4: 0.5625, mxfp4: 0.53125, fp4: 0.5625
  };
  function roundEven(x) {
    const f = Math.floor(x), d = x - f;
    if (Math.abs(d - 0.5) < 1e-12) return (f % 2 === 0) ? f : f + 1;
    return Math.round(x);
  }
  function gcd(a, b) {
    let x = Math.abs(Math.trunc(a)), y = Math.abs(Math.trunc(b));
    while (y !== 0) { const t = y; y = x % y; x = t; }
    return x || 1;
  }
  function peakFlops(dtype, chipObj) {
    const c = chipObj || CHIP;
    const dt = String(dtype || 'fp8').toLowerCase();
    const bf16 = c.bf16_flops_per_s || 0.0;
    if (dt === 'fp8' || dt === 'nvfp4' || dt === 'mxfp4' || dt === 'fp4') {
      return c.fp8_flops_per_s || (c.native_ocp_fp8 !== false ? 2.0 * bf16 : bf16);
    }
    if (dt === 'int8') return c.int8_flops_per_s || (2.0 * bf16);
    if (dt === 'fp32') return 0.5 * bf16;
    return bf16;
  }
  function rowFloor(dtype, chipObj) {
    const c = chipObj || CHIP;
    const dt = String(dtype || 'fp8').toLowerCase();
    const map = c.matmul_row_tile_by_dtype || {};
    if (map[dt]) return map[dt];
    const hasFp8 = c.native_ocp_fp8 !== false;
    return (hasFp8 && (dt === 'fp8' || dt.includes('fp4'))) ? 32 : 16;
  }
  function resolveDcnBw(chipObj, wl) {
    const c = chipObj || CHIP;
    if (wl && wl.dcnBwBytesPerS) return wl.dcnBwBytesPerS;
    const dcnBw = c.dcn_bandwidth_bytes_per_s ||
      (c.dcn_bw_gbps ? (c.dcn_bw_gbps / 8.0) * 1e9 : 12.5e9);
    return dcnBw;
  }
  function allReduceCost(msgBytes, effTp, collLatS, chipObj, wl) {
    const c = chipObj || CHIP;
    if (effTp <= 1 || msgBytes <= 0) {
      return { xferS: 0, latS: 0, hops: 0, intraHops: 0, interLatS: 0 };
    }
    const iciBw = c.ici_egress_bw_bytes_per_s;
    const hops = roundEven(Math.log2(effTp));
    if (effTp <= 8) {
      const vol = 2.0 * (effTp - 1) / effTp * msgBytes;
      return {
        xferS: vol / iciBw, latS: hops * collLatS,
        hops, intraHops: hops, interLatS: 0
      };
    }
    const numHosts = Math.ceil(effTp / 8), intraVol = 1.75 * msgBytes;
    const interVol = 2.0 * (numHosts - 1) / numHosts * (msgBytes / 8.0);
    const dcnBw = resolveDcnBw(c, wl);
    const interLatS = roundEven(Math.log2(numHosts)) * 5.0e-6;
    const latS = 3 * collLatS + interLatS;
    return {
      xferS: intraVol / iciBw + interVol / Math.max(1.0, dcnBw),
      latS, hops, intraHops: 3, interLatS
    };
  }
  function computePerTokenVmem(wl, profObj) {
    const p = profObj || PROF;
    const H = p.hidden_size, tpRaw = Math.max(1, wl.tp || 1);
    const dcp = Math.max(1, wl.dcp || 1);
    const effTp = tpRaw === 1 ? Math.max(1, tpRaw, dcp) : tpRaw;
    const ep = Math.max(1, wl.ep || 1);
    const kvSplit = Math.max(1, wl.kvSplit || wl.kv_split || 1, dcp);
    const vmemAlias = wl.vmemAlias !== undefined
      ? wl.vmemAlias : (wl.vmem_alias !== undefined ? wl.vmem_alias : true);
    const moeMode = wl.moeMode || wl.moe_mode || 'masked_on_chip';
    const spine = 6 * H + (effTp > 1 ? 6 * H : 0);
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const hd = p.head_dim;
    let attn = 0, lin = 0, moe = 0;
    if ((p.kv_lora_rank || 0) > 0) {
      const qlr = p.q_lora_rank || 0, qkr = p.qk_rope_head_dim || 0;
      attn = 2 * (qlr + p.kv_lora_rank + qkr) + 6 * (nhLoc * hd);
    } else {
      const tpPerKv = Math.max(1, Math.floor(effTp / kvSplit));
      const nkvLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpPerKv));
      attn = 2 * (2 * nhLoc + 2 * nkvLoc) * hd + 4 * nhLoc * hd;
    }
    if ((p.num_linear_layers || 0) > 0 && (p.linear_value_heads || 0) > 0) {
      const khL = Math.max(1, Math.ceil((p.linear_key_heads || 0) / effTp));
      const vhL = Math.max(1, Math.ceil(p.linear_value_heads / effTp));
      const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
      lin = 2 * (2 * khL * kd + 2 * vhL * vd) + 4 * vhL * vd;
    }
    const nExp = p.num_experts || 0;
    if ((p.num_moe_layers || 0) > 0 && nExp > 0) {
      const topK = p.top_k || 1, router = 4 * nExp + 8 * topK;
      const interDiv = Math.max(1, Math.floor(effTp / gcd(ep, effTp)));
      const interLoc = Math.ceil((p.moe_intermediate_size || 0) / interDiv);
      const shI = p.shared_expert_intermediate_size || 0;
      const sharedLoc = shI > 0 ? Math.ceil(shI / effTp) : 0;
      const expH = p.routed_expert_hidden_size || H;
      const effK = ep > 1 ? Math.min(topK, 2 * ep) : 1;
      const grp = 8 * expH +
        Math.ceil(effK * (6 * expH + 8 * interLoc) / ep) + 8 * sharedLoc;
      const isGrp = moeMode === 'grouped_on_chip';
      const tile = isGrp ? grp : (16 * expH + 8 * (interLoc + sharedLoc));
      moe = vmemAlias ? Math.max(router + 2 * expH, tile) : (router + tile);
    } else {
      moe = 6 * H + 8 * Math.ceil((p.intermediate_size || 0) / effTp);
    }
    const sumOp = attn + lin + moe;
    const opBytes = vmemAlias ? Math.max(attn, lin, moe) : sumOp;
    return {
      spineBytes: spine, attnBytes: attn, linBytes: lin, moeBytes: moe,
      opBytes, unaliasedOpBytes: sumOp, totalPerTokenBytes: spine + opBytes
    };
  }
  function expertsHit(tokensEp, topK, numExperts, expertsLocal) {
    if (numExperts <= 0 || expertsLocal <= 0) return 0.0;
    const miss = Math.pow(1.0 - Math.min(1.0, topK / numExperts), tokensEp);
    return Math.max(1.0, expertsLocal * (1.0 - miss));
  }
  function selectAdaptiveRing(
    vmemBudget, fixedNonRing, alpha, activeTokens, hasMoe, wl, dBank, eBank
  ) {
    if (wl.tiles) {
      const nbd = wl.tiles.nbd, nbe = hasMoe ? wl.tiles.nbe : 0;
      const rb = nbd * wl.tiles.bank + nbe * wl.tiles.ebank;
      return { nbd, nbe, ringB: rb, minRingB: rb };
    }
    const rp = wl.ringPlan || wl.ring_plan;
    if (rp) {
      const nbd = rp.dense_slots !== undefined ? rp.dense_slots : 2;
      const nbe = hasMoe ? (rp.expert_slots !== undefined ? rp.expert_slots : 2) : 0;
      const db = rp.dense_slot_bytes || 2097152;
      const eb = rp.expert_slot_bytes || 1048576;
      const rb = nbd * db + nbe * eb;
      return { nbd, nbe, ringB: rb, minRingB: rb };
    }
    const bufSlots = wl.dmaBuffers || wl.dma_buffers || 2;
    const minNbd = bufSlots, extra = minNbd - 2, maxNbd = 14 + extra;
    const minNbe = hasMoe ? minNbd : 0, maxNbe = hasMoe ? (10 + extra) : 0;
    const minRingB = minNbd * dBank + minNbe * eBank;
    const avail = vmemBudget - fixedNonRing - alpha * activeTokens;
    if (avail < minRingB) {
      return { nbd: minNbd, nbe: minNbe, ringB: minRingB, minRingB };
    }
    let bestRing = -1, bestBal = -1, bestNbd = minNbd, bestNbe = minNbe;
    for (let nbd = minNbd; nbd <= maxNbd; nbd++) {
      for (let nbe = minNbe; nbe <= maxNbe; nbe++) {
        const rb = nbd * dBank + nbe * eBank;
        if (rb <= avail) {
          const bal = hasMoe
            ? Math.min(nbd - minNbd, nbe - minNbe) : (nbd - minNbd);
          if (rb > bestRing || (rb === bestRing && bal > bestBal) ||
              (rb === bestRing && bal === bestBal && nbd > bestNbd)) {
            bestRing = rb; bestBal = bal; bestNbd = nbd; bestNbe = nbe;
          }
        }
      }
    }
    return { nbd: bestNbd, nbe: bestNbe, ringB: bestRing, minRingB };
  }
  function computeHbmUsedBytes(p, wl) {
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const bwKv = dtB('kv_dtype');
    const bwSt = DTYPE_BYTES[String(p.state_dtype || 'bf16').toLowerCase()] || 2.0;
    const L = p.num_layers, H = p.hidden_size;
    const nExp = p.num_experts || 0;
    const tp = Math.max(1, wl.tp || 1), pp = Math.max(1, wl.pp || 1);
    const dp = Math.max(1, wl.dp || 1), ep = Math.max(1, wl.ep || 1);
    const dcp = Math.max(1, wl.dcp || 1);
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    const kvSplit = Math.max(wl.kvSplit || wl.kv_split || 1, dcp);
    const batchLocal = Math.ceil((wl.batch || 16) / dp);
    const layersLocal = Math.ceil(L / pp);
    const islVal = (wl.isl !== undefined && wl.isl !== null && wl.isl > 0)
      ? wl.isl : 4096;
    const oslVal = (wl.osl !== undefined && wl.osl !== null) ? wl.osl : 512;
    const effCtx = (wl.context && wl.context > 0)
      ? wl.context : (islVal + (wl.phase === 'prefill' ? 0 : Math.floor(oslVal / 2)));
    const lAttn = Math.min(
      layersLocal, Math.ceil((p.num_attention_layers || 0) * layersLocal / L)
    );
    const lMoe = Math.min(
      layersLocal, Math.ceil((p.num_moe_layers || 0) * layersLocal / L)
    );
    const lLin = Math.min(
      layersLocal - lAttn, Math.ceil((p.num_linear_layers || 0) * layersLocal / L)
    );
    const hd = p.head_dim, kvLora = p.kv_lora_rank || 0;
    const qLora = p.q_lora_rank || 0, qkRope = p.qk_rope_head_dim || 0;
    const qkNope = p.qk_nope_head_dim || hd;
    const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const tpKvDiv = Math.max(1, Math.floor(effTp / kvSplit));
    const kvhLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpKvDiv));
    const qDim = nhLoc * hd, qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) +
      H * (kvLora + qkRope + qDim);
    const attnGate = (p.has_attn_output_gate && !kvLora) ? (H * qDim) : 0;
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhLoc * hd) + attnGate);
    const khLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_key_heads || 1) / effTp)) : 0;
    const vhLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_value_heads || 1) / effTp)) : 0;
    const zDim = vhLoc * vd, qkvLin = 2 * khLoc * kd + vhLoc * vd;
    const epTp = gcd(ep, effTp);
    const linParams = lLin > 0 ? H * (qkvLin + 2 * zDim) : 0;
    const expertsLoc = lMoe > 0 ? Math.floor(nExp / ep) : 0;
    const denseFfnParams = 3 * H * Math.max(
      1, Math.ceil((p.intermediate_size || 0) / effTp)
    );
    const sharedExpParams = 3 * H * Math.ceil(
      (p.shared_expert_intermediate_size || 0) / effTp
    );
    const iExpLoc = lMoe > 0 ? Math.ceil(
      (p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp))
    ) : 0;
    const routedExpParams = 3 * (p.routed_expert_hidden_size || H) * iExpLoc;
    const baseDenseParams = lAttn * attnParams + lLin * linParams +
      (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * 2.0;
    const expBytes = routedExpParams * bwE;
    const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    const kvReadBytes = lAttn * batchLocal * effCtx * ((kvRawTok * bwKv) / kvSplit);
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    return wStatic + lMoe * expertsLoc * expBytes + kvReadBytes + 0.5 * recRwBytes;
  }
  function validateWorkload(a, b, c) {
    const p = (c !== undefined) ? (a || PROF) : PROF;
    const ch = (c !== undefined) ? (b || CHIP) : CHIP;
    const wl = ((c !== undefined) ? c : a) || {};
    const batch = wl.batch !== undefined ? wl.batch : 16;
    const tp = wl.tp !== undefined ? wl.tp : 1;
    const pp = wl.pp !== undefined ? wl.pp : 1;
    const dp = wl.dp !== undefined ? wl.dp : 1;
    const ep = wl.ep !== undefined ? wl.ep : 1;
    const kvSplit = wl.kvSplit ?? wl.kv_split ?? 1;
    const dcp = wl.dcp !== undefined ? wl.dcp : 1;
    const pcp = wl.pcp !== undefined ? wl.pcp : 1;
    const isl = wl.isl !== undefined ? wl.isl : 4096;
    const osl = wl.osl !== undefined ? wl.osl : 512;
    const ctxV = wl.context !== undefined ? wl.context : 0;
    const specV = wl.specTokens ?? wl.spec_tokens ?? 0;
    const chunkV = wl.chunkSize ?? wl.chunk_size ?? 0;
    const mbV = wl.microbatches !== undefined ? wl.microbatches : 0;
    const segV = wl.segmentLayers ?? wl.segment_layers ?? 0;
    const opOv = wl.opOverheadUs ?? wl.op_overhead_us ?? 0.0;
    if (Math.min(
          p.num_layers, p.hidden_size, p.num_attention_heads, p.head_dim,
          batch, tp, pp, dp, ep, kvSplit, dcp, pcp, isl
        ) < 1 ||
        Math.min(osl, ctxV, specV, chunkV, mbV, segV) < 0 || opOv < 0.0) {
      return ['invalid_dimensions'];
    }
    const reasons = [];
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    if ((effTp & (effTp - 1)) !== 0) reasons.push('tp_not_power_of_two');
    if (p.num_attention_heads % effTp !== 0) {
      reasons.push('attn_heads_not_divisible_by_tp');
    }
    const kvTp = Math.max(1, Math.floor(effTp / Math.max(1, kvSplit)));
    if (effTp % kvSplit !== 0 ||
        (p.num_kv_heads % kvTp !== 0 && kvTp % p.num_kv_heads !== 0)) {
      reasons.push('kv_heads_not_divisible_by_tp');
    }
    if (effTp % dcp !== 0) reasons.push('dcp_not_dividing_tp');
    if ((p.num_experts || 0) > 0 &&
        (p.num_experts % ep !== 0 || !(1 <= p.top_k && p.top_k <= p.num_experts))) {
      reasons.push('experts_not_divisible_by_ep');
    }
    if ((effTp * dp * pcp) % ep !== 0) reasons.push('ep_exceeds_slice_ranks');
    if (pp > p.num_layers) reasons.push('pp_exceeds_layers');
    const hbmUsed = computeHbmUsedBytes(p, wl);
    const capFrac = wl.hbmUtilizationCap ?? wl.hbm_utilization_cap ?? 0.80;
    if (hbmUsed > capFrac * ch.hbm_capacity_bytes) reasons.push('hbm_pool_overflow');
    return reasons;
  }
  function resolveDispatchBox(wl) {
    const dr = wl.dispatchRegime || wl.dispatch_regime || 'auto';
    const rawOp = wl.opOverheadUs ?? wl.op_overhead_us ?? wl.launchGapUs ?? 0.0;
    if (dr === 'zero_floor') return { lo: 0.0, hi: 0.0 };
    if (dr === 'compiled_xla') {
      if (wl.calibrated && wl.launchGapUs !== undefined) {
        const lo = wl.launchGapUs;
        return { lo, hi: Math.max(lo, wl.launchGapHiUs ?? lo) };
      }
      return { lo: 1.0, hi: 1.5 };
    }
    if (dr === 'unfused_eager') {
      const lo = rawOp > 0.0 ? rawOp : 6.0;
      return { lo, hi: Math.max(lo, 8.0) };
    }
    if (wl.calibrated && wl.launchGapUs !== undefined) {
      const lo = wl.launchGapUs;
      return { lo, hi: Math.max(lo, wl.launchGapHiUs ?? lo) };
    }
    return { lo: rawOp, hi: Math.max(rawOp, wl.launchGapHiUs ?? 8.0) };
  }
  function evaluateWorkload(wl, profOverride, chipOverride) {
    const p = profOverride || PROF, ch = chipOverride || CHIP;
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const bwKv = dtB('kv_dtype');
    const bwSt = DTYPE_BYTES[String(p.state_dtype || 'bf16').toLowerCase()] || 2.0;
    const L = p.num_layers, H = p.hidden_size;
    const nExp = p.num_experts || 0, topK = p.top_k || 0;
    const tp = Math.max(1, wl.tp || 1), pp = Math.max(1, wl.pp || 1);
    const dp = Math.max(1, wl.dp || 1), ep = Math.max(1, wl.ep || 1);
    const dcp = Math.max(1, wl.dcp || 1), pcp = Math.max(1, wl.pcp || 1);
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    const kvSplit = Math.max(wl.kvSplit || wl.kv_split || 1, dcp);
    const isPrefOrMix = wl.phase === 'prefill' || wl.phase === 'mixed';
    const effPcp = isPrefOrMix ? pcp : 1;
    const batchVal = wl.batch || 16;
    const batchLocal = Math.ceil(batchVal / dp), layersLocal = Math.ceil(L / pp);
    const islVal = (wl.isl !== undefined && wl.isl !== null && wl.isl > 0)
      ? wl.isl : 4096;
    const oslVal = (wl.osl !== undefined && wl.osl !== null) ? wl.osl : 512;
    const fallbackCtx = islVal +
      (wl.phase === 'prefill' ? 0 : Math.floor(oslVal / 2));
    const effCtx = (wl.context && wl.context > 0) ? wl.context : fallbackCtx;
    const pfxHit = wl.prefixHitRate ?? wl.prefix_hit_rate ?? 0.0;
    const uncachedIsl = Math.max(1, roundEven(islVal * (1.0 - pfxHit)));
    const cSize = wl.chunkSize ?? wl.chunk_size ?? 0;
    const chunkRaw = cSize || (wl.chunkSize || uncachedIsl);
    const chunkTokens = Math.ceil(chunkRaw / effPcp);
    const rawSpec = wl.specTokens ?? wl.spec_tokens;
    const specTok = (rawSpec !== undefined && rawSpec !== null)
      ? rawSpec : Math.max(0, (wl.seqTokens || 1) - 1);
    const verifyTok = wl.phase === 'verify'
      ? (Math.max(1, specTok) + 1) : (specTok > 0 ? specTok + 1 : 1);
    const mbVal = wl.microbatches || 0;
    const mbDenom = Math.max(1, (mbVal || 1) * effPcp);
    const prefillTok = cSize
      ? Math.ceil(cSize / effPcp)
      : Math.ceil((batchLocal * uncachedIsl) / mbDenom);
    const decodeTok = batchLocal * (wl.phase === 'decode' ? 1 : verifyTok);
    const activeTokens = wl.phase === 'prefill'
      ? prefillTok : (decodeTok + (wl.phase === 'mixed' ? chunkTokens : 0));
    const lAttn = Math.min(
      layersLocal, Math.ceil((p.num_attention_layers || 0) * layersLocal / L)
    );
    const lMoe = Math.min(
      layersLocal, Math.ceil((p.num_moe_layers || 0) * layersLocal / L)
    );
    const lLin = Math.min(
      layersLocal - lAttn,
      Math.ceil((p.num_linear_layers || 0) * layersLocal / L)
    );
    const tokResBytes = activeTokens * H * 2.0;
    const hd = p.head_dim, kvLora = p.kv_lora_rank || 0;
    const qLora = p.q_lora_rank || 0, qkRope = p.qk_rope_head_dim || 0;
    const qkNope = p.qk_nope_head_dim || hd;
    const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const tpKvDiv = Math.max(1, Math.floor(effTp / kvSplit));
    const kvhLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpKvDiv));
    const qDim = nhLoc * hd, qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) +
      H * (kvLora + qkRope + qDim);
    const attnGate = (p.has_attn_output_gate && !kvLora) ? (H * qDim) : 0;
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhLoc * hd) + attnGate);
    const khLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_key_heads || 1) / effTp)) : 0;
    const vhLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_value_heads || 1) / effTp)) : 0;
    const zDim = vhLoc * vd, qkvLin = 2 * khLoc * kd + vhLoc * vd;
    const epTp = gcd(ep, effTp), epDp = Math.max(1, Math.floor(ep / epTp));
    const linParams = lLin > 0 ? H * (qkvLin + 2 * zDim) : 0;
    const expertsLoc = lMoe > 0 ? Math.floor(nExp / ep) : 0;
    const denseFfnParams = 3 * H * Math.max(
      1, Math.ceil((p.intermediate_size || 0) / effTp)
    );
    const sharedExpParams = 3 * H * Math.ceil(
      (p.shared_expert_intermediate_size || 0) / effTp
    );
    const iExpLoc = lMoe > 0 ? Math.ceil(
      (p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp))
    ) : 0;
    const routedExpParams = 3 * (p.routed_expert_hidden_size || H) * iExpLoc;
    const baseDenseParams = lAttn * attnParams + lLin * linParams +
      (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * 2.0;
    const expBytes = routedExpParams * bwE;
    const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    const kvBytesPerTok = (kvRawTok * bwKv) / kvSplit;
    const kvReadBytes = lAttn * batchLocal * effCtx * kvBytesPerTok;
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    const vmemUsable = ch.vmem_usable_bytes;
    const rawVmemMib = wl.vmemMib !== undefined
      ? wl.vmemMib : (wl.vmem_budget_mib !== undefined
      ? wl.vmem_budget_mib : undefined);
    const rpObj = wl.ringPlan || wl.ring_plan;
    const spillRes = (rpObj && rpObj.compiler_spill_reserve_bytes) || 0;
    const fitHead = Boolean(wl.vmemFitHeadroom || wl.vmem_fit_headroom);
    const baseVmemBytes = (rawVmemMib !== undefined && rawVmemMib !== null)
      ? Math.trunc(rawVmemMib * 1048576)
      : (wl.tiles ? (wl.tiles.vmem_mib * 1048576) : vmemUsable);
    const vmemBudget = Math.min(
      vmemUsable, Math.max(0, baseVmemBytes - (fitHead ? spillRes : 0))
    );
    const ffnChoose = lMoe > 0 ? sharedExpParams : denseFfnParams;
    const layerDenseBytes = Math.trunc(
      (Math.max(attnParams, linParams) + ffnChoose) * bwW
    );
    const tPrefill = wl.phase === 'prefill'
      ? activeTokens : (wl.phase === 'mixed' ? chunkTokens : 0);
    const dMla = (activeTokens - tPrefill) * (2 * kvLora + qkRope) +
      tPrefill * (qkNope + qkRope + hd);
    const attnCtx = (wl.phase === 'prefill' && tPrefill > 1)
      ? (Math.max(0, effCtx - tPrefill) + (tPrefill + 1) / 2.0) : effCtx;
    const flopsKv = 2.0 * lAttn * attnCtx *
      (kvLora ? nhLoc * dMla : 2.0 * activeTokens * qDim);
    const routerFlops = lMoe * H * nExp;
    const flopsDense = 2.0 * activeTokens * (baseDenseParams + routerFlops) + flopsKv;
    const flopsMoe = expertsLoc > 0
      ? (2.0 * activeTokens * epDp * lMoe * topK *
         routedExpParams * expertsLoc / nExp) : 0.0;
    const bufSlots = wl.dmaBuffers || wl.dma_buffers || 2;
    const dBank = (wl.tiles && wl.tiles.bank)
      ? wl.tiles.bank : ((rpObj && rpObj.dense_slot_bytes)
      ? rpObj.dense_slot_bytes : (wl.denseSlotBytes || 2097152));
    const eBank = (wl.tiles && wl.tiles.ebank)
      ? wl.tiles.ebank : ((rpObj && rpObj.expert_slot_bytes)
      ? rpObj.expert_slot_bytes : (wl.expertSlotBytes || 1048576));
    const slDenseStaging = Math.min(bufSlots * dBank, bufSlots * layerDenseBytes);
    const slExpertStaging = Math.min(bufSlots * eBank, Math.trunc(bufSlots * expBytes));
    const firstSlot = Math.min(
      dBank, Math.max(262144, Math.floor(layerDenseBytes / 4))
    );
    const gqaScratch = lAttn > 0
      ? (2 * 128 * (4 * Math.max(hd, kvLora)) + 2 * 128 * 128 * 4) : 0;
    const gdnScratch = lLin > 0 ? (4 * Math.min(vhLoc, 8) * kd * vd) : 0;
    const ffnScratch = lMoe > 0 ? (2 * 128 * iExpLoc) : (2 * 128 * 1024 * 4);
    const vmemParts = computePerTokenVmem(wl, p);
    const defCollUs = (ch.collective_latency_s || 1e-6) * 1e6;
    const bracketHi = ch.collective_latency_us_bracket
      ? ch.collective_latency_us_bracket[1] : defCollUs;
    const collUs = wl.collHopUs !== undefined ? wl.collHopUs : defCollUs;
    const collHiUs = wl.calibrated
      ? Math.max(collUs, wl.collHopHiUs ?? collUs)
      : Math.max(collUs, wl.collHopHiUs ?? bracketHi);
    const latColl = collUs * 1e-6, latCollHi = collHiUs * 1e-6;
    const arInfo = allReduceCost(tokResBytes, effTp, latColl, ch, wl);
    const numAr = (p.ar_slots_per_layer || 2) * layersLocal;
    const bwHbm = ch.hbm_bw_bytes_per_s;
    const peakW = peakFlops(p.weight_dtype, ch);
    const peakE = peakFlops(p.expert_dtype || p.weight_dtype, ch);
    const bf16Peak = ch.bf16_flops_per_s || peakW;
    const rFloor = rowFloor(lMoe > 0 ? p.expert_dtype : p.weight_dtype, ch);
    const iciEg = ch.ici_egress_bw_bytes_per_s;
    const iciLink = ch.ici_link_bw_bytes_per_s;
    const a2aTime = epDp > 1
      ? (2.0 * (epDp - 1) / epDp * Math.max(1, topK) * tokResBytes / iciEg) : 0.0;
    const ppStageTime = pp > 1
      ? (tokResBytes / iciLink + latColl) : 0.0;
    const pcpRingTime = (effPcp > 1 && lAttn > 0)
      ? ((effPcp - 1) * lAttn * chunkTokens * (kvBytesPerTok * kvSplit) / iciLink)
      : 0.0;
    const wlLg = wl.layerGroups || wl.layer_groups;
    const lgArr = (Array.isArray(wlLg) && wlLg.length > 0)
      ? wlLg : (p.layer_groups || []);
    const presetGroups = lgArr.length > 0
      ? Math.ceil(lgArr.length * layersLocal / L) : 0;
    const segLayers = wl.segmentLayers ?? wl.segment_layers ?? 0;
    const hasSeg = segLayers > 0 && segLayers < layersLocal;
    const segCount = hasSeg ? Math.ceil(layersLocal / segLayers) : 1;
    const fams = p.layer_families || [];
    const hasUnmodeled = fams.some(f => !SUPPORTED_FAM.has(f));
    const noWholeStep = pp > 1 || segCount > 1 || lgArr.length > 1 ||
      L === 1 || Boolean(wl.planError) || hasUnmodeled;
    const hbmUsed = wStatic + lMoe * expertsLoc * expBytes +
      kvReadBytes + 0.5 * recRwBytes;
    const hbmCapBytes = ch.hbm_capacity_bytes;
    const hbmUtilCap = wl.hbmUtilizationCap ?? wl.hbm_utilization_cap ?? 0.80;
    const hbmPoolCapBytes = hbmUtilCap * hbmCapBytes;
    const hbmFits = hbmUsed <= hbmCapBytes;
    const invalidReasons = validateWorkload(p, ch, wl);
    const hbmPoolOverflow = invalidReasons.includes('hbm_pool_overflow');
    const vmemAlias = wl.vmemAlias !== undefined
      ? wl.vmemAlias : (wl.vmem_alias !== undefined ? wl.vmem_alias : true);
    const moeMode = wl.moeMode || wl.moe_mode || 'masked_on_chip';
    const lgBox = resolveDispatchBox(wl);
    const etaDense = wl.etaDense ??
      (wl.efficiency && wl.efficiency.eta_dense) ?? ch.eta_dense ?? 0.72;
    const etaMoeGrp = wl.etaMoeGrouped ??
      (wl.efficiency && wl.efficiency.eta_moe_grouped) ??
      ch.eta_moe_grouped ?? 0.50;
    const etaHbm = wl.etaHbm ??
      (wl.efficiency && wl.efficiency.eta_hbm) ?? ch.eta_hbm ?? 0.82;
    const etaIci = wl.etaIci ??
      (wl.efficiency && wl.efficiency.eta_ici) ?? ch.eta_ici ?? 0.75;

    const scopes = [];
    for (const sc of ['per_op', 'single_layer', 'layer_group', 'whole_step']) {
      let feasible = true, reason = 'ok';
      if (sc === 'layer_group' && presetGroups === 0 && pp === 1 && segCount === 1) {
        feasible = false;
        reason = 'No layer grouping: needs a model preset, pp > 1, or segment_layers';
      } else if (sc === 'whole_step' && noWholeStep) {
        feasible = false;
        reason = wl.planError ||
          'Bounded by layer groups, single layers, or unmodeled layer family';
      }
      let staticBytes = 0, actBytes = 0, nOps = 1, tMax = 1, nSplits = 1;
      let denseStage = 0, expertStage = 0, opScratch = 0, padBytes = 0;
      let selNbd = 0, selNbe = 0;
      if (sc === 'per_op') {
        denseStage = Math.max(1048576, slDenseStaging);
        staticBytes = denseStage;
        const perOpMax = Math.floor((vmemBudget - staticBytes) / (4 * H));
        tMax = Math.max(activeTokens, perOpMax);
        nSplits = 1; nOps = 2 * layersLocal;
        actBytes = 4.0 * layersLocal * tokResBytes;
      } else {
        const isSl = sc === 'single_layer';
        const isSlOrGrp = isSl || (sc === 'layer_group' && presetGroups > 0);
        const mul = sc === 'layer_group' ? (presetGroups || segCount) : 1;
        const sumScratch = gqaScratch + gdnScratch + ffnScratch;
        opScratch = vmemAlias
          ? Math.max(gqaScratch, gdnScratch, ffnScratch) : sumScratch;
        const alpha = vmemParts.totalPerTokenBytes;
        let minStaticBytes = 0;
        if (isSlOrGrp) {
          selNbd = bufSlots;
          selNbe = lMoe > 0 ? bufSlots : 0;
          denseStage = slDenseStaging;
          expertStage = lMoe > 0 ? slExpertStaging : 0;
          padBytes = 0;
          staticBytes = denseStage + expertStage + opScratch;
          minStaticBytes = staticBytes;
        } else {
          padBytes = 2097152;
          const ringSel = selectAdaptiveRing(
            vmemBudget, opScratch + padBytes, alpha, activeTokens,
            lMoe > 0, wl, dBank, eBank
          );
          selNbd = ringSel.nbd;
          selNbe = ringSel.nbe;
          denseStage = selNbd * dBank;
          expertStage = selNbe * eBank;
          staticBytes = ringSel.ringB + opScratch + padBytes;
          minStaticBytes = ringSel.minRingB + opScratch + padBytes;
        }
        actBytes = 2.0 * (isSl ? layersLocal : mul) * tokResBytes;
        const subMul = (effTp * ep > 1 && (lAttn + lLin) > 0) ? 3 : 1;
        nOps = isSl ? layersLocal : (
          sc === 'layer_group' && presetGroups > 0 ? mul : mul * subMul
        );
        tMax = (minStaticBytes + alpha > vmemBudget)
          ? 0 : Math.max(1, Math.floor((vmemBudget - minStaticBytes) / alpha));
        nSplits = tMax === 0 ? activeTokens : Math.ceil(activeTokens / tMax);
        if (tMax === 0) { feasible = false; reason = 'Weight buffers exceed VMEM'; }
      }
      const isFusedMulti = sc === 'whole_step' || sc === 'layer_group';
      const tiled = isFusedMulti && nSplits > 1;
      const tileTokens = Math.ceil(activeTokens / Math.max(1, nSplits));
      const isGrpMode = moeMode === 'grouped_on_chip';
      const canGroupMoe = !isFusedMulti || epDp === 1 ||
        ((effTp * epDp <= 8) && isGrpMode);
      const groupedMoe = canGroupMoe && (
        sc === 'per_op' || isGrpMode || tileTokens * epDp >= rFloor
      );
      const moeTok = (groupedMoe && sc !== 'single_layer')
        ? activeTokens : tileTokens;
      let wStep = wStatic, phi = 1.0;
      if (expertsLoc > 0) {
        const tEp = moeTok * epDp;
        const eHit = expertsHit(tEp, topK, nExp, expertsLoc);
        const pairs = Math.max(1.0, tEp * topK * expertsLoc / nExp);
        const avgM = Math.max(1.0, pairs / eHit);
        const hit = tiled
          ? expertsHit(activeTokens * epDp, topK, nExp, expertsLoc) : eHit;
        wStep = wStatic + lMoe * hit * expBytes;
        phi = groupedMoe ? (Math.ceil(avgM / rFloor) * rFloor / avgM)
          : Math.max(1.0, (Math.floor((tEp + 15) / 16) * 16 * eHit) / pairs);
      }
      if (tiled) actBytes += 2.0 * layersLocal * tokResBytes;
      const xlaArFactor = (sc === 'per_op' && effTp > 1) ? 2 : 1;
      const xlaHbmCollBytes = (sc === 'per_op' && effTp > 1)
        ? 2.0 * numAr * tokResBytes : 0.0;
      const overlappedComm = isFusedMulti && nSplits === 1;
      const arAndPcpS = numAr * (
        arInfo.xferS + nSplits * xlaArFactor * arInfo.latS
      ) + xlaHbmCollBytes / bwHbm + pcpRingTime;
      const a2aSyncS = (lMoe > 0 && epDp > 1)
        ? lMoe * (a2aTime + 2 * nSplits * xlaArFactor * latColl) : 0.0;
      const dcnSyncS = numAr * nSplits * xlaArFactor * arInfo.interLatS;
      const syncStageS = overlappedComm
        ? (dcnSyncS + ppStageTime + a2aSyncS)
        : (isFusedMulti ? (arAndPcpS + ppStageTime + a2aSyncS) : 0.0);
      const commS = overlappedComm
        ? (arAndPcpS - dcnSyncS)
        : (isFusedMulti ? 0.0 : (arAndPcpS + ppStageTime + a2aSyncS));
      const effStateBytes = kvReadBytes + recRwBytes;
      let fillDrainS = (nOps - 1) * (firstSlot / bwHbm + 2.0 * latColl) + syncStageS;
      if (tiled) {
        fillDrainS += (nSplits - 1) * 2 * layersLocal *
          ((tileTokens * H * 2.0) / bwHbm + 2.0 * latColl);
      }
      const tMatmul = flopsDense / peakW + (phi * flopsMoe) / peakE;
      const tMatmulHi = (isPrefOrMix && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak +
           (phi * flopsMoe) / peakE)
        : tMatmul;
      const tHbm = ((tiled ? 1 : nSplits) * wStep + effStateBytes + actBytes) / bwHbm;
      const roofCoreS = isFusedMulti
        ? Math.max(tMatmul, tHbm, commS) : (Math.max(tMatmul, tHbm) + commS);
      const roofUs = (roofCoreS + fillDrainS) * 1e6;
      const baseAi = (flopsDense + flopsMoe) / Math.max(1.0, wStep + effStateBytes);
      const repassUs = (baseAi >= peakW / bwHbm && nSplits > 1 && !tiled)
        ? ((actBytes + (nSplits - 1) * wStep) / bwHbm * 1e6) : 0.0;
      const pureStepUs = roofUs + repassUs;
      const launchUs = nOps * lgBox.lo, stepUs = pureStepUs + launchUs;
      const etaMoe = (groupedMoe && lMoe > 0) ? etaMoeGrp : etaDense;
      const tCompAch = (flopsDense / peakW) / Math.max(1e-3, etaDense) +
        ((phi * flopsMoe) / peakE) / Math.max(1e-3, etaMoe);
      const tHbmAch = tHbm / Math.max(1e-3, etaHbm);
      const tCommAch = commS / Math.max(1e-3, etaIci);
      const syncExtraS = isFusedMulti
        ? syncStageS * (1.0 / Math.max(1e-3, etaIci) - 1.0) : 0.0;
      const achCoreS = isFusedMulti
        ? (Math.max(tCompAch, tHbmAch, tCommAch) + syncExtraS)
        : (Math.max(tCompAch, tHbmAch) + tCommAch);
      const stepAchUs = (achCoreS + fillDrainS) * 1e6 + repassUs + launchUs;
      const arExtraS = (numAr * nSplits * xlaArFactor * arInfo.intraHops) *
        Math.max(0.0, latCollHi - latColl);
      const a2aCnt = (lMoe > 0 && epDp > 1) ? (2 * lMoe * nSplits * xlaArFactor) : 0;
      const a2aExtraS = a2aCnt * Math.max(0.0, latCollHi - latColl);
      const roofHiCoreS = overlappedComm
        ? (Math.max(tMatmulHi, tHbm, commS + arExtraS) + a2aExtraS)
        : (Math.max(tMatmulHi, tHbm) + commS + arExtraS + a2aExtraS);
      const roofHiUs = (roofHiCoreS + fillDrainS) * 1e6;
      const stepHiUs = Math.max(stepUs, roofHiUs + repassUs + nOps * lgBox.hi);
      const resTok = Math.min(activeTokens, Math.max(1, tMax));
      const aliasDiff = vmemParts.unaliasedOpBytes - vmemParts.opBytes;
      scopes.push({
        scope: sc, feasible, reason, ops_per_step: nOps, t_vmem_max: tMax,
        n_splits: nSplits, tile_tokens: tileTokens,
        selected_nbd: selNbd, selected_nbe: selNbe,
        rawStepUs: stepUs, pureStepUs,
        vmem_static_mib: +(staticBytes / 1048576).toFixed(3),
        dense_stage_mib: +(denseStage / 1048576).toFixed(2),
        expert_stage_mib: +(expertStage / 1048576).toFixed(2),
        op_scratch_mib: +(opScratch / 1048576).toFixed(2),
        pad_mib: +(padBytes / 1048576).toFixed(2),
        spine_dyn_mib: +((resTok * vmemParts.spineBytes) / 1048576).toFixed(2),
        op_dyn_mib: +((resTok * vmemParts.opBytes) / 1048576).toFixed(2),
        alias_saved_kib: +(aliasDiff / 1024).toFixed(1),
        moe_flop_factor: +phi.toFixed(3), w_step_gb: +(wStep / 1e9).toFixed(4),
        act_hbm_mb: +(actBytes / 1e6).toFixed(3),
        compute_us: +(tMatmul * 1e6).toFixed(2),
        hbm_us: +(tHbm * 1e6).toFixed(2), comm_us: +(commS * 1e6).toFixed(2),
        fill_drain_us: +(fillDrainS * 1e6).toFixed(2),
        repass_spill_us: +repassUs.toFixed(2),
        launch_us: +launchUs.toFixed(2), roof_floor_us: +roofUs.toFixed(2),
        step_bound_us: +stepUs.toFixed(2),
        step_achievable_us: +stepAchUs.toFixed(2),
        step_upper_us: +stepHiUs.toFixed(2)
      });
    }
    const feas = scopes.filter(s => s.feasible);
    if (!feas.length) feas.push(scopes[0]);
    feas.sort((a, b) => (a.rawStepUs - b.rawStepUs) ||
      (a.ops_per_step - b.ops_per_step));
    const winner = feas[0] || scopes[0], runnerUp = feas[1] || winner;
    const perOp = scopes[0];
    const compilerProofGainUs = perOp.pureStepUs - winner.pureStepUs;
    const dispatchBubbleGainUs = perOp.launch_us - winner.launch_us;
    const compilerProofGainPct = +(100.0 * Math.max(0.0, compilerProofGainUs) /
      Math.max(1e-6, perOp.pureStepUs)).toFixed(2);
    const achievableGainPct = +(100.0 * (perOp.step_achievable_us - winner.step_achievable_us) /
      Math.max(1e-6, perOp.step_achievable_us)).toFixed(2);
    const recScope = invalidReasons.length > 0 ? 'invalid_config' : winner.scope;
    const tokensPerChip = (activeTokens * effPcp * (ch.devices_per_chip || 1)) /
      Math.max(1, effTp * pp);
    const tpsChip = tokensPerChip / Math.max(1e-9, winner.step_bound_us * 1e-6);
    const winFusedMulti = winner.scope === 'whole_step' || winner.scope === 'layer_group';
    const winCanGroup = !winFusedMulti || epDp === 1 ||
      ((effTp * epDp <= 8) && (moeMode === 'grouped_on_chip'));
    const tPerEp = Math.ceil(activeTokens / Math.max(1, winner.n_splits)) * epDp;
    const moeModeRec = lMoe === 0 ? 'dense_mlp'
      : (winner.scope === 'per_op' ? 'grouped_in_hbm'
      : (winCanGroup && (moeMode === 'grouped_on_chip' || tPerEp >= rowFloor(p.expert_dtype, ch))
         ? 'grouped_on_chip' : 'masked_on_chip'));
    const marginPct = runnerUp && runnerUp.scope !== winner.scope
      ? ((runnerUp.step_bound_us - winner.step_bound_us) /
         Math.max(0.01, runnerUp.step_bound_us)) * 100.0 : 100.0;
    const wsScratch = vmemAlias
      ? Math.max(gqaScratch, gdnScratch, ffnScratch)
      : (gqaScratch + gdnScratch + ffnScratch);
    const wsAlpha = vmemParts.totalPerTokenBytes;
    const extraSlots = bufSlots - 2;
    const wideRing = (14 + extraSlots) * dBank +
      (lMoe > 0 ? (10 + extraSlots) * eBank : 0);
    const shrunkRing = bufSlots * dBank + (lMoe > 0 ? bufSlots * eBank : 0);
    const wsNonRing = wsScratch + 2097152;
    const tWideMax = (wideRing + wsNonRing + wsAlpha > vmemBudget)
      ? 0 : Math.max(1, Math.floor((vmemBudget - wideRing - wsNonRing) / wsAlpha));
    const tShrunkMax = (shrunkRing + wsNonRing + wsAlpha > vmemBudget)
      ? 0 : Math.max(1, Math.floor((vmemBudget - shrunkRing - wsNonRing) / wsAlpha));
    return {
      activeTokens, effCtx, epDp, scopes, winner, runnerUp, recScope, tpsChip,
      hbmFits, hbmPoolOverflow, hbmUsedBytes: hbmUsed, hbmPoolCapBytes,
      invalidReasons, nonHomogeneousReasons: p.non_homogeneous_reasons || [],
      compilerProofGainUs, dispatchBubbleGainUs,
      compilerProofGainPct, achievableGainPct, tWideMax, tShrunkMax,
      ridgeFb: peakW / bwHbm, moeModeRec, marginPct,
      launchGapLoUs: lgBox.lo, launchGapHiUs: lgBox.hi,
      vmemBudgetMib: +(vmemBudget / 1048576).toFixed(2), vmemParts
    };
  }
    function evaluateScenario(params) {
      if (!params) return null;
      const p = params.profile || PROF;
      const ch = params.chip || CHIP;
      const wl = params.wl || params;
      const inv = validateWorkload(p, ch, wl);
      if (params.only_validate) {
        return { invalid_reasons: inv, scopes: {} };
      }
      const res = evaluateWorkload(wl, p, ch);
      const scopesOut = {};
      if (res && res.scopes) {
        for (const sc of res.scopes) {
          scopesOut[sc.scope] = sc;
        }
      }
      return { invalid_reasons: inv, scopes: scopesOut, full: res };
    }

    return {
      DATA, PROF, CHIP, REGIMES, roundEven, gcd, peakFlops, rowFloor,
      allReduceCost, computePerTokenVmem,
      expertsHit, selectAdaptiveRing, validateWorkload, evaluateWorkload,
      evaluateScenario
    };
  }

  global.__MK_CREATE_ADVISOR_ENGINE = createAdvisorEngine;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createAdvisorEngine };
  }
  if (typeof window !== 'undefined') {
    window.__MK_CREATE_ADVISOR_ENGINE = createAdvisorEngine;
    if (typeof document !== 'undefined' && document && typeof document.getElementById === 'function') {
      const el = document.getElementById('advisor-data');
      if (el && el.textContent) {
        window.__MK_ADVISOR_ENGINE = createAdvisorEngine(JSON.parse(el.textContent));
      }
    }
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
