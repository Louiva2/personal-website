---
title: "DeepSeek-V4.1-Flash: architecture and operator sequence (work in progress)"
date: 2026-11-17
summary: The first step before working out the edge for DeepSeek-V4.1-Flash on 8×H20. Structure, shapes, and formulas for the FLOPs and bytes of every operator. No hardware numbers, and no prediction or measurement yet.
---

> **Work in progress.** This is step 1 of the planner work for this series: the model side only. It gives shapes, and FLOPs and bytes as symbolic formulas. There are no hardware numbers (no bandwidth, no FLOP/s) and no timing. No prediction has been written down and nothing has been measured. None of it has been checked on an H20 yet. The byte counts per cache entry and per token below describe the layout SGLang selects for H20. They come from the model and the code, not from the card.
>
> Three open items have to be settled before any number goes on the chart:
>
> - **MoE combine in the baseline.** Is it one fused all-reduce of the routed and shared partial sums, with A2A set to none? That is inferred and needs a trace.
> - **Shared expert TP-sharded, 288 per rank.** This is inferred: only that layout reproduces the 2.5 GiB BF16-cache figure. A replicated shared expert would give 4.8 GiB.
> - **MMA precision of attention and the indexer on SM90.** For attention, BF16 MMA after FP8→BF16 dequant is likely but unverified. It decides which ridge the executed 224 FLOP/B is compared against. For the indexer score, the FP4 indexer flag is rejected on SM90, so the actual dtype of the logits GEMM is unknown.
>
> Also not verified yet:
>
> - SWA pool page size (≈256, inferred from the 13 GiB figure).
> - Which mHC traffic variant the SM90 kernels achieve.
> - Residual-stream dtype (BF16 assumed).
> - `q_head_norm` default (negligible).
> - Prefill sparse-gather reuse in FlashMLA (the upper bound is given).
> - DSpark drafter SWA cache sizing.
> - How the Engram sharded lookup communicates.

Dimensions come from the HF `config.json` and the headers of all 48 safetensors shards (dtype and shape of all 96,085 tensors) at revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, fetched 2026-11-17. Deployment semantics come from SGLang at SHA `3e13f1b2c725d773b4dd845d12bd82903a7dba3f`. The technical report is used for intent only. The baseline deployment is the one in my H20 deployment notes: 8×H20, TP 8, EP 8, no DP attention, MoE A2A none, bounded replay off on both encoder and decoder, no speculative decoding.

Checks against the report. The parameters summed from the headers are 551.9B for the backbone (report: 552B) and 196.6B for the Engram tables (report: 196B). The activated parameters worked out from these dimensions are 16.1B in decode (report: 16B) and 7.9B for the encoder alone (report: "8B in prefill"). The BF16 cache at TP 8 comes to 2.46 GiB per rank (deployment notes: 2.5 GiB). The checkpoint is 510,286,023,000 bytes. The maximum position is 1,048,576.

## 0. Symbols

| symbol | meaning | value (baseline 8×H20) |
|---|---|---|
| $B$ | requests in the step | – |
| $S$ | new tokens per request (prefill) | – |
| $L$ | visible context of a query: cached prefix + new, including itself | – |
| $T$ | tokens in the step: prefill $B S$, decode $B$, DSpark verify $B(\gamma+1)=6B$ | – |
| $d$ | hidden size | 5120 |
| $n$ | mHC streams | 4 |
| $H$, $h$ | query heads, total / per attention-TP rank | 64 / 8 |
| $h_k$ | heads the FlashMLA kernel computes (SM90 pads to 64 when TP > 1) | 64 |
| $D$, $r$ | latent head dim / RoPE dims inside it | 512 / 64 |
| $q_r$ | q LoRA rank | 1280 |
| $G$, $g$ | output groups, total / per rank | 8 / 1 |
| $o_r$ | per-group output rank | 1024 |
| $m$ | compress ratio of the layer (0 = SWA only) | 2 (layers 2–19), 1 (layers 20–39) |
| $N_c$ | visible compressed entries of the layer's kv_src: $\lceil L/m\rceil$ | – |
| $K_{\mathrm{sel}}$ | selected entries: $\min(512, N_c)$ | – |
| $W_{\mathrm{eff}}$ | SWA keys: $\min(128, L)$ | – |
| $P_c$ | Reindex search domain: $\min(16384, N_c)$ | – |
| $H_I$, $D_I$ | indexer heads / dim | 32 / 128 |
| $E$, $k$, $f$ | routed experts / top-k / expert FFN dim | 384 / 6 / 2304 |
| TP, EP | tensor / expert parallel | 8 / 8 |
| $E_{\mathrm{act}}$ | distinct experts hit in a layer: $E\,(1-(1-k/E)^T)$ (uniform-routing assumption) | – |
| $E_r$ | distinct experts hit on one EP rank: $E_{\mathrm{act}}/\mathrm{EP}$ | – |
| $T_r$ | (token, expert) pairs landing on one rank: $\approx T k/\mathrm{EP}$ | – |
| $a$ | activation bytes per element (BF16 assumed) | 2 |
| $b_w(M)$ | dense-weight bytes per parameter, a function of the GEMM's M: $1+1/1024$ (FP8, block 32) if $M<64$, else $2$ (Hopper BF16 cache) | – |
| $b_e$ | routed-expert bytes per parameter (MXFP4 + E8M0 per 32) | 0.53125 |
| $b_{kv}$ | one KV entry (SGLang V4 layout: 448 FP8 + 64 BF16 + 8 B scales) | 584 |
| $b_{ik}$ | one indexer-K entry (FP4 + scale per 32) | 68 |

All formulas are **per GPU rank** unless marked "global". Anything replicated across TP is *not* divided by TP. That covers `wq_a`, `wkv`, the compressor, the whole indexer, the router, the Engram `wkv`, and the KV cache itself.

## 1. Diagrams

### 1a. The whole model

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 628" role="img" aria-labelledby="v41-model-en-title v41-model-en-desc" font-size="11">
<title id="v41-model-en-title">DeepSeek-V4.1-Flash, the whole model</title>
<desc id="v41-model-en-desc">Top to bottom: input ids (plus image patches) go through the vocab-parallel embedding and the mHC expand into four streams. Layers 0 and 1 are SWA-only with MoE, with Engram before layer 1. The encoder, layers 2 to 19, is three groups of one CSA2 Full layer at ratio 2 and five Reuse layers, with kv sources 2, 8 and 14 and Engram before layer 14. Its output H_20 feeds layer 20, the decoder's Full layer at ratio 1, whose compressor builds the decoder global KV and whose indexer scans all of L and builds the candidate pool. Then layers 21 to 23 Reuse, layers 24 to 39 four groups of one pool-restricted Reindex layer and three Reuse layers, the mHC collapse with the last A coefficients and RMSNorm, and the LM head. On the side: the vision encoder feeds image tokens into the embedding; every layer keeps a 128-entry SWA ring; layers 2, 8, 14 and 20 write compressed KV and indexer K into the global cache, 1630 bytes per token on H20; the DSpark drafter, off in the baseline, reads the hidden states of layers 37 to 39 and proposes drafts to the head.</desc>
<rect class="flow-opt" x="238" y="391" width="116" height="166" rx="3"/>
<text text-anchor="middle"><tspan x="296" y="408.0">DSpark drafter</tspan><tspan x="296" y="422.0" class="flow-quiet">(off in baseline)</tspan><tspan x="296" y="436.0">reads hidden of</tspan><tspan x="296" y="450.0">L37, L38, L39</tspan><tspan x="296" y="464.0">main_proj</tspan><tspan x="296" y="478.0">15360→5120</tspan><tspan x="296" y="492.0">3 SWA-only blocks</tspan><tspan x="296" y="506.0">(MoE 128e top-3)</tspan><tspan x="296" y="520.0">5 draft positions</tspan><tspan x="296" y="534.0">Markov head r=256</tspan><tspan x="296" y="548.0">confidence head</tspan></text>
<path class="flow-store" d="M238 252V348A58 4 0 0 0 354 348V252A58 4 0 0 0 238 252Z"/><path class="flow-store-rim" d="M238 252A58 4 0 0 0 354 252"/>
<text text-anchor="middle"><tspan x="296" y="273.0">global KV cache</tspan><tspan x="296" y="287.0">1630 B/token</tspan><tspan x="296" y="301.0" class="flow-quiet">on H20</tspan><tspan x="296" y="315.0" class="flow-quiet">← L2/L8/L14 (m=2)</tspan><tspan x="296" y="329.0" class="flow-quiet">← L20 (m=1)</tspan><tspan x="296" y="343.0" class="flow-quiet">each + indexer K</tspan></text>
<path class="flow-store" d="M238 176V244A58 4 0 0 0 354 244V176A58 4 0 0 0 238 176Z"/><path class="flow-store-rim" d="M238 176A58 4 0 0 0 354 176"/>
<text text-anchor="middle"><tspan x="296" y="197.0">SWA cache,</tspan><tspan x="296" y="211.0">all 40 layers</tspan><tspan x="296" y="225.0" class="flow-quiet">ring 128 × 584 B</tspan><tspan x="296" y="239.0" class="flow-quiet">per layer per req</tspan></text>
<rect class="flow-opt" x="238" y="6" width="116" height="68" rx="3"/>
<text text-anchor="middle"><tspan x="296" y="23.0">DeepSeek-ViT</tspan><tspan x="296" y="37.0">32L, d=1024</tspan><tspan x="296" y="51.0">+ 3×3 unshuffle</tspan><tspan x="296" y="65.0">+ aligner MLP</tspan></text>
<rect class="flow-box" x="6" y="596" width="220" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="613.0">head 129280×5120 BF16 → logits</tspan></text>
<rect class="flow-box" x="6" y="538" width="220" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="555.0">mHC collapse</tspan><tspan x="116" y="569.0">(hc_combine with last A) → RMSNorm</tspan></text>
<rect class="flow-box" x="6" y="480" width="220" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="497.0">L24–39: 4 × [Reindex</tspan><tspan x="116" y="511.0">(pool-restricted) + 3 × Reuse] + MoE</tspan></text>
<rect class="flow-box" x="6" y="436" width="220" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="453.0">L21–23 Reuse</tspan></text>
<rect class="flow-box" x="6" y="336" width="220" height="82" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="353.0">L20 CSA2 (m=1) Full:</tspan><tspan x="116" y="367.0">compressor builds decoder</tspan><tspan x="116" y="381.0">global KV from H_20</tspan><tspan x="116" y="395.0">+ indexer over all L</tspan><tspan x="116" y="409.0">+ candidate pool (2048 blocks × 8)</tspan></text>
<rect class="flow-box" x="6" y="244" width="220" height="68" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="261.0">Encoder L2–19:</tspan><tspan x="116" y="275.0">3 × [CSA2 (m=2) Full + 5 × Reuse]</tspan><tspan x="116" y="289.0">+ MoE</tspan><tspan x="116" y="303.0" class="flow-quiet">(kv_src 2/8/14; Engram before L14)</tspan></text>
<rect class="flow-box" x="6" y="182" width="220" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="199.0">L0–1 SWA-only + MoE</tspan><tspan x="116" y="213.0" class="flow-quiet">(Engram before L1)</tspan></text>
<rect class="flow-box" x="6" y="138" width="220" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="155.0">mHC expand → X [T,4,d]</tspan></text>
<rect class="flow-box" x="6" y="80" width="220" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="97.0">embed 129280×5120 BF16</tspan><tspan x="116" y="111.0">(vocab-parallel) → x [T,d]</tspan></text>
<rect class="flow-box" x="6" y="6" width="220" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="116" y="23.0">input_ids [B,S] (+ image patches)</tspan></text>
<path class="flow-edge" d="M116 32L116 80"/><path class="flow-head" d="M116 80L112.8 74.0L119.2 74.0Z"/>
<path class="flow-edge" d="M116 120L116 138"/><path class="flow-head" d="M116 138L112.8 132.0L119.2 132.0Z"/>
<path class="flow-edge" d="M116 164L116 182"/><path class="flow-head" d="M116 182L112.8 176.0L119.2 176.0Z"/>
<path class="flow-edge" d="M116 222L116 244"/><path class="flow-head" d="M116 244L112.8 238.0L119.2 238.0Z"/>
<path class="flow-edge" d="M116 312L116 336"/><path class="flow-head" d="M116 336L112.8 330.0L119.2 330.0Z"/>
<path class="flow-edge" d="M116 418L116 436"/><path class="flow-head" d="M116 436L112.8 430.0L119.2 430.0Z"/>
<path class="flow-edge" d="M116 462L116 480"/><path class="flow-head" d="M116 480L112.8 474.0L119.2 474.0Z"/>
<path class="flow-edge" d="M116 520L116 538"/><path class="flow-head" d="M116 538L112.8 532.0L119.2 532.0Z"/>
<path class="flow-edge" d="M116 578L116 596"/><path class="flow-head" d="M116 596L112.8 590.0L119.2 590.0Z"/>
<text class="flow-quiet" x="121" y="327" text-anchor="start" font-size="10">H_20 = encoder output</text>
<path class="flow-edge flow-edge-opt" d="M296 74L296 100L226 100"/><path class="flow-head flow-head-opt" d="M226 100L232.0 96.8L232.0 103.2Z"/>
<text class="flow-quiet" x="292" y="113" text-anchor="end" font-size="10">image tokens</text>
<path class="flow-edge flow-edge-opt" d="M226 202L238 202"/><path class="flow-head flow-head-opt" d="M238 202L232.0 205.2L232.0 198.8Z"/>
<path class="flow-edge flow-edge-opt" d="M226 278L238 278"/><path class="flow-head flow-head-opt" d="M238 278L232.0 281.2L232.0 274.8Z"/>
<path class="flow-edge flow-edge-opt" d="M226 377L252 377L252 349"/><path class="flow-head flow-head-opt" d="M252 349L255.2 355.0L248.8 355.0Z"/>
<path class="flow-edge flow-edge-opt" d="M226 500L238 500"/><path class="flow-head flow-head-opt" d="M238 500L232.0 503.2L232.0 496.8Z"/>
<path class="flow-edge flow-edge-opt" d="M296 557L296 609L226 609"/><path class="flow-head flow-head-opt" d="M226 609L232.0 605.8L232.0 612.2Z"/>
<text class="flow-quiet" x="292" y="605" text-anchor="end" font-size="10">drafts</text>
</svg>
<figcaption>The whole model. Dashed: the image path, the DSpark drafter (off in the baseline), and writes into the caches.</figcaption>
</figure>

With decoder SWA bounded replay on, layers 21–39 process only the last 128 tokens of each prompt. It is off in the baseline, so all 40 layers process all $S$ tokens.

### 1b. One block: layer $l$ with CSA2 and MoE

The block is split in two to stay legible on a phone. The dashed boxes are what Full mode adds.

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 936" role="img" aria-labelledby="v41-block-attn-en-title v41-block-attn-en-desc" font-size="11">
<title id="v41-block-attn-en-title">One block, attention half</title>
<desc id="v41-block-attn-en-desc">Layer l with CSA2 attention. The four mHC streams X_l and the previous sublayer output enter the mHC boundary, which updates the residual, computes 24 coefficients with 20 Sinkhorn iterations, mixes the input and applies attn_norm to give x. A fused replicated GEMM computes wq_a and wkv. wkv goes through kv_norm and RoPE into the SWA ring; wq_a goes through q_norm to the query latent, then wq_b to 8 heads per rank, RoPE, and zero padding to 64 heads. Dashed boxes run only in Full mode: the compressor pools m tokens into a compressed entry, indexer K projects it, the indexer weights and indexer Q are computed, the score sums relu of q dot k over the visible entries weighted per head, and the top 512 indices are kept. Reuse layers read the indices of their topk source. FlashMLA sparse MQA attends to the selected entries plus the SWA window, then inverse RoPE, the grouped wo_a in BF16, the row-parallel wo_b, and an all-reduce.</desc>
<path class="flow-comm" d="M94 904H266L274 917L266 930H94L86 917Z"/>
<text text-anchor="middle"><tspan x="180" y="921.0">all-reduce [T,d]</tspan></text>
<rect class="flow-box" x="66" y="862" width="228" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="879.0">wo_b [d,g·1024] (row-parallel)</tspan></text>
<rect class="flow-box" x="66" y="806" width="228" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="823.0">wo_a grouped [g,1024,4096] BF16</tspan><tspan x="180" y="837.0">→ [T,g·1024]</tspan></text>
<rect class="flow-box" x="66" y="764" width="228" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="781.0">inverse RoPE on o[…,−64:]</tspan></text>
<rect class="flow-box" x="6" y="708" width="348" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="725.0">FlashMLA sparse MQA, K = V latent, attn_sink</tspan><tspan x="180" y="739.0">keys = K_sel gathered + W_eff SWA → o [T,h,512]</tspan></text>
<rect class="flow-opt" x="198" y="638" width="156" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="655.0">idx [T,512]</tspan><tspan x="276" y="669.0" class="flow-quiet">(Reuse layers read</tspan><tspan x="276" y="683.0" class="flow-quiet">the idx of topk_src)</tspan></text>
<rect class="flow-opt" x="198" y="540" width="156" height="82" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="557.0">score s =</tspan><tspan x="276" y="571.0">Σ_h w_h·relu(q_h·k)</tspan><tspan x="276" y="585.0">over N_c (Full)</tspan><tspan x="276" y="599.0">or P_c (Reindex)</tspan><tspan x="276" y="613.0">→ top-512 idx [T,512]</tspan></text>
<rect class="flow-opt" x="198" y="456" width="156" height="68" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="473.0">q_lat → indexer Q:</tspan><tspan x="276" y="487.0">wq_b_I [4096,1280]</tspan><tspan x="276" y="501.0" class="flow-quiet">(all 32 heads, replicated)</tspan><tspan x="276" y="515.0">→ [T,32,128]</tspan></text>
<rect class="flow-opt" x="198" y="406" width="156" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="423.0">x → weights_proj [32,d]</tspan><tspan x="276" y="437.0">→ w [T,32]</tspan></text>
<rect class="flow-opt" x="198" y="342" width="156" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="359.0">indexer K: wk [128,512]</tspan><tspan x="276" y="373.0">→ k_norm, RoPE, FP4</tspan><tspan x="276" y="387.0">→ [T/m,128] (68 B)</tspan></text>
<rect class="flow-opt" x="198" y="216" width="156" height="110" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="233.0">x → compressor:</tspan><tspan x="276" y="247.0">wkv (+wgate) [512,d] BF16</tspan><tspan x="276" y="261.0">→ pool m tokens</tspan><tspan x="276" y="275.0" class="flow-quiet">(softmax gate)</tspan><tspan x="276" y="289.0">→ RMSNorm, RoPE</tspan><tspan x="276" y="303.0">→ [T/m,512]</tspan><tspan x="276" y="317.0">→ extra cache (584 B)</tspan></text>
<rect class="flow-box" x="6" y="216" width="172" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="92" y="233.0">wq_b [h·512,1280]</tspan><tspan x="92" y="247.0">→ q [T,h,512] + RoPE</tspan><tspan x="92" y="261.0">pad to h_k = 64</tspan></text>
<rect class="flow-box" x="6" y="174" width="172" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="92" y="191.0">q_norm → q_lat [T,1280]</tspan></text>
<rect class="flow-box" x="198" y="118" width="156" height="68" rx="3"/>
<text text-anchor="middle"><tspan x="276" y="135.0">wkv → kv_norm</tspan><tspan x="276" y="149.0">+ RoPE (last 64)</tspan><tspan x="276" y="163.0">→ SWA KV [T,512]</tspan><tspan x="276" y="177.0">→ SWA ring (584 B/entry)</tspan></text>
<rect class="flow-box" x="6" y="118" width="172" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="92" y="135.0">wq_a [1280,d] + wkv [512,d]</tspan><tspan x="92" y="149.0" class="flow-quiet">(replicated, fused)</tspan></text>
<rect class="flow-box" x="6" y="48" width="348" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="65.0">mHC boundary: residual update + coeffs</tspan><tspan x="180" y="79.0">(24 = A4+B16+C4, Sinkhorn 20)</tspan><tspan x="180" y="93.0">+ input mix with A_(l−1) + attn_norm → x [T,d]</tspan></text>
<rect class="flow-box" x="6" y="6" width="348" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="23.0">X_l [T,4,d] (mHC streams) + Y_(l−1) [T,d]</tspan></text>
<text class="flow-quiet" x="354" y="210" text-anchor="end" font-size="10">dashed: Full mode only</text>
<path class="flow-edge" d="M180 32L180 48"/><path class="flow-head" d="M180 48L176.8 42.0L183.2 42.0Z"/>
<path class="flow-edge" d="M92 102L92 118"/><path class="flow-head" d="M92 118L88.8 112.0L95.2 112.0Z"/>
<path class="flow-edge" d="M178 138L198 138"/><path class="flow-head" d="M198 138L192.0 141.2L192.0 134.8Z"/>
<path class="flow-edge" d="M92 158L92 174"/><path class="flow-head" d="M92 174L88.8 168.0L95.2 168.0Z"/>
<path class="flow-edge" d="M92 200L92 216"/><path class="flow-head" d="M92 216L88.8 210.0L95.2 210.0Z"/>
<path class="flow-edge" d="M92 270L92 708"/><path class="flow-head" d="M92 708L88.8 702.0L95.2 702.0Z"/>
<path class="flow-edge" d="M354 152L358 152L358 708"/><path class="flow-head" d="M358 708L354.8 702.0L361.2 702.0Z"/>
<path class="flow-edge flow-edge-opt" d="M276 326L276 342"/><path class="flow-head flow-head-opt" d="M276 342L272.8 336.0L279.2 336.0Z"/>
<path class="flow-edge flow-edge-opt" d="M198 369H189"/>
<path class="flow-edge flow-edge-opt" d="M198 426H189"/>
<path class="flow-edge flow-edge-opt" d="M198 490H189"/>
<path class="flow-edge flow-edge-opt" d="M189 369L189 581L198 581"/><path class="flow-head flow-head-opt" d="M198 581L192.0 584.2L192.0 577.8Z"/>
<path class="flow-edge flow-edge-opt" d="M276 622L276 638"/><path class="flow-head flow-head-opt" d="M276 638L272.8 632.0L279.2 632.0Z"/>
<path class="flow-edge" d="M276 692L276 708"/><path class="flow-head" d="M276 708L272.8 702.0L279.2 702.0Z"/>
<path class="flow-edge" d="M180 748L180 764"/><path class="flow-head" d="M180 764L176.8 758.0L183.2 758.0Z"/>
<path class="flow-edge" d="M180 790L180 806"/><path class="flow-head" d="M180 806L176.8 800.0L183.2 800.0Z"/>
<path class="flow-edge" d="M180 846L180 862"/><path class="flow-head" d="M180 862L176.8 856.0L183.2 856.0Z"/>
<path class="flow-edge" d="M180 888L180 904"/><path class="flow-head" d="M180 904L176.8 898.0L183.2 898.0Z"/>
</svg>
<figcaption>One block, attention half. Dashed: what Full mode adds (layers 2, 8, 14, 20). Reindex layers keep only indexer Q, weights_proj and the score, over the candidate pool. Reuse layers skip all of it and read idx from their topk_src.</figcaption>
</figure>

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 430" role="img" aria-labelledby="v41-block-moe-en-title v41-block-moe-en-desc" font-size="11">
<title id="v41-block-moe-en-title">One block, MoE half</title>
<desc id="v41-block-moe-en-desc">After the attention all-reduce, the mHC boundary between attention and FFN applies ffn_norm to give x2. The router, a replicated BF16 GEMM with sqrtsoftplus scores plus a selection bias, picks the top 6 experts, renormalizes and scales by 1.5. Dispatch is empty in the baseline, or a DeepEP all-to-all. Each rank holds 48 routed experts in MXFP4, run as W4A16: w1 and w3, SwiGLU clamped at 10, w2. The shared expert runs in FP8, TP-sharded 288 per rank. The combine is an all-reduce in the baseline, or an all-to-all combine, and gives Y_l for the next mHC boundary.</desc>
<rect class="flow-box" x="66" y="398" width="228" height="26" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="415.0">Y_l [T,d] → next mHC boundary</tspan></text>
<path class="flow-comm" d="M44 342H316L324 362L316 382H44L36 362Z"/>
<text text-anchor="middle"><tspan x="180" y="359.0">combine: all-reduce [T,d] (baseline)</tspan><tspan x="180" y="373.0" class="flow-quiet">/ A2A combine</tspan></text>
<rect class="flow-box" x="6" y="244" width="204" height="82" rx="3"/>
<text text-anchor="middle"><tspan x="108" y="261.0">routed experts, 48 per rank,</tspan><tspan x="108" y="275.0">MXFP4 W4A16:</tspan><tspan x="108" y="289.0">w1+w3 [2×2304,d]</tspan><tspan x="108" y="303.0">→ SwiGLU clamp 10</tspan><tspan x="108" y="317.0">→ w2 [d,2304]</tspan></text>
<path class="flow-comm" d="M22 174H194L202 201L194 228H22L14 201Z"/>
<text text-anchor="middle"><tspan x="108" y="191.0">dispatch</tspan><tspan x="108" y="205.0" class="flow-quiet">(none in baseline /</tspan><tspan x="108" y="219.0" class="flow-quiet">DeepEP A2A)</tspan></text>
<rect class="flow-box" x="226" y="104" width="128" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="290" y="121.0">shared expert</tspan><tspan x="290" y="135.0">FP8, TP-sharded</tspan><tspan x="290" y="149.0">288 / rank</tspan></text>
<rect class="flow-box" x="6" y="104" width="204" height="54" rx="3"/>
<text text-anchor="middle"><tspan x="108" y="121.0">router [384,d] BF16</tspan><tspan x="108" y="135.0">sqrtsoftplus + bias, top-6,</tspan><tspan x="108" y="149.0">norm, ×1.5</tspan></text>
<rect class="flow-box" x="6" y="48" width="348" height="40" rx="3"/>
<text text-anchor="middle"><tspan x="180" y="65.0">mHC boundary (post-attn / pre-ffn)</tspan><tspan x="180" y="79.0">+ ffn_norm → x2 [T,d]</tspan></text>
<path class="flow-comm" d="M94 6H266L274 19L266 32H94L86 19Z"/>
<text text-anchor="middle"><tspan x="180" y="23.0">all-reduce [T,d] (from above)</tspan></text>
<path class="flow-edge" d="M180 32L180 48"/><path class="flow-head" d="M180 48L176.8 42.0L183.2 42.0Z"/>
<path class="flow-edge" d="M108 88L108 104"/><path class="flow-head" d="M108 104L104.8 98.0L111.2 98.0Z"/>
<path class="flow-edge" d="M290 88L290 104"/><path class="flow-head" d="M290 104L286.8 98.0L293.2 98.0Z"/>
<path class="flow-edge" d="M108 158L108 174"/><path class="flow-head" d="M108 174L104.8 168.0L111.2 168.0Z"/>
<path class="flow-edge" d="M108 228L108 244"/><path class="flow-head" d="M108 244L104.8 238.0L111.2 238.0Z"/>
<path class="flow-edge" d="M108 326L108 342"/><path class="flow-head" d="M108 342L104.8 336.0L111.2 336.0Z"/>
<path class="flow-edge" d="M290 158L290 342"/><path class="flow-head" d="M290 342L286.8 336.0L293.2 336.0Z"/>
<path class="flow-edge" d="M180 382L180 398"/><path class="flow-head" d="M180 398L176.8 392.0L183.2 392.0Z"/>
</svg>
<figcaption>One block, MoE half. The pointed boxes are communication.</figcaption>
</figure>

## 2. Which tables apply to which layer

The config has **no dense-FFN layer**: every one of the 40 layers is MoE, and there is no hash routing. The axis that changes the operator list is the **attention mode**. Section 3 gives the full operator sequence for a **Reuse + MoE** block, which covers 30 of the 40 layers. Section 4 gives the deltas for the other modes.

| mode | layers | count | what it adds or removes vs. Reuse |
|---|---|---|---|
| SWA-only | 0, 1 | 2 | No sparse part (no idx read, $K_{\mathrm{sel}}=0$). RoPE θ = 1e4 instead of θ = 1.6e5 + YaRN. |
| Full, $m=2$ | 2, 8, 14 | 3 | + compressor (with gate) + indexer K + indexer Q + score over $N_c=\lceil L/2\rceil$ + top-k |
| Full, $m=1$ | 20 | 1 | + compressor (no gate) + indexer K + indexer Q + score over $N_c=L$ + top-k + **candidate pool** build |
| Reindex, $m=1$ | 24, 28, 32, 36 | 4 | + indexer Q + score over the pool $P_c=\min(16384, L)$ + top-k |
| Reuse | 3–7, 9–13, 15–19, 21–23, 25–27, 29–31, 33–35, 37–39 | 30 | (the baseline table) |
| + Engram (before the layer) | 1, 14 | 2 | + hash, table gather, `wkv` GEMM, gating |

Per layer, `kv_src` is the layer whose compressed KV and indexer K are read, and `topk_src` the layer whose top-512 indices are used. Layers 0–1 have neither. Layers 2–7 read layer 2 for both, 8–13 layer 8, 14–19 layer 14. Layers 20–39 all read KV from layer 20; their `topk_src` is 20 for layers 20–23, then 24, 28, 32 and 36 for each group of four from there on. Only the four `kv_src` layers own compressed storage. Engram runs on the residual *before* layers 1 and 14.

**Token count per layer in prefill.** $T_l = BS$ for $l \le 20$. For $l \ge 21$, $T_l = BS$ when decoder bounded replay is off (the baseline) and $T_l = B\min(S,128)$ when it is on. The split comes from SGLang's `late_layer_start = max(kv_source) + 1 = 21`, which is not $L/2$.

## 3. Reuse + MoE block: the full operator sequence

Columns: W = weight bytes, A = activation bytes read, KV = cache bytes read, Wr = bytes written. Comm rows are placeholders and are not sized.

### 3a. Decode (one new token per request, $T=B$, context $L$)

| # | op | kind | input | weight | output | FLOPs (per rank) | W read | A read | KV read | Wr |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mHC boundary (post previous FFN + coeffs + mix + attn_norm + quant) | norm / elementwise + tiny GEMM | X `[T,4,d]`, Y_prev `[T,d]` | `hc_attn_fn` `[24,4d]` F32 | X′ `[T,4,d]`, x `[T,d]` | $2T\cdot 4d\cdot 24+O(T\cdot 4d\cdot 20)$ | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$ (ideal; see §5.9) | – | $T(n+1)d\,a$ |
| 2 | `wq_a` ‖ `wkv` (fused, replicated) | GEMM | x `[T,d]` | `[1792,d]` FP8 | `[T,1280]`, `[T,512]` | $2Td\cdot 1792$ | $1792\,d\,b_w(T)$ | $Td\,a$ | – | $T\cdot 1792\,a$ |
| 3 | `q_norm`; `kv_norm` + RoPE + quant + SWA store | norm / elementwise | `[T,1280]`, `[T,512]` | `[1280]`, `[512]` | q_lat `[T,1280]`; SWA entry | $O(T\cdot 1792)$ | ~0 | $T\cdot 1792\,a$ | – | $T\cdot 1280\,a+T\,b_{kv}$ |
| 4 | `wq_b` (column-parallel) | GEMM | q_lat `[T,1280]` | `[h·512,1280]` FP8 | q `[T,h,512]` | $2T\cdot 1280\cdot 512h$ | $512h\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | – | $Th\cdot 512\,a$ |
| 5 | RoPE(q[…,−64:]) + zero-pad to $h_k$ | elementwise | q | – | q_pad `[T,h_k,512]` | ~0 | – | $Th\cdot 512\,a$ | – | $Th_k\cdot 512\,a$ |
| 6 | sparse MQA attention (FlashMLA sparse decode, sink) | attention | q_pad, idx `[T,512]` from topk_src, SWA ring | – | o `[T,h_k,512]` | executed $2048\,h_k T(K_{\mathrm{sel}}+W_{\mathrm{eff}})$; useful $2048\,hT(K_{\mathrm{sel}}+W_{\mathrm{eff}})$ | – | $Th_k\cdot 512\,a+T\cdot 512\cdot 4$ | $T(K_{\mathrm{sel}}+W_{\mathrm{eff}})\,b_{kv}$ (one latent per key, K = V) | $Th_k\cdot 512\,a$ |
| 7 | slice to $h$, inverse RoPE on o[…,−64:] | elementwise | o | – | `[T,h,512]` | ~0 | – | $Th\cdot 512\,a$ | – | $Th\cdot 512\,a$ |
| 8 | `wo_a` (grouped, block-diagonal) | GEMM (batched over $g$) | `[T,g,4096]` | `[g,1024,4096]` **BF16** | `[T,g·1024]` | $2Tg\cdot 1024\cdot 4096$ | $g\cdot 1024\cdot 4096\cdot 2$ | $Tg\cdot 4096\,a$ | – | $Tg\cdot 1024\,a$ |
| 9 | `wo_b` (row-parallel) | GEMM | `[T,g·1024]` | `[d,g·1024]` FP8 | partial `[T,d]` | $2Tg\cdot 1024\,d$ | $d\,g\cdot 1024\,b_w(T)$ | $Tg\cdot 1024\,a$ | – | $Td\,a$ |
| 10 | **COMM: all-reduce (attention TP)** | comm placeholder | `[T,d]` | – | `[T,d]` | – | – | – | – | – |
| 11 | mHC boundary (post-attn + pre-ffn + ffn_norm) | norm / elementwise + tiny GEMM | X′, y | `hc_ffn_fn` `[24,4d]` F32 | X″, x2 `[T,d]` | as #1 | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$ | – | $T(n+1)d\,a$ |
| 12 | router (sqrtsoftplus, bias / bias_vl select, top-6, renorm, ×1.5) | router | x2 | `[384,d]` BF16 (replicated) | idx `[T,6]`, w `[T,6]` | $2Td\cdot 384+O(T\cdot 384)$ | $384\,d\cdot 2$ | $Td\,a$ | – | $T\cdot 6\cdot 8$ |
| 13 | **COMM: MoE dispatch** (none in the baseline: tokens are already replicated; DeepEP A2A is the alternative) | comm placeholder | `[T,d]` | – | `[T_r,d]` | – | – | – | – | – |
| 14 | routed `w1` ‖ `w3` (grouped over the $E_r$ local experts) | GEMM (grouped) | `[T_r,d]` | $E_r\times$ `[4608,d]` MXFP4 | `[T_r,4608]` | $2T_r d\cdot 4608$ | $E_r\cdot 4608\,d\,b_e$ | $T_r d\,a$ | – | $T_r\cdot 4608\,a$ |
| 15 | SwiGLU + clamp 10 | elementwise | `[T_r,4608]` | – | `[T_r,2304]` | $O(T_r\cdot 2304)$ | – | $T_r\cdot 4608\,a$ | – | $T_r\cdot 2304\,a$ |
| 16 | routed `w2` | GEMM (grouped) | `[T_r,2304]` | $E_r\times$ `[d,2304]` MXFP4 | `[T_r,d]` | $2T_r\cdot 2304\,d$ | $E_r\cdot 2304\,d\,b_e$ | $T_r\cdot 2304\,a$ | – | $T_r d\,a$ |
| 17 | top-k weighted reduce (local hits) | elementwise | `[T_r,d]`, w | – | partial `[T,d]` | $O(T_r d)$ | – | $T_r d\,a$ | – | $Td\,a$ |
| 18 | shared `w1` ‖ `w3` (TP-sharded) | GEMM | x2 `[T,d]` | `[2f/TP,d]` FP8 | `[T,2f/TP]` | $2Td\cdot 2f/\mathrm{TP}$ | $(2f/\mathrm{TP})\,d\,b_w(T)$ | $Td\,a$ | – | $T(2f/\mathrm{TP})\,a$ |
| 19 | SwiGLU + clamp | elementwise | | – | `[T,f/TP]` | $O(Tf/\mathrm{TP})$ | – | $T(2f/\mathrm{TP})\,a$ | – | $T(f/\mathrm{TP})\,a$ |
| 20 | shared `w2` | GEMM | `[T,f/TP]` | `[d,f/TP]` FP8 | partial `[T,d]` | $2T(f/\mathrm{TP})\,d$ | $d\,(f/\mathrm{TP})\,b_w(T)$ | $T(f/\mathrm{TP})\,a$ | – | $Td\,a$ |
| 21 | **COMM: MoE combine** (baseline: all-reduce of the routed + shared partial sums; DeepEP: A2A combine + local add) | comm placeholder | `[T,d]` | – | Y `[T,d]` | – | – | – | – | – |

Notes on decode:

- **#6.** $K_{\mathrm{sel}}=\min(512,\lceil L/m\rceil)$, with the $m$ of the kv_src ($m=2$ in the encoder, $m=1$ in the decoder). KV traffic is per request, so it never amortizes over $B$. Every TP rank reads the same bytes, because the single latent head is replicated.
- **#14 and #16.** The expected number of tokens per hit expert is $Tk/E_{\mathrm{act}}$. For small $T$, $E_{\mathrm{act}}\approx kT$ and the expert GEMM is a GEMV per expert. EP balance is not guaranteed: the busiest rank sets the pace, so the planner uses $\max_{\text{rank}} E_r$, not the mean.
- **DSpark verify.** Replace $T=B$ with $T=B(\gamma+1)$. Each of those tokens has its own $L$, so $K_{\mathrm{sel}}$ and $W_{\mathrm{eff}}$ stay per token.

### 3b. Prefill ($S$ new tokens per request, causal; cached prefix $P_0$, so query $i$ sees $L_i=P_0+i$, $i=1,\dots,S$)

The per-request sums:

- $\Sigma_{\mathrm{sel}}(m)=\sum_{i=1}^{S}\min(512,\lceil (P_0+i)/m\rceil)$. For $P_0=0$ and $S\ge 512m$ this is about $512S-131072\,m$.
- $\Sigma_{\mathrm{win}}=\sum_{i}\min(128,P_0+i)$. For $P_0=0$ this is about $128S-8192$.

| # | op | kind | input | weight | output | FLOPs (per rank) | W read | A read | KV read | Wr |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mHC boundary + attn_norm | norm / elementwise | `[T,4,d]`, `[T,d]` | `[24,4d]` F32 | `[T,4,d]`, `[T,d]` | $2T\cdot 4d\cdot 24+O(T\cdot 4d\cdot 20)$ | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$ | – | $T(n+1)d\,a$ |
| 2 | `wq_a` ‖ `wkv` | GEMM | `[T,d]` | `[1792,d]` | `[T,1792]` | $2Td\cdot 1792$ | $1792\,d\,b_w(T)$ (BF16 once $T\ge 64$) | $Td\,a$ | – | $T\cdot 1792\,a$ |
| 3 | `q_norm`; `kv_norm` + RoPE + quant + SWA store | norm / elementwise | | | | $O(T\cdot 1792)$ | ~0 | $T\cdot 1792\,a$ | – | $T\cdot 1280\,a+T\,b_{kv}$ |
| 4 | `wq_b` | GEMM | `[T,1280]` | `[512h,1280]` | `[T,h,512]` | $2T\cdot 1280\cdot 512h$ | $512h\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | – | $T\cdot 512h\,a$ |
| 5 | RoPE(q) + pad to $h_k$ | elementwise | | | `[T,h_k,512]` | ~0 | – | $T\cdot 512h\,a$ | – | $T\cdot 512\,h_k\,a$ |
| 6 | sparse MQA attention (FlashMLA sparse prefill) | attention | q_pad, idx `[T,512]`, SWA (current chunk + ring) | – | `[T,h_k,512]` | executed $2048\,h_k B(\Sigma_{\mathrm{sel}}+\Sigma_{\mathrm{win}})$; useful with $h$ | – | $T\cdot 512\,h_k\,a+T\cdot 512\cdot 4$ | sparse: up to $B\,\Sigma_{\mathrm{sel}}\,b_{kv}$ (gathered per query; reuse across neighbouring queries depends on the kernel). SWA: $\ge B(S+W_{\mathrm{eff}})\,b_{kv}$ (dense tile) | $T\cdot 512\,h_k\,a$ |
| 7 | inverse RoPE | elementwise | | | `[T,h,512]` | ~0 | – | $T\cdot 512h\,a$ | – | $T\cdot 512h\,a$ |
| 8 | `wo_a` grouped | GEMM | `[T,g,4096]` | `[g,1024,4096]` BF16 | `[T,1024g]` | $2Tg\cdot 1024\cdot 4096$ | $g\cdot 1024\cdot 4096\cdot 2$ | $T\cdot 4096g\,a$ | – | $T\cdot 1024g\,a$ |
| 9 | `wo_b` | GEMM | `[T,1024g]` | `[d,1024g]` | `[T,d]` | $2T\cdot 1024g\,d$ | $d\cdot 1024g\,b_w(T)$ | $T\cdot 1024g\,a$ | – | $Td\,a$ |
| 10 | **COMM: all-reduce** | comm | `[T,d]` | | | | | | | |
| 11 | mHC boundary + ffn_norm | norm / elementwise | | `[24,4d]` | | as #1 | | | | |
| 12 | router | router | `[T,d]` | `[384,d]` BF16 | `[T,6]` | $2Td\cdot 384$ | $384\,d\cdot 2$ | $Td\,a$ | – | $48\,T$ |
| 13 | **COMM: dispatch** (placeholder) | comm | | | | | | | | |
| 14 | routed `w1` ‖ `w3` | GEMM (grouped) | `[T_r,d]` | $E_r\times$ `[4608,d]` | `[T_r,4608]` | $2T_r d\cdot 4608$ | $E_r\cdot 4608\,d\,b_e$ ($E_r\to 48$ for large $T$) | $T_r d\,a$ | – | $T_r\cdot 4608\,a$ |
| 15 | SwiGLU clamp | elementwise | | | | | | $T_r\cdot 4608\,a$ | | $T_r\cdot 2304\,a$ |
| 16 | routed `w2` | GEMM (grouped) | `[T_r,2304]` | $E_r\times$ `[d,2304]` | `[T_r,d]` | $2T_r\cdot 2304\,d$ | $E_r\cdot 2304\,d\,b_e$ | $T_r\cdot 2304\,a$ | – | $T_r d\,a$ |
| 17 | weighted reduce | elementwise | | | `[T,d]` | | | $T_r d\,a$ | | $Td\,a$ |
| 18–20 | shared expert (`w1` ‖ `w3`, SwiGLU, `w2`) | GEMM, elementwise, GEMM | `[T,d]` | `[2f/TP,d]`, `[d,f/TP]` FP8 | `[T,d]` | $6Td\,f/\mathrm{TP}$ | $3\,(f/\mathrm{TP})\,d\,b_w(T)$ | $Td\,a+\dots$ | – | $Td\,a$ |
| 21 | **COMM: combine** (placeholder) | comm | | | | | | | | |

Notes on prefill:

- **#6.** This attention is sparse even in prefill, so its FLOPs are linear in $S$, not quadratic. Per query and per kept head it costs $2048(K_{\mathrm{sel}}+W_{\mathrm{eff}})$ FLOP. The quadratic term in prefill sits in the Full-layer indexer (§4).
- **Chunked prefill.** The baseline uses `--chunked-prefill-size 4096`. Apply the table per chunk with $P_0=\text{chunk index}\cdot C$ and $T$ = tokens in the chunk. That value is the GEMM M that decides $b_w(M)$, and it also sets the bounded-replay tail when replay is on.
- **#14.** For large $T$ every local expert is hit ($E_r=48$), so routed weight bytes per layer per rank saturate at $48\cdot 3\cdot 2304\,d\,b_e$.

## 4. Delta tables for the other modes

The rows go in after row #3 of the Reuse table; attention row #6 then uses this layer's own idx.

### 4a. Full mode: layers 2, 8, 14 ($m=2$) and layer 20 ($m=1$)

$\gamma_m = 1+[m=2]$: the gate is there only at ratio 2. Decode emits a compressed entry every $m$ tokens.

| # | op | kind | weight | FLOPs, decode | FLOPs, prefill | W read | A / KV read | Wr |
|---|---|---|---|---|---|---|---|---|
| F1 | compressor projection (kv ‖ gate), replicated, FP32 out | GEMM | $\gamma_m\times$ `[512,d]` BF16 | $2Td\cdot 512\,\gamma_m$ | same | $\gamma_m\cdot 512\,d\cdot 2$ | $Td\,a$ | $T\cdot 512\,\gamma_m\cdot 4$ |
| F2 | pool $m$ tokens (per-channel softmax over the gate), RMSNorm(512), RoPE, quant + store | elementwise / norm | `[512]` | $O(T\cdot 512)$ | same | ~0 | $T\cdot 512\,\gamma_m\cdot 4$ (+ the ratio-2 pending-pair state) | $\lceil T/m\rceil\,b_{kv}$ |
| F3 | indexer K: `wk`, `k_norm`, RoPE, FP4 quant + store | GEMM + elementwise | `[128,512]` BF16 | $2\lceil T/m\rceil\cdot 512\cdot 128$ | same | $128\cdot 512\cdot 2$ | $\lceil T/m\rceil\cdot 512\,a$ | $\lceil T/m\rceil\,b_{ik}$ |
| F4 | indexer Q: `wq_b_I` on q_lat (all 32 heads on every rank), RoPE, FP4 fake-quant | GEMM | `[4096,1280]` FP8 | $2T\cdot 1280\cdot 4096$ | same | $4096\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | $T\cdot 4096\,a$ |
| F5 | `weights_proj` | GEMM | `[32,d]` BF16 | $2Td\cdot 32$ | same | $32\,d\cdot 2$ | $Td\,a$ | $T\cdot 32\,a$ |
| F6 | index score $\sum_h w_h\,\mathrm{relu}(q_h\cdot k)$ over all visible entries | attention-like (GEMM) | – | $2\cdot 32\cdot 128\,T N_c$ (+ $2\cdot 32\,T N_c$) | $8192\,B\sum_i\lceil (P_0+i)/m\rceil\approx 8192\,B\,(P_0 S+S^2/2)/m$ | – | KV: decode $T N_c\,b_{ik}$; prefill $B N_c\,b_{ik}\lceil S/\beta_q\rceil$ ($\beta_q$ = query tile, kernel-specific) | logits $T N_c\cdot 4$ if materialized |
| F7 | top-512 over $N_c$ | elementwise (select) | – | $O(T N_c)$ | $O(B\sum_i N_c(i))$ | – | $T N_c\cdot 4$ (if materialized) | $T\cdot 512\cdot 4$ |
| F8 | **layer 20 only**: block max over 8-position blocks, top-2048 blocks, candidate table | elementwise (select) | – | $O(T N_c)$ | $O(B\sum_i N_c(i))$ | – | $T N_c\cdot 4$ | $T\cdot 2048\cdot 4$ |

In layer 20 the compressor's input is the hidden state entering the layer, that is, the encoder output $H_{20}$. This is CED: the decoder's global KV is $f(H_{20})$.

### 4b. Reindex mode: layers 24, 28, 32, 36 ($m=1$, pool from layer 20)

| # | op | kind | FLOPs | W read | A / KV read | Wr |
|---|---|---|---|---|---|---|
| R1 | indexer Q (= F4) | GEMM | $2T\cdot 1280\cdot 4096$ | $4096\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | $T\cdot 4096\,a$ |
| R2 | `weights_proj` (= F5) | GEMM | $2Td\cdot 32$ | $32\,d\cdot 2$ | $Td\,a$ | $T\cdot 32\,a$ |
| R3 | score over the candidate pool | attention-like | $8192\,T P_c$ (prefill $8192\,B\sum_i\min(16384,P_0+i)$) | – | KV: $T P_c\,b_{ik}$ (gathered through the candidate table, which costs another $T\cdot 2048\cdot 4$) | |
| R4 | top-512 within the pool | select | $O(T P_c)$ | – | | $T\cdot 512\cdot 4$ |

### 4c. SWA-only: layers 0, 1

Remove the idx read from row #6. The attention keys are the $W_{\mathrm{eff}}$ SWA entries only: FLOPs $2048\,h_k T W_{\mathrm{eff}}$, KV $T W_{\mathrm{eff}}\,b_{kv}$ (decode).

### 4d. Engram: before layers 1 and 14

| # | op | kind | input | weight | output | FLOPs | W read | A read | Wr |
|---|---|---|---|---|---|---|---|---|---|
| G1 | n-gram hash (orders 2, 3, 4 × 8 heads = 24 ids per token) | elementwise (int) | input_ids | – | `[T,24]` | $O(24\,T)$ | – | $T\cdot 4\cdot 8$ | $T\cdot 24\cdot 8$ |
| G2 | table gather (FP8 rows of 256 + E8M0 per 32) from the GPU-sharded table | gather | `[T,24]` | rows `[~384M,256]` | `[T,6144]` | – | $T\cdot 24\cdot 264=6336\,T$ (spread over the shards) | – | $T\cdot 6144\,a$ |
| G3 | **COMM: sharded-table exchange** (placeholder) | comm | `[T,6144]` | | | | | | |
| G4 | `wkv` (replicated) | GEMM | `[T,6144]` | `[25600,6144]` FP8 | `[T,5·d]`: 4 per-stream keys + 1 value | $2T\cdot 6144\cdot 25600$ | $25600\cdot 6144\,b_w(T)$ | $T\cdot 6144\,a$ | $T\cdot 5d\,a$ |
| G5 | gate per stream: $\sigma(\mathrm{norm}(X_s)\cdot\mathrm{norm}(k_s))$, add the gated v to $X_s$ | elementwise | X `[T,4,d]`, kv | `q_weight` / `k_weight` `[4,d]` | X `[T,4,d]` | $O(T\cdot 4d)$ | $2\cdot 4d\cdot 2$ | $T\cdot 4d\,a+T\cdot 5d\,a$ | $T\cdot 4d\,a$ |

### 4e. The two ends of the model (once per step)

| # | op | kind | FLOPs | W read | Wr / comm |
|---|---|---|---|---|---|
| M1 | embedding lookup + mHC expand | gather | – | $T d\cdot 2$ (rows touched) | $T\cdot 4d\,a$; **COMM: vocab-parallel all-reduce** (placeholder) |
| M2 | mHC collapse + final RMSNorm | elementwise | $O(T_{\mathrm{out}}\cdot 4d)$ | – | $T_{\mathrm{out}}\,d\,a$ |
| M3 | LM head (vocab-parallel) on $T_{\mathrm{out}}$ rows | GEMM | $2\,T_{\mathrm{out}}\,d\cdot 129280/\mathrm{TP}$ | $129280\,d\cdot 2/\mathrm{TP}$ | **COMM: logits gather** (placeholder) |

Here $T_{\mathrm{out}}=B$ for prefill (the last token) and for decode, and $B(\gamma+1)$ for verify.

## 5. Watch-outs and source discrepancies

1. **"Dense vs MoE" does not exist here.** All 40 layers are MoE: there is no `first_k_dense_replace`, and there are experts on layers 0–39. There are no hash-routed layers either. V4 and V4-Flash used hash routing on their first 3 MoE layers; that does not carry over. Classify layers by attention mode (§2) instead.
2. **No MLA absorption question.** There is no `kv_b` up-projection. `wq_b` maps straight into the 512-d latent (64 × 512), K and V are the same latent (MQA, one KV head), and inverse RoPE is applied to the output before `wo_a`. Prefill and decode use the same math. The absorbed vs. non-absorbed trade-off from V3 does not apply.
3. **KV bytes on H20 differ from the report.**
   - SGLang's `select_dsv4_kv_layout()` picks the **V4 layout on SM90**: 584 B per entry (448 FP8 + 64 BF16 + 8 B of scales). The FP4 V4.1 layout (288 B per entry) is SM100/SM103 only.
   - Indexer K is forced to FP4: 68 B.
   - Global KV per context token = $3\cdot(584+68)/2+(584+68)=1630$ B, which matches the ~1.6 KB in my deployment notes. On the report's FP4 design it would be ~890 B.
   - SWA is per request: $40\cdot\min(128,L)\cdot 584\approx 2.99$ MB at the full window.
4. **KV is replicated across attention TP.** There is one latent head, so all 8 ranks hold and read the full cache; per-rank KV bytes do not shrink with TP. DP attention (off in the baseline) is the lever that changes this.
5. **Storage is not traffic.**
   - Only the 4 kv_src layers store global KV, but **all 38 CSA2 layers each gather their own $K_{\mathrm{sel}}\cdot 584$ B** per query. In decode ($L\ge 1024$) that is $38\cdot 512\cdot 584\approx 11.4$ MB per request per step.
   - Reuse mode saves indexer compute, not gather bytes.
   - The indexer scan adds $(3\lceil L/2\rceil+L)\cdot 68+4\min(16384,L)\cdot 68$ B per request per step. That is $\approx 170L+4.46$ MB for $L\ge 16$K, and it overtakes the gather + SWA (~14.4 MB) at about $L\approx 58$K. It grows with $L$ only through Full layers 2, 8, 14 and 20.
6. **Sparse top-k takes attention bytes and FLOPs from $O(L)$ to $O(\min(512,L/m)+128)$.** Only the indexer is $O(L)$ in decode and $O(S^2/m)$ in prefill (Full layers). Layer 20 scans at $m=1$, so its scan costs twice that of each encoder Full layer.
7. **FlashMLA head padding on SM90.** With TP > 1, `_kernel_num_heads` pads the 8 real heads to 64 and fills the rest with zeros. The executed attention FLOPs are 8× the useful ones.
   - Executed decode AI = $2048\,h_k/584\approx 224$ FLOP/B at $h_k=64$.
   - Useful AI = $2048\cdot 8/584\approx 28$ FLOP/B at $h=8$.
   - The roofline has to use the executed value.
8. **The indexer is replicated across TP.** All 32 heads run on every rank, and the indexer `wq_b` (FP8 `[4096,1280]`) is replicated. Its FLOPs and bytes are not divided by TP. `wq_a` and `wkv` are replicated too.
9. **mHC traffic per sublayer** (two sublayers per layer): ideal Single-Pass $(2n+2)d=10d$ elements per token, two-pass $(3n+2)d=14d$, V4 multi-kernel $(4n+4)d=20d$. Which one the SM90 kernels actually achieve is unverified (in SGLang #42245 is merged; #40891, the SM90 small-batch path, is not). The coefficient GEMM is `[T,20480]` × `[20480,24]` in F32/TF32.
10. **Dense weight bytes depend on M.**
    - The Hopper BF16 cache (on by default) runs FP8 dense linears from a BF16 copy when the actual M ≥ 64, so weight bytes double and the MMA is BF16. Below 64 they run as FP8 group-32 Triton.
    - **`wo_a` is always BF16.** It is dequantized at load, because the FP8 einsum path needs 128 × 128 blocks and the checkpoint uses 32 × 32.
11. **Routed experts.** They are stored as `I8` in safetensors but are packed MXFP4 (two per byte, plus an E8M0 scale per 32), so do not count them as INT8. On H20 they run W4A16 (CUTLASS, BF16 MMA) by default; W4A8 (FP8 MMA) is a flag, `--flashinfer-mxfp4-moe-precision fp8`. The per-expert effective batch is $Tk/E_{\mathrm{act}}$ with $E_{\mathrm{act}}=384\,(1-(1-6/384)^T)$. The shared expert is always hit, with $M=T$.
12. **The CED prefill saving is conditional.**
    - The report's "8B activated in prefill, halving prefill" assumes decoder SWA bounded replay. The baseline has it off, so all 40 layers run all $S$ tokens.
    - When it is on, SGLang's boundary is layer 21, not 20. Layer 20 (the decoder's Full layer) must run every token to build the decoder KV and the candidate pool.
13. **The decode indexer has a deployment trap.** On SM90 with DSpark static verify, the verify indexer scores the configured context capacity, not the visible length (SGLang #40945, not merged). There the scan cost uses `--context-length` instead of $L$.
14. **Naming traps.**
    - In SGLang, "main cache" is SWA and "extra cache" is the compressed C1/C2. The report's "main KV" is the compressed one.
    - `num_nextn_predict_layers: 3` is the **DSpark drafter**, not V3-style MTP (the report has no classic MTP).
    - `compress_ratios` has 43 entries; the last 3 are the drafter blocks.
15. **My earlier notes are about V4 / V4-Pro / V4-Flash, not V4.1-Flash.** They have CSA at $m=4$ plus HCA at $m=128$, 64 indexer heads, $d=4096$, 43 layers, 256 experts, q rank 1024 and hash routing. **None of these apply.** V4.1-Flash has pure CSA2 with $m\in\{2,1\}$, 32 indexer heads, $d=5120$, 40 layers, 384 experts and q rank 1280. My notes on the Sinkhorn / Birkhoff mathematics are relevant only to the mHC $B_l$ projection.
16. **There is no `hc_head` in the checkpoint.** With Single-Pass mHC, the final 4 → 1 collapse uses the last sublayer's A coefficients (`hc_combine`). Do not add an `hc_head` GEMM.
17. **Ratio-2 Full layers run the compressor GEMM on every token**, but emit a KV entry every 2 tokens. Storage and indexer-scan counts use $\lceil L/2\rceil$.
18. **RoPE differs by layer type.** Layers 0–1 use θ = 1e4. CSA2 layers use θ = 1.6e5 with YaRN ×16 (original 64K, β_fast 32, β_slow 1); it applies to Q, the SWA branch and the compressed KV. This has no effect on the roofline, but it matters when reproducing kernels.
19. **SWA capacity is not SWA traffic.**
    - **Traffic.** Each step, each request reads its live window: $40\min(128,L)\cdot 584$ B (≤ 2.99 MB).
    - **Capacity.** SGLang reserves a paged SWA pool of $R(W+E_I D+2p)+2C+p+4R(W+p)$ tokens × $40\cdot 584$ B (cap mode). Here $R$ is `max_running_requests`, $W=128$, $E_I$ the SWA eviction interval, $D$ the draft tokens (1 without speculation), $4R$ the prefix tails kept for radix reuse, $p$ the page size and $C$ the chunk size. That gives about 13 GiB at $R=256$, per the deployment notes, if $p\approx 256$.
    - **Without `--max-running-requests`.** SGLang falls back to ratio mode, which adds $0.1\cdot 40\cdot 584=2336$ B to *every* context token, more than the 1630 B of global KV. Set the flag explicitly.
20. **Engram is a capacity item, not a per-step bandwidth item.** The two FP8 tables take ~23.6 GiB per rank, sharded across the GPUs. A lookup is only ~6.3 KB per token per module, but it needs a cross-GPU exchange (placeholder G3). The replicated `wkv` GEMM (`[25600,6144]`) is a non-trivial ~157 MB weight read per module in small-batch decode.

## 6. Derived model-side quantities (no hardware)

| quantity | formula | value |
|---|---|---|
| global KV stored per context token (H20, per rank, replicated) | $3\cdot 652/2+652$ | 1630 B |
| the same on the report's FP4 design | $3\cdot(288+68)/2+(288+68)$ | 890 B |
| SWA stored per request | $40\cdot 128\cdot 584$ | 2.99 MB |
| decode KV gathered per request per step ($L\ge 1024$) | $38\cdot 512\cdot 584+40\cdot 128\cdot 584$ | 11.36 + 2.99 = 14.35 MB |
| decode indexer K scanned per request per step | $68\,(3\lceil L/2\rceil+L+4\min(16384,L))$ | $170L$ + 4.46 MB ($L\ge 16$K) |
| decode attention AI per key (executed / useful) | $2048\,h_k/584$, $2048\,h/584$ | 224 / 28 FLOP/B |
| indexer score AI per key | $2\cdot 32\cdot 128/68$ | 120 FLOP/B |
| activated params per token (decode) | attention 126.6M + MoE (6 + 1 experts) 247.7M per layer, × 40, + head + Engram `wkv` | ≈ 16.1B |
| per-rank dense weights read per decode step, $M<64$ (experts excluded; head shard and Engram `wkv` included) | sum of the W columns | ≈ 2.1 GB (layers 1.53 + Engram `wkv` 0.32 + head 0.17 + indexer / compressor 0.08) |
| per-rank routed-expert bytes per layer | $E_r\cdot 3\cdot 2304\cdot 5120\cdot 0.53125$ | $E_r\times$ 18.8 MB (saturates at 48: 0.90 GB) |
| BF16 cache per rank (cross-check against the 2.5 GiB in the deployment notes) | all FP8 dense linears at TP 8, shared expert sharded | 2.46 GiB |

## 7. Precision: checkpoint vs. what runs on H20

The checkpoint is FP8 with 32 × 32 blocks and UE8M0 scales, dynamic activation scales; the routed experts are MXFP4. What runs is the SM90 path at the pinned SGLang SHA with the baseline flags.

| component | checkpoint | runs on H20 as | TP |
|---|---|---|---|
| `wq_a`, `wkv` (SWA) | FP8 E4M3 | FP8 W8A8 group-32 Triton if M < 64, else BF16 `F.linear` from the BF16 cache | replicated |
| `wq_b` | FP8 E4M3 | as `wq_a` | column, 8 heads per rank |
| `wo_a` | FP8 E4M3 | BF16, dequantized at load | column, 1 group per rank |
| `wo_b` | FP8 E4M3 | as `wq_a` | row + all-reduce |
| compressor `wkv`, `wgate` | BF16 | BF16, FP32 accumulate and output | replicated |
| indexer `wq_b` | FP8 E4M3 | as `wq_a` | replicated |
| indexer `wk`, `weights_proj` | BF16 | BF16 | replicated |
| attention MMA | – | not verified (likely BF16 after dequant) | – |
| indexer score | – | not verified (the FP4 indexer flag is rejected on SM90) | – |
| router gate | BF16 | BF16 | replicated |
| shared expert | FP8 E4M3 | as `wq_a` | column / row, 288 per rank (inferred) |
| routed experts | MXFP4 | W4A16 (CUTLASS, BF16 MMA) by default; W4A8 (FP8 MMA) with the flag | EP 8, 48 experts per rank |
| Engram tables | FP8 E4M3 + E8M0 per 32 | same | rows sharded across the GPUs |
| Engram `wkv` | FP8 E4M3 | as `wq_a` (inferred to be in the BF16 cache) | replicated |
| mHC `hc_*_fn` | F32 | F32/TF32 GEMM + fused pre/post kernels | replicated |
| embed, head | BF16 | BF16 | vocab-parallel (inferred) |

Bytes per parameter, scales included: FP8 block 32 is $1+1/1024$, BF16 is 2, MXFP4 is $0.5+1/32=0.53125$.

## 8. The rest of the model, briefly

- **Router.** Scores by sqrtsoftplus, `noaux_tc` top-k; the selection bias is `gate.bias` for text and `gate.bias_vl` for image tokens (F32 `[384]`, used only to select); the top-k weights are renormalized and scaled by 1.5. One shared expert with FFN dim 2304. SwiGLU with $|x|\le 10$.
- **mHC.** 4 streams, 20 Sinkhorn iterations, `hc_eps` 1e-6. The 24 mixing coefficients are $n(2+n)$ = A(4) + B(16) + C(4), from a `[24,20480]` F32 projection, two per layer (before attention, before FFN). Single-Pass: the final collapse reuses the last A.
- **Engram.** Two modules, before layers 1 and 14, on the residual streams. n-gram orders 2, 3, 4 with 8 hash heads each, head dim 256, so 6144 per token. The tables have 384,006,168 and 384,016,682 rows. `wkv` is `[25600,6144]` with $25600=(4+1)\cdot 5120$: four per-stream keys and one value. No short convolution.
- **Embedding and head.** Untied, both BF16 `[129280,5120]`. The final norm is RMSNorm(5120). `rms_norm_eps` is written as 1e-20 in the config.
- **Attention details.** 64 query heads, 1 KV head, head dim 512 (448 without RoPE + 64 RoPE), q LoRA rank 1280, 8 output groups of rank 1024, `wo_a` `[8,1024,4096]` as a block-diagonal grouped GEMM, `wo_b` `[5120,8192]`, a per-head F32 attention sink, sliding window 128. `wkv` for SWA is `[512,5120]` on all 40 layers.
- **DSpark drafter** (`mtp.*`, not loaded in the baseline). 3 SWA-only blocks with the backbone's MQA dimensions; MoE with 128 routed experts, top 3, one shared, FFN 2304; `main_proj` `[5120,15360]` FP8 over the hidden states of layers 37, 38 and 39; 5 draft positions, so the verify window is 6; a Markov head of rank 256 and a confidence head `[1,5376]`. On disk `mtp.*` is 7.93 GB. With DSpark on, the SWA cache grows by $3\cdot 128\cdot 584$ B per request (sizing in SGLang unverified).
- **Vision** (not on the text roofline path). DeepSeek-ViT with 32 layers, width 1024, 16 heads, MLP 2816, patch 14, a 3 × 3 pixel unshuffle, an aligner of `[5120,9216]` and `[5120,5120]` in BF16, at most 1024 image tokens.
- **KV cache, the rest.** The ratio-2 Full layers keep a small per-request ring of compress state (not sized here). The paged SWA pool pads its page stride to a multiple of 576 B. The V4.1 design persists only global KV to SSD, not SWA.

## Sources

"Notes" below are my private H20 deployment notes for this model on SGLang (cited by line, L…); "report" is the DeepSeek-V4.1-Flash technical report (by line, or by PDF page). SGLang is at the pinned SHA above.

<details>
<summary>Where each claim comes from</summary>

| claim | source |
|---|---|
| revision `2cba9e42…` | notes L19 |
| dimensions: hidden 5120, 40 layers, vocab 129280, 64 heads, head dim 512, q rank 1280, 8 output groups of 1024, window 128, top-k 512, indexer 32 × 128, experts 384 / top 6 / 2304, 4 mHC streams, 20 Sinkhorn iterations | `config.json`; report L1247–1276 |
| `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id`, 2048 blocks × 8, `engram_layer_ids` | `config.json`; report L1249–1258, L1266, Fig. 3 (p. 7) |
| tensor shapes and dtypes, 551.9B and 196.6B parameters, checkpoint size | safetensors headers, `model.safetensors.index.json`; report L403 |
| 16B activated in decode, "8B in prefill" | report L404 |
| no dense FFN, no hash routing | `config.json`, headers; report L1269–1270; my earlier V4 notes L145 (V4 hash routing) |
| mHC traffic $(2n+2)d$, $(3n+2)d$, $(4n+4)d$; Single-Pass | report L743–748, L742–743, L707–711, L729–736 |
| 24 mixing coefficients; collapse by `hc_combine`, no `hc_head` | SGLang `deepseek_v4.py` `make_hc_mixing_params`, `hc_pre_from_prev_sublayer`; headers |
| MQA with K = V, one KV head; inverse RoPE on the output | SGLang `deepseek_v4.py` `MQALayer` (RadixAttention, `num_kv_heads=1`), `fused_rope_inplace(..., inverse=True)` |
| `wo_a` as einsum `tgd,grd->tgr`; `wo_a` in BF16 | SGLang; `wo_a_fp8_gemm_enabled()` requires `weight_block_size == [128,128]` |
| heads padded to 64 | SGLang `MqaAttentionBase._kernel_num_heads` |
| RoPE per layer type | `config.json` `rope_theta`, `compress_rope_theta`, `rope_scaling`; SGLang `MqaAttentionBase` |
| compressor pooling; indexer K from the compressed latent | SGLang `dsv41_sparse.py` `DeepseekV41Compressor`; report L572–577, L576 |
| CED: layer 20 compresses the encoder output | report L482–489 eq. (1), L640–642 |
| indexer score, heads replicated, candidates only in layers after 20 | SGLang `dsv41_sparse.py` `DeepseekV41Indexer` (`scores`, docstring, `uses_candidates`); report L653 |
| only kv_src layers own compressed storage | SGLang `collect_sources_by_ratio` |
| V4 layout on SM90, 584 B per entry, page stride 576 B | SGLang `deepseek_v4_memory_pool.py` `select_dsv4_kv_layout()`, `kv_layout.py` `KVLayout.V4`, `KVLayout.page_bytes`; notes L142 (`SGLANG_DSV4_KV_LAYOUT=v4`) |
| indexer K 68 B, forced FP4 | SGLang `get_dsv4_indexer_bytes_per_token(128, fp4=True)`, `pool_configurator.py` `DSV4PoolConfigurator` |
| 1630 B per context token | SGLang `DSV4PoolConfigurator._compressed_bytes_per_full_token`; notes L146 |
| report's FP4 KV design, 288 B; SWA in FP8 | report L798–826 (L810–812, L824–825) |
| SWA pool reservation, cap mode | SGLang `pool_configurator.py` `compute_swa_request_cap`, `DSV4PoolConfigurator._resolve_swa_cap_tokens`, `_get_bytes_per_swa_token`; `SGLANG_SWA_EVICTION_INTERVAL` |
| ratio-mode fallback, 2336 B per token | SGLang `DSV4_DEFAULT_SWA_FULL_TOKENS_RATIO = 0.1`, `_get_bytes_per_full_token`; notes L146 |
| ratio-2 compress-state ring | SGLang `get_compress_state_ring_size(2)` |
| BF16 cache at M ≥ 64 | `SGLANG_OPT_HOPPER_BLOCK_FP8_BF16=1`; notes L142, L245 |
| 2.5 GiB BF16 cache, ~13 GiB SWA at R = 256, ~23.6 GiB Engram per rank, `mtp.*` not loaded without speculation | notes L146 |
| routed experts W4A16 by default, W4A8 by flag | notes L23, L43, L144, L257 |
| baseline: TP 8, EP 8, no DP attention; A2A none; bounded replay off, no speculation | notes L22, L144, L24 |
| DSpark flags `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`; verify window 6 | notes L279 |
| FP4 indexer flag rejected on SM90 | SGLang `serving_hook.py` L428 |
| mHC on SM90: #42245 merged, #40891 not | notes L318 |
| verify indexer scores the configured capacity (#40945 not merged) | notes L296 |
| "main cache" vs "extra cache" | notes L323 |
| `late_layer_start = max(kv_source) + 1 = 21` | SGLang |
| Engram before the layer, on the residual; n-gram orders, 8 heads, tables, no short conv | SGLang `DeepseekV4Model.forward` L4067–4087; report L762, L764, L765, L756–757 |
| selection bias `bias` / `bias_vl` | report L455–463 |
| DSpark drafter; no classic MTP | report L776–796, L782–783, L425–427 |
| vision tower, 3 × 3 unshuffle | `config.json` `vision_config`; report L1274–1276, L449 |
| global KV persisted to SSD, SWA not | report L1095–1106 |
| shared expert TP-sharded; Engram `wkv` in the BF16 cache; embed / head vocab-parallel | inferred (the BF16-cache sum; standard SGLang) |

</details>
