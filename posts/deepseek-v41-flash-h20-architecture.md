---
title: DeepSeek-V4.1-Flash：结构和算子顺序（施工中）
date: 2026-11-17
summary: 在 8×H20 上给 DeepSeek-V4.1-Flash 算边之前的第一步：只列结构、形状和每个算子的 FLOPs 与字节公式，不带硬件数字，还没有预测和实测。
---

> **施工中。** 这是本栏目给这只模型做规划的第一步，只管模型这一侧：形状，以及写成符号公式的 FLOPs 和字节数。没有硬件数字（不带带宽，不带 FLOP/s），也没有时间。预测还没写，也还没量过。这里的东西都还没在 H20 上验证。下面每条缓存条目、每个 token 的字节数，说的是 SGLang 在 H20 上选的布局，来自模型和代码，不是从卡上量的。
>
> 往图上放任何数之前，要先确认三件事：
>
> - **基线里的 MoE combine。** 是不是一次融合的 all-reduce，把路由专家和共享专家的部分和一起加起来，A2A 设为 none？这是推出来的，要抓 trace 看。
> - **共享专家按 TP 切，每卡 288。** 这也是推出来的：只有这种切法能对上 2.5 GiB 的 BF16 缓存。共享专家要是每卡复制一份，会是 4.8 GiB。
> - **SM90 上注意力和索引器的 MMA 精度。** 注意力大概是先把 FP8 反量化成 BF16 再做 BF16 MMA，但没验证；它决定执行出来的 224 FLOP/B 要和哪一个脊点比。索引器打分那边，SM90 不接受 FP4 索引器的开关，所以打分 GEMM 实际用什么类型还不知道。
>
> 另外还没验证的：
>
> - SWA 池的页大小（约 256，是从 13 GiB 这个数倒推的）。
> - SM90 的 kernel 实际做到 mHC 流量的哪一档。
> - 残差流的数据类型（先按 BF16 算）。
> - `q_head_norm` 的默认值（影响可以忽略）。
> - FlashMLA 在 prefill 里稀疏 gather 能复用多少（表里给的是上界）。
> - DSpark 草稿器的 SWA 缓存怎么分配。
> - Engram 分片查表时怎么通信。

维度来自 HF 上的 `config.json`，以及全部 48 个 safetensors 分片的头（96,085 个张量的类型和形状），版本是 `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`，2026-11-17 取的。部署上的语义来自 SGLang，SHA `3e13f1b2c725d773b4dd845d12bd82903a7dba3f`。技术报告只用来看设计意图。基线部署是我 H20 部署笔记里的那一套：8×H20，TP 8，EP 8，不开 DP attention，MoE 的 A2A 是 none，编码器和解码器的 bounded replay 都关，不开投机解码。

和报告对一下。从分片头加起来的参数量，主干是 551.9B（报告：552B），Engram 的表是 196.6B（报告：196B）。按这些维度算出的激活参数，decode 是 16.1B（报告：16B），只算编码器是 7.9B（报告说"prefill 8B"）。TP 8 下的 BF16 缓存是每卡 2.46 GiB（部署笔记：2.5 GiB）。checkpoint 一共 510,286,023,000 字节。最大位置是 1,048,576。

## 0. 符号

| 符号 | 含义 | 取值（基线 8×H20） |
|---|---|---|
| $B$ | 这一步里的请求数 | – |
| $S$ | 每个请求新进来的 token 数（prefill） | – |
| $L$ | 一个 query 能看到的上下文：缓存的前缀加新 token，包括它自己 | – |
| $T$ | 这一步的 token 数：prefill 是 $B S$，decode 是 $B$，DSpark 验证是 $B(\gamma+1)=6B$ | – |
| $d$ | 隐藏维 | 5120 |
| $n$ | mHC 的流数 | 4 |
| $H$, $h$ | query 头数，总数 / 每个注意力 TP rank | 64 / 8 |
| $h_k$ | FlashMLA kernel 实际算的头数（TP > 1 时 SM90 补到 64） | 64 |
| $D$, $r$ | latent 头维 / 其中的 RoPE 维 | 512 / 64 |
| $q_r$ | q 的 LoRA 秩 | 1280 |
| $G$, $g$ | 输出分组，总数 / 每卡 | 8 / 1 |
| $o_r$ | 每组的输出秩 | 1024 |
| $m$ | 这一层的压缩比（0 表示只有 SWA） | 2（第 2–19 层），1（第 20–39 层） |
| $N_c$ | 这一层 kv_src 里可见的压缩条目数：$\lceil L/m\rceil$ | – |
| $K_{\mathrm{sel}}$ | 选中的条目数：$\min(512, N_c)$ | – |
| $W_{\mathrm{eff}}$ | SWA 的 key 数：$\min(128, L)$ | – |
| $P_c$ | Reindex 的搜索范围：$\min(16384, N_c)$ | – |
| $H_I$, $D_I$ | 索引器的头数 / 头维 | 32 / 128 |
| $E$, $k$, $f$ | 路由专家数 / top-k / 专家 FFN 维 | 384 / 6 / 2304 |
| TP, EP | 张量并行 / 专家并行 | 8 / 8 |
| $E_{\mathrm{act}}$ | 一层里被命中的不同专家数：$E\,(1-(1-k/E)^T)$（假设路由均匀） | – |
| $E_r$ | 一个 EP rank 上被命中的不同专家数：$E_{\mathrm{act}}/\mathrm{EP}$ | – |
| $T_r$ | 落到一个 rank 上的（token，专家）对数：$\approx T k/\mathrm{EP}$ | – |
| $a$ | 激活每个元素的字节数（按 BF16 算） | 2 |
| $b_w(M)$ | 稠密权重每个参数的字节数，是 GEMM 的 M 的函数：$M<64$ 时是 $1+1/1024$（FP8，块 32），否则是 $2$（Hopper 的 BF16 缓存） | – |
| $b_e$ | 路由专家每个参数的字节数（MXFP4，每 32 个一个 E8M0） | 0.53125 |
| $b_{kv}$ | 一条 KV（SGLang 的 V4 布局：448 FP8 + 64 BF16 + 8 B scale） | 584 |
| $b_{ik}$ | 一条索引器 K（FP4，每 32 个一个 scale） | 68 |

除非标了"全局"，公式都是**每张卡**的。在 TP 上复制的东西*不*除以 TP：`wq_a`、`wkv`、压缩器、整个索引器、router、Engram 的 `wkv`，还有 KV 缓存本身。

## 1. 图

### 1a. 整个模型

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 642" role="img" aria-labelledby="v41-model-zh-title v41-model-zh-desc" font-size="11"><title id="v41-model-zh-title">DeepSeek-V4.1-Flash 整个模型</title><desc id="v41-model-zh-desc">从上到下：input_ids（加图像 patch）经过按词表切分的 embedding 和 mHC 展开，变成 4 条流。第 0、1 层只有 SWA，FFN 是 MoE，第 1 层之前有 Engram。编码器是第 2 到 19 层，三组，每组一个压缩比 2 的 CSA2 Full 层加五个 Reuse 层，kv 来源是第 2、8、14 层，第 14 层之前有 Engram。编码器的输出 H_20 进入第 20 层，也就是解码器的 Full 层，压缩比 1：它的压缩器从 H_20 建出解码器的全局 KV，索引器扫全部 L，并建出候选池。之后第 21 到 23 层是 Reuse；第 24 到 39 层四组，每组一个只在候选池里重排的 Reindex 层加三个 Reuse 层；再用最后一组 A 系数做 mHC 合流、RMSNorm，最后是 LM head。旁边几样：视觉编码器把图像 token 送进 embedding；每一层都有 128 条的 SWA 环；第 2、8、14、20 层把压缩 KV 和索引器 K 写进全局缓存，H20 上每个 token 1630 字节；DSpark 草稿器在基线里关着，读第 37 到 39 层的隐状态，给 head 提草稿。</desc><rect class="flow-opt" x="238" y="405" width="116" height="166" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="296" y="422.0">DSpark 草稿器</tspan><tspan x="296" y="436.0" class="flow-quiet">（基线里关着）</tspan><tspan x="296" y="450.0">读 L37、L38、L39</tspan><tspan x="296" y="464.0">的隐状态</tspan><tspan x="296" y="478.0">main_proj</tspan><tspan x="296" y="492.0">15360→5120</tspan><tspan x="296" y="506.0">3 个只有 SWA 的块</tspan><tspan x="296" y="520.0">（MoE 128 选 3）</tspan><tspan x="296" y="534.0">5 个草稿位置</tspan><tspan x="296" y="548.0">Markov head r=256</tspan><tspan x="296" y="562.0">置信度 head</tspan></text><path class="flow-store" d="M238 252V348A58 4 0 0 0 354 348V252A58 4 0 0 0 238 252Z"/><path class="flow-store-rim" d="M238 252A58 4 0 0 0 354 252"/><text text-anchor="middle"><tspan x="296" y="273.0">全局 KV 缓存</tspan><tspan x="296" y="287.0">1630 B/token</tspan><tspan x="296" y="301.0" class="flow-quiet">（H20 上）</tspan><tspan x="296" y="315.0" class="flow-quiet">← L2/L8/L14 (m=2)</tspan><tspan x="296" y="329.0" class="flow-quiet">← L20 (m=1)</tspan><tspan x="296" y="343.0" class="flow-quiet">各带索引器 K</tspan></text><path class="flow-store" d="M238 176V244A58 4 0 0 0 354 244V176A58 4 0 0 0 238 176Z"/><path class="flow-store-rim" d="M238 176A58 4 0 0 0 354 176"/><text text-anchor="middle"><tspan x="296" y="197.0">SWA 缓存，</tspan><tspan x="296" y="211.0">40 层都有</tspan><tspan x="296" y="225.0" class="flow-quiet">每层每请求</tspan><tspan x="296" y="239.0" class="flow-quiet">环 128 × 584 B</tspan></text><rect class="flow-opt" x="238" y="6" width="116" height="68" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="296" y="23.0">DeepSeek-ViT</tspan><tspan x="296" y="37.0">32 层，d=1024</tspan><tspan x="296" y="51.0">+ 3×3 unshuffle</tspan><tspan x="296" y="65.0">+ aligner MLP</tspan></text><rect class="flow-box" x="6" y="610" width="220" height="26" rx="3"/><text text-anchor="middle"><tspan x="116" y="627.0">head 129280×5120 BF16 → logits</tspan></text><rect class="flow-box" x="6" y="552" width="220" height="40" rx="3"/><text text-anchor="middle"><tspan x="116" y="569.0">mHC 合流（hc_combine，用最后的 A）</tspan><tspan x="116" y="583.0">→ RMSNorm</tspan></text><rect class="flow-box" x="6" y="494" width="220" height="40" rx="3"/><text text-anchor="middle"><tspan x="116" y="511.0">L24–39：4 × [Reindex（限在候选池）</tspan><tspan x="116" y="525.0">+ 3 × Reuse] + MoE</tspan></text><rect class="flow-box" x="6" y="450" width="220" height="26" rx="3"/><text text-anchor="middle"><tspan x="116" y="467.0">L21–23 Reuse</tspan></text><rect class="flow-box" x="6" y="350" width="220" height="82" rx="3"/><text text-anchor="middle"><tspan x="116" y="367.0">L20 CSA2 (m=1) Full：</tspan><tspan x="116" y="381.0">压缩器从 H_20 建出</tspan><tspan x="116" y="395.0">解码器的全局 KV</tspan><tspan x="116" y="409.0">+ 索引器扫全部 L</tspan><tspan x="116" y="423.0">+ 候选池（2048 块 × 8）</tspan></text><rect class="flow-box" x="6" y="244" width="220" height="82" rx="3"/><text text-anchor="middle"><tspan x="116" y="261.0">编码器 L2–19：</tspan><tspan x="116" y="275.0">3 × [CSA2 (m=2) Full + 5 × Reuse]</tspan><tspan x="116" y="289.0">+ MoE</tspan><tspan x="116" y="303.0" class="flow-quiet">（kv_src 2/8/14；</tspan><tspan x="116" y="317.0" class="flow-quiet">L14 之前有 Engram）</tspan></text><rect class="flow-box" x="6" y="182" width="220" height="40" rx="3"/><text text-anchor="middle"><tspan x="116" y="199.0">L0–1 只有 SWA + MoE</tspan><tspan x="116" y="213.0" class="flow-quiet">（L1 之前有 Engram）</tspan></text><rect class="flow-box" x="6" y="138" width="220" height="26" rx="3"/><text text-anchor="middle"><tspan x="116" y="155.0">mHC 展开 → X [T,4,d]</tspan></text><rect class="flow-box" x="6" y="80" width="220" height="40" rx="3"/><text text-anchor="middle"><tspan x="116" y="97.0">embed 129280×5120 BF16</tspan><tspan x="116" y="111.0">（按词表切分）→ x [T,d]</tspan></text><rect class="flow-box" x="6" y="6" width="220" height="26" rx="3"/><text text-anchor="middle"><tspan x="116" y="23.0">input_ids [B,S]（+ 图像 patch）</tspan></text><path class="flow-edge" d="M116 32L116 80"/><path class="flow-head" d="M116 80L112.8 74.0L119.2 74.0Z"/><path class="flow-edge" d="M116 120L116 138"/><path class="flow-head" d="M116 138L112.8 132.0L119.2 132.0Z"/><path class="flow-edge" d="M116 164L116 182"/><path class="flow-head" d="M116 182L112.8 176.0L119.2 176.0Z"/><path class="flow-edge" d="M116 222L116 244"/><path class="flow-head" d="M116 244L112.8 238.0L119.2 238.0Z"/><path class="flow-edge" d="M116 326L116 350"/><path class="flow-head" d="M116 350L112.8 344.0L119.2 344.0Z"/><path class="flow-edge" d="M116 432L116 450"/><path class="flow-head" d="M116 450L112.8 444.0L119.2 444.0Z"/><path class="flow-edge" d="M116 476L116 494"/><path class="flow-head" d="M116 494L112.8 488.0L119.2 488.0Z"/><path class="flow-edge" d="M116 534L116 552"/><path class="flow-head" d="M116 552L112.8 546.0L119.2 546.0Z"/><path class="flow-edge" d="M116 592L116 610"/><path class="flow-head" d="M116 610L112.8 604.0L119.2 604.0Z"/><text class="flow-quiet" x="121" y="341" text-anchor="start" font-size="10">H_20 = 编码器输出</text><path class="flow-edge flow-edge-opt" d="M296 74L296 100L226 100"/><path class="flow-head flow-head-opt" d="M226 100L232.0 96.8L232.0 103.2Z"/><text class="flow-quiet" x="292" y="113" text-anchor="end" font-size="10">图像 token</text><path class="flow-edge flow-edge-opt" d="M226 202L238 202"/><path class="flow-head flow-head-opt" d="M238 202L232.0 205.2L232.0 198.8Z"/><path class="flow-edge flow-edge-opt" d="M226 285L238 285"/><path class="flow-head flow-head-opt" d="M238 285L232.0 288.2L232.0 281.8Z"/><path class="flow-edge flow-edge-opt" d="M226 391L252 391L252 349"/><path class="flow-head flow-head-opt" d="M252 349L255.2 355.0L248.8 355.0Z"/><path class="flow-edge flow-edge-opt" d="M226 514L238 514"/><path class="flow-head flow-head-opt" d="M238 514L232.0 517.2L232.0 510.8Z"/><path class="flow-edge flow-edge-opt" d="M296 571L296 623L226 623"/><path class="flow-head flow-head-opt" d="M226 623L232.0 619.8L232.0 626.2Z"/><text class="flow-quiet" x="292" y="619" text-anchor="end" font-size="10">草稿</text></svg>
<figcaption>整个模型。虚线：图像那条路、DSpark 草稿器（基线里关着），以及往缓存里写。</figcaption>
</figure>

如果打开解码器的 SWA bounded replay，第 21–39 层只处理每个 prompt 的最后 128 个 token。基线里它是关的，所以 40 层都处理全部 $S$ 个 token。

### 1b. 一个块：第 $l$ 层，CSA2 加 MoE

为了在手机上看得清，一个块拆成两张图。虚线框是 Full 模式多出来的。

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 936" role="img" aria-labelledby="v41-block-attn-zh-title v41-block-attn-zh-desc" font-size="11"><title id="v41-block-attn-zh-title">一个块的注意力半边</title><desc id="v41-block-attn-zh-desc">第 l 层，CSA2 注意力。4 条 mHC 流 X_l 和上一子层的输出进入 mHC 边界：更新残差，用 20 次 Sinkhorn 算出 24 个系数，混合输入，做 attn_norm，得到 x。一个复制、融合的 GEMM 同时算 wq_a 和 wkv。wkv 经过 kv_norm 和 RoPE 写进 SWA 环；wq_a 经过 q_norm 得到 query latent，再经 wq_b 得到每卡 8 个头，加 RoPE，补零到 64 个头。虚线框只在 Full 模式里跑：压缩器把 m 个 token 池化成一条压缩条目，索引器 K 投影它，再算索引器权重和索引器 Q，打分是按头加权的 relu(q·k) 之和，扫所有可见条目，留下前 512 个下标。Reuse 层读它的 topk_src 那一层的下标。FlashMLA 稀疏 MQA 对选中的条目加 SWA 窗口做注意力，然后是逆 RoPE、BF16 的分组 wo_a、行并行的 wo_b，最后 all-reduce。</desc><path class="flow-comm" d="M94 904H266L274 917L266 930H94L86 917Z"/><text text-anchor="middle"><tspan x="180" y="921.0">all-reduce [T,d]</tspan></text><rect class="flow-box" x="66" y="862" width="228" height="26" rx="3"/><text text-anchor="middle"><tspan x="180" y="879.0">wo_b [d,g·1024]（行并行）</tspan></text><rect class="flow-box" x="66" y="806" width="228" height="40" rx="3"/><text text-anchor="middle"><tspan x="180" y="823.0">wo_a 分组 [g,1024,4096] BF16</tspan><tspan x="180" y="837.0">→ [T,g·1024]</tspan></text><rect class="flow-box" x="66" y="764" width="228" height="26" rx="3"/><text text-anchor="middle"><tspan x="180" y="781.0">对 o[…,−64:] 做逆 RoPE</tspan></text><rect class="flow-box" x="6" y="708" width="348" height="40" rx="3"/><text text-anchor="middle"><tspan x="180" y="725.0">FlashMLA 稀疏 MQA，K = V = latent，attn_sink</tspan><tspan x="180" y="739.0">keys = 取来的 K_sel 条 + W_eff 条 SWA → o [T,h,512]</tspan></text><rect class="flow-opt" x="198" y="638" width="156" height="54" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="655.0">idx [T,512]</tspan><tspan x="276" y="669.0" class="flow-quiet">（Reuse 层读</tspan><tspan x="276" y="683.0" class="flow-quiet">topk_src 那层的 idx）</tspan></text><rect class="flow-opt" x="198" y="540" width="156" height="82" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="557.0">打分 s =</tspan><tspan x="276" y="571.0">Σ_h w_h·relu(q_h·k)</tspan><tspan x="276" y="585.0">扫 N_c（Full）</tspan><tspan x="276" y="599.0">或 P_c（Reindex）</tspan><tspan x="276" y="613.0">→ 前 512 的 idx [T,512]</tspan></text><rect class="flow-opt" x="198" y="456" width="156" height="68" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="473.0">q_lat → 索引器 Q：</tspan><tspan x="276" y="487.0">wq_b_I [4096,1280]</tspan><tspan x="276" y="501.0" class="flow-quiet">（32 头全算，复制）</tspan><tspan x="276" y="515.0">→ [T,32,128]</tspan></text><rect class="flow-opt" x="198" y="406" width="156" height="40" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="423.0">x → weights_proj [32,d]</tspan><tspan x="276" y="437.0">→ w [T,32]</tspan></text><rect class="flow-opt" x="198" y="342" width="156" height="54" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="359.0">索引器 K：wk [128,512]</tspan><tspan x="276" y="373.0">→ k_norm，RoPE，FP4</tspan><tspan x="276" y="387.0">→ [T/m,128]（68 B）</tspan></text><rect class="flow-opt" x="198" y="216" width="156" height="110" rx="3"/><text class="flow-opt-text" text-anchor="middle"><tspan x="276" y="233.0">x → 压缩器：</tspan><tspan x="276" y="247.0">wkv（+wgate）[512,d] BF16</tspan><tspan x="276" y="261.0">→ 池化 m 个 token</tspan><tspan x="276" y="275.0" class="flow-quiet">（softmax gate）</tspan><tspan x="276" y="289.0">→ RMSNorm，RoPE</tspan><tspan x="276" y="303.0">→ [T/m,512]</tspan><tspan x="276" y="317.0">→ extra cache（584 B）</tspan></text><rect class="flow-box" x="6" y="216" width="172" height="54" rx="3"/><text text-anchor="middle"><tspan x="92" y="233.0">wq_b [h·512,1280]</tspan><tspan x="92" y="247.0">→ q [T,h,512] + RoPE</tspan><tspan x="92" y="261.0">补零到 h_k = 64</tspan></text><rect class="flow-box" x="6" y="174" width="172" height="26" rx="3"/><text text-anchor="middle"><tspan x="92" y="191.0">q_norm → q_lat [T,1280]</tspan></text><rect class="flow-box" x="198" y="118" width="156" height="68" rx="3"/><text text-anchor="middle"><tspan x="276" y="135.0">wkv → kv_norm</tspan><tspan x="276" y="149.0">+ RoPE（后 64 维）</tspan><tspan x="276" y="163.0">→ SWA KV [T,512]</tspan><tspan x="276" y="177.0">→ SWA 环（584 B/条）</tspan></text><rect class="flow-box" x="6" y="118" width="172" height="40" rx="3"/><text text-anchor="middle"><tspan x="92" y="135.0">wq_a [1280,d] + wkv [512,d]</tspan><tspan x="92" y="149.0" class="flow-quiet">（复制，融合成一个）</tspan></text><rect class="flow-box" x="6" y="48" width="348" height="54" rx="3"/><text text-anchor="middle"><tspan x="180" y="65.0">mHC 边界：残差更新 + 系数</tspan><tspan x="180" y="79.0">（24 = A4+B16+C4，Sinkhorn 20 次）</tspan><tspan x="180" y="93.0">+ 用 A_(l−1) 混合输入 + attn_norm → x [T,d]</tspan></text><rect class="flow-box" x="6" y="6" width="348" height="26" rx="3"/><text text-anchor="middle"><tspan x="180" y="23.0">X_l [T,4,d]（mHC 流）+ Y_(l−1) [T,d]</tspan></text><text class="flow-quiet" x="354" y="210" text-anchor="end" font-size="10">虚线：只有 Full 模式有</text><path class="flow-edge" d="M180 32L180 48"/><path class="flow-head" d="M180 48L176.8 42.0L183.2 42.0Z"/><path class="flow-edge" d="M92 102L92 118"/><path class="flow-head" d="M92 118L88.8 112.0L95.2 112.0Z"/><path class="flow-edge" d="M178 138L198 138"/><path class="flow-head" d="M198 138L192.0 141.2L192.0 134.8Z"/><path class="flow-edge" d="M92 158L92 174"/><path class="flow-head" d="M92 174L88.8 168.0L95.2 168.0Z"/><path class="flow-edge" d="M92 200L92 216"/><path class="flow-head" d="M92 216L88.8 210.0L95.2 210.0Z"/><path class="flow-edge" d="M92 270L92 708"/><path class="flow-head" d="M92 708L88.8 702.0L95.2 702.0Z"/><path class="flow-edge" d="M354 152L358 152L358 708"/><path class="flow-head" d="M358 708L354.8 702.0L361.2 702.0Z"/><path class="flow-edge flow-edge-opt" d="M276 326L276 342"/><path class="flow-head flow-head-opt" d="M276 342L272.8 336.0L279.2 336.0Z"/><path class="flow-edge flow-edge-opt" d="M198 369H189"/><path class="flow-edge flow-edge-opt" d="M198 426H189"/><path class="flow-edge flow-edge-opt" d="M198 490H189"/><path class="flow-edge flow-edge-opt" d="M189 369L189 581L198 581"/><path class="flow-head flow-head-opt" d="M198 581L192.0 584.2L192.0 577.8Z"/><path class="flow-edge flow-edge-opt" d="M276 622L276 638"/><path class="flow-head flow-head-opt" d="M276 638L272.8 632.0L279.2 632.0Z"/><path class="flow-edge" d="M276 692L276 708"/><path class="flow-head" d="M276 708L272.8 702.0L279.2 702.0Z"/><path class="flow-edge" d="M180 748L180 764"/><path class="flow-head" d="M180 764L176.8 758.0L183.2 758.0Z"/><path class="flow-edge" d="M180 790L180 806"/><path class="flow-head" d="M180 806L176.8 800.0L183.2 800.0Z"/><path class="flow-edge" d="M180 846L180 862"/><path class="flow-head" d="M180 862L176.8 856.0L183.2 856.0Z"/><path class="flow-edge" d="M180 888L180 904"/><path class="flow-head" d="M180 904L176.8 898.0L183.2 898.0Z"/></svg>
<figcaption>一个块的注意力半边。虚线是 Full 模式多出来的（第 2、8、14、20 层）。Reindex 层只留索引器 Q、weights_proj 和打分，扫的是候选池。Reuse 层这些都不做，直接读 topk_src 的 idx。</figcaption>
</figure>

<figure class="flow">
<svg class="flow-svg" viewBox="0 0 360 416" role="img" aria-labelledby="v41-block-moe-zh-title v41-block-moe-zh-desc" font-size="11"><title id="v41-block-moe-zh-title">一个块的 MoE 半边</title><desc id="v41-block-moe-zh-desc">注意力的 all-reduce 之后，是注意力和 FFN 之间的 mHC 边界，做 ffn_norm，得到 x2。router 是复制在每张卡上的 BF16 GEMM，用 sqrtsoftplus 打分、加选择用的 bias，选前 6 个专家，归一，乘 1.5。基线里 dispatch 是空的，另一种做法是 DeepEP 的 all-to-all。每张卡放 48 个路由专家，MXFP4，按 W4A16 跑：w1 和 w3、限幅 10 的 SwiGLU、w2。共享专家用 FP8，按 TP 切，每卡 288。combine 在基线里是一次 all-reduce，另一种是 all-to-all combine，得到 Y_l，交给下一个 mHC 边界。</desc><rect class="flow-box" x="66" y="384" width="228" height="26" rx="3"/><text text-anchor="middle"><tspan x="180" y="401.0">Y_l [T,d] → 下一个 mHC 边界</tspan></text><path class="flow-comm" d="M44 328H316L324 348L316 368H44L36 348Z"/><text text-anchor="middle"><tspan x="180" y="345.0">combine：all-reduce [T,d]（基线）</tspan><tspan x="180" y="359.0" class="flow-quiet">/ A2A combine</tspan></text><rect class="flow-box" x="6" y="230" width="204" height="82" rx="3"/><text text-anchor="middle"><tspan x="108" y="247.0">路由专家，每卡 48 个，</tspan><tspan x="108" y="261.0">MXFP4 W4A16：</tspan><tspan x="108" y="275.0">w1+w3 [2×2304,d]</tspan><tspan x="108" y="289.0">→ SwiGLU 限幅 10</tspan><tspan x="108" y="303.0">→ w2 [d,2304]</tspan></text><path class="flow-comm" d="M22 174H194L202 194L194 214H22L14 194Z"/><text text-anchor="middle"><tspan x="108" y="191.0">dispatch</tspan><tspan x="108" y="205.0" class="flow-quiet">（基线里没有 / DeepEP A2A）</tspan></text><rect class="flow-box" x="226" y="104" width="128" height="54" rx="3"/><text text-anchor="middle"><tspan x="290" y="121.0">共享专家</tspan><tspan x="290" y="135.0">FP8，按 TP 切</tspan><tspan x="290" y="149.0">每卡 288</tspan></text><rect class="flow-box" x="6" y="104" width="204" height="54" rx="3"/><text text-anchor="middle"><tspan x="108" y="121.0">router [384,d] BF16</tspan><tspan x="108" y="135.0">sqrtsoftplus + bias，选前 6，</tspan><tspan x="108" y="149.0">归一，×1.5</tspan></text><rect class="flow-box" x="6" y="48" width="348" height="40" rx="3"/><text text-anchor="middle"><tspan x="180" y="65.0">mHC 边界（attn 后 / ffn 前）</tspan><tspan x="180" y="79.0">+ ffn_norm → x2 [T,d]</tspan></text><path class="flow-comm" d="M94 6H266L274 19L266 32H94L86 19Z"/><text text-anchor="middle"><tspan x="180" y="23.0">all-reduce [T,d]（接上图）</tspan></text><path class="flow-edge" d="M180 32L180 48"/><path class="flow-head" d="M180 48L176.8 42.0L183.2 42.0Z"/><path class="flow-edge" d="M108 88L108 104"/><path class="flow-head" d="M108 104L104.8 98.0L111.2 98.0Z"/><path class="flow-edge" d="M290 88L290 104"/><path class="flow-head" d="M290 104L286.8 98.0L293.2 98.0Z"/><path class="flow-edge" d="M108 158L108 174"/><path class="flow-head" d="M108 174L104.8 168.0L111.2 168.0Z"/><path class="flow-edge" d="M108 214L108 230"/><path class="flow-head" d="M108 230L104.8 224.0L111.2 224.0Z"/><path class="flow-edge" d="M108 312L108 328"/><path class="flow-head" d="M108 328L104.8 322.0L111.2 322.0Z"/><path class="flow-edge" d="M290 158L290 328"/><path class="flow-head" d="M290 328L286.8 322.0L293.2 322.0Z"/><path class="flow-edge" d="M180 368L180 384"/><path class="flow-head" d="M180 384L176.8 378.0L183.2 378.0Z"/></svg>
<figcaption>一个块的 MoE 半边。两头尖的框是通信。</figcaption>
</figure>

## 2. 哪张表管哪一层

config 里**没有稠密 FFN 层**：40 层每一层都是 MoE，也没有哈希路由。真正改变算子列表的是**注意力模式**。第 3 节给出 **Reuse + MoE** 块的完整算子顺序，40 层里有 30 层是这种。第 4 节给出其他模式的增减。

| 模式 | 层 | 层数 | 相对 Reuse 增减了什么 |
|---|---|---|---|
| 只有 SWA | 0, 1 | 2 | 没有稀疏部分（不读 idx，$K_{\mathrm{sel}}=0$）。RoPE 用 θ = 1e4，不是 θ = 1.6e5 加 YaRN。 |
| Full，$m=2$ | 2, 8, 14 | 3 | + 压缩器（带 gate）+ 索引器 K + 索引器 Q + 在 $N_c=\lceil L/2\rceil$ 上打分 + top-k |
| Full，$m=1$ | 20 | 1 | + 压缩器（不带 gate）+ 索引器 K + 索引器 Q + 在 $N_c=L$ 上打分 + top-k + 建**候选池** |
| Reindex，$m=1$ | 24, 28, 32, 36 | 4 | + 索引器 Q + 在候选池 $P_c=\min(16384, L)$ 上打分 + top-k |
| Reuse | 3–7, 9–13, 15–19, 21–23, 25–27, 29–31, 33–35, 37–39 | 30 | （基准那张表） |
| + Engram（在这一层之前） | 1, 14 | 2 | + 哈希、查表、`wkv` GEMM、门控 |

每一层的 `kv_src` 是读哪一层的压缩 KV 和索引器 K，`topk_src` 是用哪一层的前 512 个下标。第 0–1 层两样都没有。第 2–7 层两样都读第 2 层，8–13 读第 8 层，14–19 读第 14 层。第 20–39 层的 KV 都读第 20 层；`topk_src` 在第 20–23 层是 20，之后每四层一组，依次是 24、28、32、36。只有这四个 `kv_src` 层自己存压缩 KV。Engram 作用在第 1 层和第 14 层*之前*的残差上。

**prefill 里每层的 token 数。** $l \le 20$ 时 $T_l = BS$。$l \ge 21$ 时，解码器 bounded replay 关着（基线）是 $T_l = BS$，开着是 $T_l = B\min(S,128)$。分界来自 SGLang 的 `late_layer_start = max(kv_source) + 1 = 21`，不是 $L/2$。

## 3. Reuse + MoE 块：完整算子顺序

各列：W 是权重字节，A 是读的激活字节，KV 是读的缓存字节，Wr 是写的字节。通信那几行只占位，不算大小。

### 3a. Decode（每个请求一个新 token，$T=B$，上下文 $L$）

| # | 算子 | 类型 | 输入 | 权重 | 输出 | FLOPs（每卡） | 读 W | 读 A | 读 KV | 写 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mHC 边界（上一层 FFN 之后 + 系数 + 混合 + attn_norm + 量化） | norm / 逐元素 + 小 GEMM | X `[T,4,d]`，Y_prev `[T,d]` | `hc_attn_fn` `[24,4d]` F32 | X′ `[T,4,d]`，x `[T,d]` | $2T\cdot 4d\cdot 24+O(T\cdot 4d\cdot 20)$ | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$（理想值，见 §5 第 9 条） | – | $T(n+1)d\,a$ |
| 2 | `wq_a` ‖ `wkv`（融合，复制） | GEMM | x `[T,d]` | `[1792,d]` FP8 | `[T,1280]`，`[T,512]` | $2Td\cdot 1792$ | $1792\,d\,b_w(T)$ | $Td\,a$ | – | $T\cdot 1792\,a$ |
| 3 | `q_norm`；`kv_norm` + RoPE + 量化 + 写 SWA | norm / 逐元素 | `[T,1280]`，`[T,512]` | `[1280]`，`[512]` | q_lat `[T,1280]`；一条 SWA | $O(T\cdot 1792)$ | ~0 | $T\cdot 1792\,a$ | – | $T\cdot 1280\,a+T\,b_{kv}$ |
| 4 | `wq_b`（列并行） | GEMM | q_lat `[T,1280]` | `[h·512,1280]` FP8 | q `[T,h,512]` | $2T\cdot 1280\cdot 512h$ | $512h\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | – | $Th\cdot 512\,a$ |
| 5 | RoPE(q[…,−64:]) + 补零到 $h_k$ | 逐元素 | q | – | q_pad `[T,h_k,512]` | ~0 | – | $Th\cdot 512\,a$ | – | $Th_k\cdot 512\,a$ |
| 6 | 稀疏 MQA 注意力（FlashMLA sparse decode，带 sink） | 注意力 | q_pad，topk_src 的 idx `[T,512]`，SWA 环 | – | o `[T,h_k,512]` | 执行 $2048\,h_k T(K_{\mathrm{sel}}+W_{\mathrm{eff}})$；有用 $2048\,hT(K_{\mathrm{sel}}+W_{\mathrm{eff}})$ | – | $Th_k\cdot 512\,a+T\cdot 512\cdot 4$ | $T(K_{\mathrm{sel}}+W_{\mathrm{eff}})\,b_{kv}$（每个 key 一条 latent，K = V） | $Th_k\cdot 512\,a$ |
| 7 | 切回 $h$ 个头，对 o[…,−64:] 做逆 RoPE | 逐元素 | o | – | `[T,h,512]` | ~0 | – | $Th\cdot 512\,a$ | – | $Th\cdot 512\,a$ |
| 8 | `wo_a`（分组，块对角） | GEMM（按 $g$ 批量） | `[T,g,4096]` | `[g,1024,4096]` **BF16** | `[T,g·1024]` | $2Tg\cdot 1024\cdot 4096$ | $g\cdot 1024\cdot 4096\cdot 2$ | $Tg\cdot 4096\,a$ | – | $Tg\cdot 1024\,a$ |
| 9 | `wo_b`（行并行） | GEMM | `[T,g·1024]` | `[d,g·1024]` FP8 | 部分和 `[T,d]` | $2Tg\cdot 1024\,d$ | $d\,g\cdot 1024\,b_w(T)$ | $Tg\cdot 1024\,a$ | – | $Td\,a$ |
| 10 | **通信：all-reduce（注意力 TP）** | 通信，占位 | `[T,d]` | – | `[T,d]` | – | – | – | – | – |
| 11 | mHC 边界（注意力之后 + FFN 之前 + ffn_norm） | norm / 逐元素 + 小 GEMM | X′，y | `hc_ffn_fn` `[24,4d]` F32 | X″，x2 `[T,d]` | 同 #1 | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$ | – | $T(n+1)d\,a$ |
| 12 | router（sqrtsoftplus，bias / bias_vl 选择，选前 6，重新归一，×1.5） | router | x2 | `[384,d]` BF16（复制） | idx `[T,6]`，w `[T,6]` | $2Td\cdot 384+O(T\cdot 384)$ | $384\,d\cdot 2$ | $Td\,a$ | – | $T\cdot 6\cdot 8$ |
| 13 | **通信：MoE dispatch**（基线里没有：token 本来就在每张卡上；另一种做法是 DeepEP A2A） | 通信，占位 | `[T,d]` | – | `[T_r,d]` | – | – | – | – | – |
| 14 | 路由专家 `w1` ‖ `w3`（在本卡 $E_r$ 个专家上分组） | GEMM（分组） | `[T_r,d]` | $E_r\times$ `[4608,d]` MXFP4 | `[T_r,4608]` | $2T_r d\cdot 4608$ | $E_r\cdot 4608\,d\,b_e$ | $T_r d\,a$ | – | $T_r\cdot 4608\,a$ |
| 15 | SwiGLU + 限幅 10 | 逐元素 | `[T_r,4608]` | – | `[T_r,2304]` | $O(T_r\cdot 2304)$ | – | $T_r\cdot 4608\,a$ | – | $T_r\cdot 2304\,a$ |
| 16 | 路由专家 `w2` | GEMM（分组） | `[T_r,2304]` | $E_r\times$ `[d,2304]` MXFP4 | `[T_r,d]` | $2T_r\cdot 2304\,d$ | $E_r\cdot 2304\,d\,b_e$ | $T_r\cdot 2304\,a$ | – | $T_r d\,a$ |
| 17 | 按 top-k 权重加起来（本卡命中的） | 逐元素 | `[T_r,d]`，w | – | 部分和 `[T,d]` | $O(T_r d)$ | – | $T_r d\,a$ | – | $Td\,a$ |
| 18 | 共享专家 `w1` ‖ `w3`（按 TP 切） | GEMM | x2 `[T,d]` | `[2f/TP,d]` FP8 | `[T,2f/TP]` | $2Td\cdot 2f/\mathrm{TP}$ | $(2f/\mathrm{TP})\,d\,b_w(T)$ | $Td\,a$ | – | $T(2f/\mathrm{TP})\,a$ |
| 19 | SwiGLU + 限幅 | 逐元素 | | – | `[T,f/TP]` | $O(Tf/\mathrm{TP})$ | – | $T(2f/\mathrm{TP})\,a$ | – | $T(f/\mathrm{TP})\,a$ |
| 20 | 共享专家 `w2` | GEMM | `[T,f/TP]` | `[d,f/TP]` FP8 | 部分和 `[T,d]` | $2T(f/\mathrm{TP})\,d$ | $d\,(f/\mathrm{TP})\,b_w(T)$ | $T(f/\mathrm{TP})\,a$ | – | $Td\,a$ |
| 21 | **通信：MoE combine**（基线：路由加共享的部分和做一次 all-reduce；DeepEP：A2A combine 再本地相加） | 通信，占位 | `[T,d]` | – | Y `[T,d]` | – | – | – | – | – |

关于 decode：

- **#6。** $K_{\mathrm{sel}}=\min(512,\lceil L/m\rceil)$，$m$ 取 kv_src 那一层的（编码器里 $m=2$，解码器里 $m=1$）。KV 流量按请求算，所以不会被 $B$ 摊薄。每个 TP rank 读的字节一样，因为唯一的那个 latent 头是复制的。
- **#14 和 #16。** 每个被命中的专家平均分到 $Tk/E_{\mathrm{act}}$ 个 token。$T$ 小的时候 $E_{\mathrm{act}}\approx kT$，专家 GEMM 就是每个专家一次 GEMV。EP 不保证均衡：最忙的那张卡定节奏，所以规划时用 $\max_{\text{rank}} E_r$，不用平均值。
- **DSpark 验证。** 把 $T=B$ 换成 $T=B(\gamma+1)$。这些 token 各有各的 $L$，所以 $K_{\mathrm{sel}}$ 和 $W_{\mathrm{eff}}$ 仍然按 token 算。

### 3b. Prefill（每个请求 $S$ 个新 token，因果；缓存的前缀长 $P_0$，所以第 $i$ 个 query 看到 $L_i=P_0+i$，$i=1,\dots,S$）

每个请求的两个和：

- $\Sigma_{\mathrm{sel}}(m)=\sum_{i=1}^{S}\min(512,\lceil (P_0+i)/m\rceil)$。$P_0=0$、$S\ge 512m$ 时约为 $512S-131072\,m$。
- $\Sigma_{\mathrm{win}}=\sum_{i}\min(128,P_0+i)$。$P_0=0$ 时约为 $128S-8192$。

| # | 算子 | 类型 | 输入 | 权重 | 输出 | FLOPs（每卡） | 读 W | 读 A | 读 KV | 写 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mHC 边界 + attn_norm | norm / 逐元素 | `[T,4,d]`，`[T,d]` | `[24,4d]` F32 | `[T,4,d]`，`[T,d]` | $2T\cdot 4d\cdot 24+O(T\cdot 4d\cdot 20)$ | $24\cdot 4d\cdot 4$ | $T(n+1)d\,a$ | – | $T(n+1)d\,a$ |
| 2 | `wq_a` ‖ `wkv` | GEMM | `[T,d]` | `[1792,d]` | `[T,1792]` | $2Td\cdot 1792$ | $1792\,d\,b_w(T)$（$T\ge 64$ 起是 BF16） | $Td\,a$ | – | $T\cdot 1792\,a$ |
| 3 | `q_norm`；`kv_norm` + RoPE + 量化 + 写 SWA | norm / 逐元素 | | | | $O(T\cdot 1792)$ | ~0 | $T\cdot 1792\,a$ | – | $T\cdot 1280\,a+T\,b_{kv}$ |
| 4 | `wq_b` | GEMM | `[T,1280]` | `[512h,1280]` | `[T,h,512]` | $2T\cdot 1280\cdot 512h$ | $512h\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | – | $T\cdot 512h\,a$ |
| 5 | RoPE(q) + 补到 $h_k$ | 逐元素 | | | `[T,h_k,512]` | ~0 | – | $T\cdot 512h\,a$ | – | $T\cdot 512\,h_k\,a$ |
| 6 | 稀疏 MQA 注意力（FlashMLA sparse prefill） | 注意力 | q_pad，idx `[T,512]`，SWA（当前 chunk + 环） | – | `[T,h_k,512]` | 执行 $2048\,h_k B(\Sigma_{\mathrm{sel}}+\Sigma_{\mathrm{win}})$；有用的换成 $h$ | – | $T\cdot 512\,h_k\,a+T\cdot 512\cdot 4$ | 稀疏：至多 $B\,\Sigma_{\mathrm{sel}}\,b_{kv}$（按 query 取；相邻 query 之间能复用多少看 kernel）。SWA：$\ge B(S+W_{\mathrm{eff}})\,b_{kv}$（稠密 tile） | $T\cdot 512\,h_k\,a$ |
| 7 | 逆 RoPE | 逐元素 | | | `[T,h,512]` | ~0 | – | $T\cdot 512h\,a$ | – | $T\cdot 512h\,a$ |
| 8 | `wo_a` 分组 | GEMM | `[T,g,4096]` | `[g,1024,4096]` BF16 | `[T,1024g]` | $2Tg\cdot 1024\cdot 4096$ | $g\cdot 1024\cdot 4096\cdot 2$ | $T\cdot 4096g\,a$ | – | $T\cdot 1024g\,a$ |
| 9 | `wo_b` | GEMM | `[T,1024g]` | `[d,1024g]` | `[T,d]` | $2T\cdot 1024g\,d$ | $d\cdot 1024g\,b_w(T)$ | $T\cdot 1024g\,a$ | – | $Td\,a$ |
| 10 | **通信：all-reduce** | 通信 | `[T,d]` | | | | | | | |
| 11 | mHC 边界 + ffn_norm | norm / 逐元素 | | `[24,4d]` | | 同 #1 | | | | |
| 12 | router | router | `[T,d]` | `[384,d]` BF16 | `[T,6]` | $2Td\cdot 384$ | $384\,d\cdot 2$ | $Td\,a$ | – | $48\,T$ |
| 13 | **通信：dispatch**（占位） | 通信 | | | | | | | | |
| 14 | 路由专家 `w1` ‖ `w3` | GEMM（分组） | `[T_r,d]` | $E_r\times$ `[4608,d]` | `[T_r,4608]` | $2T_r d\cdot 4608$ | $E_r\cdot 4608\,d\,b_e$（$T$ 大时 $E_r\to 48$） | $T_r d\,a$ | – | $T_r\cdot 4608\,a$ |
| 15 | SwiGLU 限幅 | 逐元素 | | | | | | $T_r\cdot 4608\,a$ | | $T_r\cdot 2304\,a$ |
| 16 | 路由专家 `w2` | GEMM（分组） | `[T_r,2304]` | $E_r\times$ `[d,2304]` | `[T_r,d]` | $2T_r\cdot 2304\,d$ | $E_r\cdot 2304\,d\,b_e$ | $T_r\cdot 2304\,a$ | – | $T_r d\,a$ |
| 17 | 加权求和 | 逐元素 | | | `[T,d]` | | | $T_r d\,a$ | | $Td\,a$ |
| 18–20 | 共享专家（`w1` ‖ `w3`，SwiGLU，`w2`） | GEMM，逐元素，GEMM | `[T,d]` | `[2f/TP,d]`，`[d,f/TP]` FP8 | `[T,d]` | $6Td\,f/\mathrm{TP}$ | $3\,(f/\mathrm{TP})\,d\,b_w(T)$ | $Td\,a+\dots$ | – | $Td\,a$ |
| 21 | **通信：combine**（占位） | 通信 | | | | | | | | |

关于 prefill：

- **#6。** 这里的注意力在 prefill 里也是稀疏的，所以 FLOPs 对 $S$ 是线性的，不是平方。每个 query、每个保留的头花 $2048(K_{\mathrm{sel}}+W_{\mathrm{eff}})$ FLOP。prefill 里的平方项在 Full 层的索引器里（§4）。
- **分块 prefill。** 基线用 `--chunked-prefill-size 4096`。表按块套用：$P_0=\text{块号}\cdot C$，$T$ 是这一块的 token 数。这个 $T$ 就是决定 $b_w(M)$ 的那个 GEMM 的 M；开 bounded replay 时，它也决定尾巴有多长。
- **#14。** $T$ 大时本卡每个专家都会被命中（$E_r=48$），所以每层每卡的路由专家权重字节封顶在 $48\cdot 3\cdot 2304\,d\,b_e$。

## 4. 其他模式的增减表

这些行插在 Reuse 表第 #3 行之后；第 #6 行的注意力改用本层自己的 idx。

### 4a. Full 模式：第 2、8、14 层（$m=2$）和第 20 层（$m=1$）

$\gamma_m = 1+[m=2]$：只有压缩比 2 时有 gate。decode 每 $m$ 个 token 吐出一条压缩条目。

| # | 算子 | 类型 | 权重 | FLOPs，decode | FLOPs，prefill | 读 W | 读 A / KV | 写 |
|---|---|---|---|---|---|---|---|---|
| F1 | 压缩器投影（kv ‖ gate），复制，FP32 输出 | GEMM | $\gamma_m\times$ `[512,d]` BF16 | $2Td\cdot 512\,\gamma_m$ | 同左 | $\gamma_m\cdot 512\,d\cdot 2$ | $Td\,a$ | $T\cdot 512\,\gamma_m\cdot 4$ |
| F2 | 池化 $m$ 个 token（按通道对 gate 做 softmax），RMSNorm(512)，RoPE，量化 + 写入 | 逐元素 / norm | `[512]` | $O(T\cdot 512)$ | 同左 | ~0 | $T\cdot 512\,\gamma_m\cdot 4$（+ 压缩比 2 待配对的状态） | $\lceil T/m\rceil\,b_{kv}$ |
| F3 | 索引器 K：`wk`，`k_norm`，RoPE，FP4 量化 + 写入 | GEMM + 逐元素 | `[128,512]` BF16 | $2\lceil T/m\rceil\cdot 512\cdot 128$ | 同左 | $128\cdot 512\cdot 2$ | $\lceil T/m\rceil\cdot 512\,a$ | $\lceil T/m\rceil\,b_{ik}$ |
| F4 | 索引器 Q：`wq_b_I` 作用在 q_lat 上（每张卡都算全部 32 个头），RoPE，FP4 假量化 | GEMM | `[4096,1280]` FP8 | $2T\cdot 1280\cdot 4096$ | 同左 | $4096\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | $T\cdot 4096\,a$ |
| F5 | `weights_proj` | GEMM | `[32,d]` BF16 | $2Td\cdot 32$ | 同左 | $32\,d\cdot 2$ | $Td\,a$ | $T\cdot 32\,a$ |
| F6 | 索引打分 $\sum_h w_h\,\mathrm{relu}(q_h\cdot k)$，扫所有可见条目 | 类注意力（GEMM） | – | $2\cdot 32\cdot 128\,T N_c$（+ $2\cdot 32\,T N_c$） | $8192\,B\sum_i\lceil (P_0+i)/m\rceil\approx 8192\,B\,(P_0 S+S^2/2)/m$ | – | KV：decode $T N_c\,b_{ik}$；prefill $B N_c\,b_{ik}\lceil S/\beta_q\rceil$（$\beta_q$ 是 query tile，看 kernel） | 要落地的话，logits $T N_c\cdot 4$ |
| F7 | 在 $N_c$ 里取前 512 | 逐元素（选择） | – | $O(T N_c)$ | $O(B\sum_i N_c(i))$ | – | $T N_c\cdot 4$（要落地的话） | $T\cdot 512\cdot 4$ |
| F8 | **只有第 20 层**：每 8 个位置一块取最大，选前 2048 块，建候选表 | 逐元素（选择） | – | $O(T N_c)$ | $O(B\sum_i N_c(i))$ | – | $T N_c\cdot 4$ | $T\cdot 2048\cdot 4$ |

第 20 层压缩器的输入，是进入这一层的隐状态，也就是编码器的输出 $H_{20}$。这就是 CED：解码器的全局 KV 是 $f(H_{20})$。

### 4b. Reindex 模式：第 24、28、32、36 层（$m=1$，候选池来自第 20 层）

| # | 算子 | 类型 | FLOPs | 读 W | 读 A / KV | 写 |
|---|---|---|---|---|---|---|
| R1 | 索引器 Q（= F4） | GEMM | $2T\cdot 1280\cdot 4096$ | $4096\cdot 1280\,b_w(T)$ | $T\cdot 1280\,a$ | $T\cdot 4096\,a$ |
| R2 | `weights_proj`（= F5） | GEMM | $2Td\cdot 32$ | $32\,d\cdot 2$ | $Td\,a$ | $T\cdot 32\,a$ |
| R3 | 在候选池上打分 | 类注意力 | $8192\,T P_c$（prefill 是 $8192\,B\sum_i\min(16384,P_0+i)$） | – | KV：$T P_c\,b_{ik}$（经候选表去取，候选表本身还要读 $T\cdot 2048\cdot 4$） | |
| R4 | 在候选池里取前 512 | 选择 | $O(T P_c)$ | – | | $T\cdot 512\cdot 4$ |

### 4c. 只有 SWA：第 0、1 层

去掉第 #6 行里读 idx 的那部分。注意力的 key 只有 $W_{\mathrm{eff}}$ 条 SWA：FLOPs $2048\,h_k T W_{\mathrm{eff}}$，KV $T W_{\mathrm{eff}}\,b_{kv}$（decode）。

### 4d. Engram：在第 1 层和第 14 层之前

| # | 算子 | 类型 | 输入 | 权重 | 输出 | FLOPs | 读 W | 读 A | 写 |
|---|---|---|---|---|---|---|---|---|---|
| G1 | n-gram 哈希（2、3、4 阶 × 8 个头，每个 token 24 个 id） | 逐元素（整数） | input_ids | – | `[T,24]` | $O(24\,T)$ | – | $T\cdot 4\cdot 8$ | $T\cdot 24\cdot 8$ |
| G2 | 从按 GPU 分片的表里取行（每行 256 个 FP8，每 32 个一个 E8M0） | gather | `[T,24]` | 行 `[~384M,256]` | `[T,6144]` | – | $T\cdot 24\cdot 264=6336\,T$（分散在各分片） | – | $T\cdot 6144\,a$ |
| G3 | **通信：分片表之间交换**（占位） | 通信 | `[T,6144]` | | | | | | |
| G4 | `wkv`（复制） | GEMM | `[T,6144]` | `[25600,6144]` FP8 | `[T,5·d]`：4 条流各一个 key，加 1 个 value | $2T\cdot 6144\cdot 25600$ | $25600\cdot 6144\,b_w(T)$ | $T\cdot 6144\,a$ | $T\cdot 5d\,a$ |
| G5 | 每条流一个门：$\sigma(\mathrm{norm}(X_s)\cdot\mathrm{norm}(k_s))$，把门控后的 v 加到 $X_s$ 上 | 逐元素 | X `[T,4,d]`，kv | `q_weight` / `k_weight` `[4,d]` | X `[T,4,d]` | $O(T\cdot 4d)$ | $2\cdot 4d\cdot 2$ | $T\cdot 4d\,a+T\cdot 5d\,a$ | $T\cdot 4d\,a$ |

### 4e. 模型两头（每步一次）

| # | 算子 | 类型 | FLOPs | 读 W | 写 / 通信 |
|---|---|---|---|---|---|
| M1 | 查 embedding + mHC 展开 | gather | – | $T d\cdot 2$（碰到的行） | $T\cdot 4d\,a$；**通信：按词表切分的 all-reduce**（占位） |
| M2 | mHC 合流 + 最后的 RMSNorm | 逐元素 | $O(T_{\mathrm{out}}\cdot 4d)$ | – | $T_{\mathrm{out}}\,d\,a$ |
| M3 | LM head（按词表切分），作用在 $T_{\mathrm{out}}$ 行上 | GEMM | $2\,T_{\mathrm{out}}\,d\cdot 129280/\mathrm{TP}$ | $129280\,d\cdot 2/\mathrm{TP}$ | **通信：收集 logits**（占位） |

这里 prefill（只算最后一个 token）和 decode 的 $T_{\mathrm{out}}=B$，验证时是 $B(\gamma+1)$。

## 5. 要留神的地方，和来源之间对不上的地方

1. **这里没有"稠密还是 MoE"的问题。** 40 层全是 MoE：没有 `first_k_dense_replace`，第 0–39 层都有专家。也没有哈希路由的层。V4 和 V4-Flash 前 3 个 MoE 层用哈希路由，这一条没有带过来。按注意力模式给层分类（§2）。
2. **没有 MLA 吸收的问题。** 没有 `kv_b` 上投影。`wq_b` 直接映到 512 维的 latent（64 × 512），K 和 V 是同一个 latent（MQA，一个 KV 头），输出在 `wo_a` 之前做逆 RoPE。prefill 和 decode 的算法一样。V3 里吸收还是不吸收的取舍，这里不存在。
3. **H20 上的 KV 字节和报告不一样。**
   - SGLang 的 `select_dsv4_kv_layout()` 在 **SM90 上选 V4 布局**：每条 584 B（448 FP8 + 64 BF16 + 8 B scale）。FP4 的 V4.1 布局（每条 288 B）只给 SM100/SM103。
   - 索引器 K 强制用 FP4：68 B。
   - 每个上下文 token 的全局 KV $=3\cdot(584+68)/2+(584+68)=1630$ B，和部署笔记里的约 1.6 KB 对得上。按报告的 FP4 设计会是约 890 B。
   - SWA 按请求算：窗口满时 $40\cdot\min(128,L)\cdot 584\approx 2.99$ MB。
4. **KV 在注意力 TP 上是复制的。** 只有一个 latent 头，所以 8 张卡都存、都读完整的缓存；每卡的 KV 字节不随 TP 变小。能改变这一点的是 DP attention（基线里关着）。
5. **存多少不等于读多少。**
   - 只有 4 个 kv_src 层存全局 KV，但 **38 个 CSA2 层每层都要为每个 query 取自己的 $K_{\mathrm{sel}}\cdot 584$ B**。decode（$L\ge 1024$）里，每个请求每步是 $38\cdot 512\cdot 584\approx 11.4$ MB。
   - Reuse 模式省的是索引器的计算，不是取 KV 的字节。
   - 索引器扫描每个请求每步再加 $(3\lceil L/2\rceil+L)\cdot 68+4\min(16384,L)\cdot 68$ B。$L\ge 16$K 时约为 $170L+4.46$ MB，大约在 $L\approx 58$K 处超过取 KV 加 SWA 的那部分（约 14.4 MB）。它随 $L$ 增长，只是因为第 2、8、14、20 这几个 Full 层。
6. **稀疏 top-k 把注意力的字节和 FLOPs 从 $O(L)$ 降到 $O(\min(512,L/m)+128)$。** 只剩索引器在 decode 里是 $O(L)$，在 prefill 里是 $O(S^2/m)$（Full 层）。第 20 层按 $m=1$ 扫，扫描代价是每个编码器 Full 层的两倍。
7. **SM90 上 FlashMLA 补头。** TP > 1 时，`_kernel_num_heads` 把 8 个真头补到 64 个，其余填零。执行的注意力 FLOPs 是有用的 8 倍。
   - 执行的 decode 算术强度 $=2048\,h_k/584\approx 224$ FLOP/B（$h_k=64$）。
   - 有用的算术强度 $=2048\cdot 8/584\approx 28$ FLOP/B（$h=8$）。
   - roofline 上要用执行的那个值。
8. **索引器在 TP 上是复制的。** 32 个头在每张卡上都跑，索引器的 `wq_b`（FP8 `[4096,1280]`）也是复制的。它的 FLOPs 和字节不除以 TP。`wq_a` 和 `wkv` 也是复制的。
9. **每个子层的 mHC 流量**（每层两个子层）：理想的 Single-Pass 是每 token $(2n+2)d=10d$ 个元素，两遍是 $(3n+2)d=14d$，V4 的多 kernel 是 $(4n+4)d=20d$。SM90 的 kernel 实际做到哪一档还没验证（SGLang 里 #42245 已合，#40891，也就是 SM90 小 batch 那条路，还没合）。系数 GEMM 是 `[T,20480]` × `[20480,24]`，F32/TF32。
10. **稠密权重的字节数取决于 M。**
    - Hopper 的 BF16 缓存（默认开）在实际 M ≥ 64 时，用 BF16 副本跑 FP8 稠密线性层，所以权重字节翻倍，MMA 是 BF16。M < 64 时用 FP8 group-32 的 Triton。
    - **`wo_a` 永远是 BF16。** 加载时就反量化了，因为 FP8 einsum 那条路要 128 × 128 的块，checkpoint 用的是 32 × 32。
11. **路由专家。** safetensors 里存成 `I8`，其实是打包的 MXFP4（一个字节两个，每 32 个一个 E8M0 scale），别按 INT8 算。H20 上默认按 W4A16 跑（CUTLASS，BF16 MMA）；W4A8（FP8 MMA）要加开关 `--flashinfer-mxfp4-moe-precision fp8`。每个专家的有效 batch 是 $Tk/E_{\mathrm{act}}$，$E_{\mathrm{act}}=384\,(1-(1-6/384)^T)$。共享专家每次都命中，$M=T$。
12. **CED 在 prefill 上省的那一半是有条件的。**
    - 报告里"prefill 只激活 8B、prefill 减半"，前提是解码器的 SWA bounded replay 开着。基线里它是关的，所以 40 层都跑全部 $S$ 个 token。
    - 开着的时候，SGLang 的分界在第 21 层，不是第 20 层。第 20 层（解码器的 Full 层）每个 token 都得跑，才能建出解码器的 KV 和候选池。
13. **decode 的索引器有一个部署上的坑。** 在 SM90 上用 DSpark 静态验证时，验证用的索引器打分扫的是配置的上下文容量，不是可见长度（SGLang #40945，还没合）。这种情况下扫描代价里的 $L$ 要换成 `--context-length`。
14. **名字上的坑。**
    - SGLang 里的"main cache"是 SWA，"extra cache"是压缩后的 C1/C2。报告里的"main KV"指的是压缩的那个。
    - `num_nextn_predict_layers: 3` 是 **DSpark 草稿器**，不是 V3 那种 MTP（报告里没有经典的 MTP）。
    - `compress_ratios` 有 43 项，最后 3 项是草稿器的块。
15. **我之前的笔记写的是 V4 / V4-Pro / V4-Flash，不是 V4.1-Flash。** 那里是 $m=4$ 的 CSA 加 $m=128$ 的 HCA，索引器 64 头，$d=4096$，43 层，256 个专家，q 秩 1024，还有哈希路由。**这些都不适用。** V4.1-Flash 是纯 CSA2，$m\in\{2,1\}$，索引器 32 头，$d=5120$，40 层，384 个专家，q 秩 1280。我关于 Sinkhorn / Birkhoff 的数学笔记，只和 mHC 的 $B_l$ 投影有关。
16. **checkpoint 里没有 `hc_head`。** 用 Single-Pass mHC 时，最后 4 → 1 的合流用最后一个子层的 A 系数（`hc_combine`）。不要多加一个 `hc_head` GEMM。
17. **压缩比 2 的 Full 层，每个 token 都跑压缩器 GEMM**，但每 2 个 token 才吐一条 KV。存储和索引器扫描的条数用 $\lceil L/2\rceil$。
18. **不同类型的层 RoPE 不一样。** 第 0–1 层用 θ = 1e4。CSA2 层用 θ = 1.6e5 加 YaRN ×16（原长 64K，β_fast 32，β_slow 1），作用在 Q、SWA 分支和压缩 KV 上。这对 roofline 没影响，但复现 kernel 时要管。
19. **SWA 的容量不等于 SWA 的流量。**
    - **流量。** 每一步每个请求读它当前的窗口：$40\min(128,L)\cdot 584$ B（≤ 2.99 MB）。
    - **容量。** SGLang 预留一个分页的 SWA 池，大小是 $R(W+E_I D+2p)+2C+p+4R(W+p)$ 个 token × $40\cdot 584$ B（cap 模式）。这里 $R$ 是 `max_running_requests`，$W=128$，$E_I$ 是 SWA 的淘汰间隔，$D$ 是草稿 token 数（不开投机时是 1），$4R$ 是为 radix 复用留的前缀尾巴，$p$ 是页大小，$C$ 是 chunk 大小。按部署笔记，$R=256$ 时约 13 GiB，前提是 $p\approx 256$。
    - **不设 `--max-running-requests`。** SGLang 会退回比例模式，给*每个*上下文 token 加 $0.1\cdot 40\cdot 584=2336$ B，比 1630 B 的全局 KV 还多。这个开关要显式设上。
20. **Engram 是容量问题，不是每步带宽问题。** 两张 FP8 表每卡约 23.6 GiB，按 GPU 分片。每个模块每个 token 只查约 6.3 KB，但要跨卡交换（占位 G3）。复制的 `wkv` GEMM（`[25600,6144]`）在小 batch 的 decode 里，每个模块要读约 157 MB 权重，不算小。

## 6. 只从模型推出来的量（不带硬件）

| 量 | 公式 | 值 |
|---|---|---|
| 每个上下文 token 存的全局 KV（H20，每卡，复制） | $3\cdot 652/2+652$ | 1630 B |
| 同上，按报告的 FP4 设计 | $3\cdot(288+68)/2+(288+68)$ | 890 B |
| 每个请求存的 SWA | $40\cdot 128\cdot 584$ | 2.99 MB |
| decode 每个请求每步取的 KV（$L\ge 1024$） | $38\cdot 512\cdot 584+40\cdot 128\cdot 584$ | 11.36 + 2.99 = 14.35 MB |
| decode 每个请求每步扫的索引器 K | $68\,(3\lceil L/2\rceil+L+4\min(16384,L))$ | $170L$ + 4.46 MB（$L\ge 16$K） |
| decode 注意力每个 key 的算术强度（执行 / 有用） | $2048\,h_k/584$，$2048\,h/584$ | 224 / 28 FLOP/B |
| 索引器打分每个 key 的算术强度 | $2\cdot 32\cdot 128/68$ | 120 FLOP/B |
| 每个 token 的激活参数（decode） | 每层注意力 126.6M + MoE（6 + 1 个专家）247.7M，× 40，+ head + Engram `wkv` | ≈ 16.1B |
| decode 每步每卡读的稠密权重，$M<64$（不含专家；含 head 分片和 Engram `wkv`） | W 列加起来 | ≈ 2.1 GB（各层 1.53 + Engram `wkv` 0.32 + head 0.17 + 索引器 / 压缩器 0.08） |
| 每层每卡的路由专家字节 | $E_r\cdot 3\cdot 2304\cdot 5120\cdot 0.53125$ | $E_r\times$ 18.8 MB（封顶在 48：0.90 GB） |
| 每卡 BF16 缓存（和部署笔记里的 2.5 GiB 对一下） | TP 8 下全部 FP8 稠密线性层，共享专家按 TP 切 | 2.46 GiB |

## 7. 精度：checkpoint 里存的和 H20 上跑的

checkpoint 是 FP8，32 × 32 的块，UE8M0 scale，激活 scale 动态算；路由专家是 MXFP4。下面"跑的"指固定 SHA 的 SGLang 在 SM90 上、按基线开关走的那条路。

| 部件 | checkpoint | H20 上怎么跑 | TP |
|---|---|---|---|
| `wq_a`，`wkv`（SWA） | FP8 E4M3 | M < 64 时 FP8 W8A8 group-32 Triton，否则从 BF16 缓存跑 BF16 `F.linear` | 复制 |
| `wq_b` | FP8 E4M3 | 同 `wq_a` | 列并行，每卡 8 个头 |
| `wo_a` | FP8 E4M3 | BF16，加载时反量化 | 列并行，每卡 1 组 |
| `wo_b` | FP8 E4M3 | 同 `wq_a` | 行并行 + all-reduce |
| 压缩器 `wkv`，`wgate` | BF16 | BF16，FP32 累加和输出 | 复制 |
| 索引器 `wq_b` | FP8 E4M3 | 同 `wq_a` | 复制 |
| 索引器 `wk`，`weights_proj` | BF16 | BF16 | 复制 |
| 注意力 MMA | – | 没验证（大概是反量化后 BF16） | – |
| 索引器打分 | – | 没验证（SM90 不接受 FP4 索引器开关） | – |
| router gate | BF16 | BF16 | 复制 |
| 共享专家 | FP8 E4M3 | 同 `wq_a` | 列 / 行并行，每卡 288（推出来的） |
| 路由专家 | MXFP4 | 默认 W4A16（CUTLASS，BF16 MMA）；加开关是 W4A8（FP8 MMA） | EP 8，每卡 48 个专家 |
| Engram 表 | FP8 E4M3 + 每 32 个一个 E8M0 | 同左 | 按行分片到各卡 |
| Engram `wkv` | FP8 E4M3 | 同 `wq_a`（推断在 BF16 缓存里） | 复制 |
| mHC `hc_*_fn` | F32 | F32/TF32 GEMM + 融合的前后处理 kernel | 复制 |
| embed，head | BF16 | BF16 | 按词表切分（推出来的） |

每个参数的字节数（含 scale）：FP8 块 32 是 $1+1/1024$，BF16 是 2，MXFP4 是 $0.5+1/32=0.53125$。

## 8. 模型其余部分，简要

- **router。** 用 sqrtsoftplus 打分，`noaux_tc` 取 top-k；选择用的 bias 文本是 `gate.bias`，图像 token 是 `gate.bias_vl`（F32 `[384]`，只用来选）；top-k 权重重新归一后乘 1.5。一个共享专家，FFN 维 2304。SwiGLU，$|x|\le 10$。
- **mHC。** 4 条流，Sinkhorn 20 次，`hc_eps` 1e-6。24 个混合系数是 $n(2+n)$ = A(4) + B(16) + C(4)，来自一个 `[24,20480]` 的 F32 投影，每层两个（注意力前、FFN 前）。Single-Pass：最后的合流沿用最后一组 A。
- **Engram。** 两个模块，在第 1、14 层之前，作用在残差流上。n-gram 阶数 2、3、4，每阶 8 个哈希头，头维 256，每个 token 一共 6144。两张表分别有 384,006,168 和 384,016,682 行。`wkv` 是 `[25600,6144]`，$25600=(4+1)\cdot 5120$：4 条流各一个 key，加一个 value。没有短卷积。
- **embedding 和 head。** 不共享权重，都是 BF16 `[129280,5120]`。最后的 norm 是 RMSNorm(5120)。config 里 `rms_norm_eps` 写的是 1e-20。
- **注意力的细节。** 64 个 query 头，1 个 KV 头，头维 512（448 不带 RoPE + 64 带 RoPE），q 的 LoRA 秩 1280，8 个输出组，每组秩 1024，`wo_a` `[8,1024,4096]` 是块对角的分组 GEMM，`wo_b` `[5120,8192]`，每个头一个 F32 的 attention sink，滑动窗口 128。SWA 用的 `wkv` 是 `[512,5120]`，40 层都有。
- **DSpark 草稿器**（`mtp.*`，基线里不加载）。3 个只有 SWA 的块，MQA 维度和主干一样；MoE 是 128 个路由专家选 3，一个共享专家，FFN 2304；`main_proj` `[5120,15360]` FP8，输入是第 37、38、39 层隐状态拼起来；5 个草稿位置，所以验证窗口是 6；一个秩 256 的 Markov head 和一个置信度 head `[1,5376]`。`mtp.*` 在盘上是 7.93 GB。开 DSpark 时，SWA 缓存每个请求多 $3\cdot 128\cdot 584$ B（SGLang 里怎么分配没验证）。
- **视觉**（不在文本的 roofline 路径上）。DeepSeek-ViT，32 层，宽 1024，16 头，MLP 2816，patch 14，3 × 3 的 pixel unshuffle，aligner 是 BF16 的 `[5120,9216]` 和 `[5120,5120]`，最多 1024 个图像 token。
- **KV 缓存的其余部分。** 压缩比 2 的 Full 层每个请求留一个小的压缩状态环（这里没算大小）。分页的 SWA 池把页跨度补到 576 B 的倍数。V4.1 的设计只把全局 KV 存到 SSD 上，不存 SWA。
