// 站点的全部目录数据都在这个文件里：文章、两个栏目、栏目里还没发的计划。
//
// 新文章：在 posts/ 加一份 Markdown，在 SITE_POSTS 加一条，再运行 node tools/build-posts.mjs
// 生成 blog/<slug>/。标题、日期、摘要以这里为准。列表按 date 从新到旧排列。
//
// 栏目文章多写两项：series（"weekly" 或 "roofline"）和 no（期号，同一栏目里不能重复）。
// 这一期如果原来列在下面的 planned 里，发布时把那一条删掉。
//
// 找边儿里写某一只模型的文章，再带一项 ledger，栏目页的账本就多出一行（也可以是数组，一篇多行）：
//   ledger: { model: "模型名", gpu: "卡", precision: "fp16", unit: "tok/s", bound: 150, measured: 120 }
// bound 是纸上算出的上界，measured 是实测，同一个单位；"到边"那一列由这两个数算出来。
globalThis.SITE_POSTS = [
  {
    slug: "welcome",
    title: "写在前面",
    date: "2026-11-15",
    summary: "这个站点用来放笔记。这里说明它是什么，以及一篇笔记怎样写进来。",
  },
  {
    slug: "harmonic-series",
    title: "调和级数为什么发散",
    date: "2026-11-14",
    summary: "用分组和积分比较看调和数为什么没有上界，以及它发散得有多慢。这是教科书里的整理，不是新结果。",
  },
];

// 栏目。planned 是还没发的期：没有 Markdown，也不生成页面，只出现在栏目页的目录里。
// status 写 "writing"（在写）或 "planned"（计划）。title 可以先留空。
// unit 是期号的量词：有 unit 时写成"第 2 周"，没有时写成两位数"02"。
globalThis.SITE_SERIES = {
  weekly: {
    title: "每周一拆",
    en: "Infra Weekly",
    path: "weekly/",
    unit: "周",
    deck: "每周挑 AI infra 里的一样东西，读到实现，拆开，把背后的数学推一遍，讲清楚它为什么长这样。",
    planned: [
      {
        no: 1,
        title: "FP4：为什么是 4 位",
        note: "先算量化误差的分布（泊松求和）和每少一位的代价（率失真，6 dB 一位），再算块 scale 的税；最后是推理和训练两条回本线，加上硬件只给 4 和 8 两档。",
        status: "writing",
      },
      {
        no: 2,
        title: "投机解码：从拒绝采样到 DFlash、DSpark 的实现",
        note: "验证那一步是 p 和 q 的最大耦合：边缘分布还是 p，所以输出一个 token 都不变；接受率恰好是 1 − TV，不可能更高。再读 DFlash 和 DSpark 两份实现，看验证的预算怎么按置信度分。",
        status: "planned",
      },
      {
        no: 3,
        title: "vLLM 和 SGLang：两套架构差在哪儿",
        note: "分三层对着读：上层的调度和 KV cache，中层的计算图编译，下层的 kernel。只下机制上的结论，不比谁更快。",
        status: "planned",
      },
      {
        no: 4,
        title: "tensor core 里的加法：是截断，不是舍入",
        note: "Hopper 的 FP8 MMA 一次融合 32 个乘积，对齐后只留 13 位小数，其余截掉。截断的误差有偏，按 K 累积；随机舍入无偏，误差是鞅，只按 K 的平方根长。再看为什么 batch 一变，同一行的结果也跟着变。",
        status: "planned",
      },
      {
        no: 5,
        title: "Triton 的 LinearLayout：把张量布局写成 F₂ 上的矩阵",
        note: "寄存器号、线程号、warp 号的二进制位到张量下标的二进制位，是 F₂ 上的一个线性映射。布局转换变成解线性方程，swizzle 怎么选才少撞 bank 变成子空间求交；代价是形状只能是 2 的幂。",
        status: "planned",
      },
      {
        no: 6,
        title: "attention 的 I/O 下界：FlashAttention 最优吗",
        note: "红蓝卵石游戏加 Loomis–Whitney 不等式，给出矩阵乘法搬数据量的下界，tile 取方块正是不等式取等的条件。到了 attention：SRAM 装得下 d² 个数时 FlashAttention 已经在界上，装不下时另有更好的分块。",
        status: "planned",
      },
      {
        no: 7,
        title: "Muon 的正交化：只用矩阵乘法做极分解",
        note: "谱范数下的最速下降方向是 UVᵀ。Newton–Schulz 用一个奇次多项式反复作用在奇异值上，系数怎么选是等波纹逼近问题；DeepSeek-V4 为什么先走 8 步不收敛的系数，再走 2 步收敛的。",
        status: "planned",
      },
      {
        no: 8,
        title: "EP 的负载均衡：从 EPLB 到 MoonEP",
        note: "EP 的一步由最慢的 rank 决定。EPLB 按统计负载复制热门 expert，是启发式；MoonEP 证明每个 rank 留 E/R 个冗余槽，任意路由都能配到完全均衡。证明是贪心配对，和采样里的 alias method 是同一个构造。",
        status: "planned",
      },
      {
        no: 9,
        title: "all-reduce 最少要搬多少字节",
        note: "N 个进程做 all-reduce，总有一个进程至少要传 2(N−1)/N 份数据。ring 取到这个界，代价是 2(N−1) 步的延迟；NCCL 的双二叉树把延迟降到对数。再拿实测的斜率和截距去对 TP 的通信账。",
        status: "planned",
      },
      {
        no: 10,
        title: "胖树不堵，ECMP 会堵",
        note: "胖树可重排无阻塞：任何一组收发配对都存在互不相撞的路，证明用的是 Hall 定理。但 ECMP 让每条流自己哈希选路，训练流量又只有少数几条大流：16 条流分到 16 条上行，一条不撞的概率是百万分之一。",
        status: "planned",
      },
      {
        no: 11,
        title: "约束解码：mask 之后采的不是条件分布",
        note: "每一步要算出哪些 token 还合法，读 XGrammar 怎么把这件事压到几十微秒。但逐步屏蔽再归一化，采到的不是模型在“输出合法”这个条件下的分布；差的那个因子，概率论里叫 Doob 的 h-变换。",
        status: "planned",
      },
      {
        no: 12,
        title: "continuous batching 是一个处理器共享队列",
        note: "batch 里有 n 个请求时一步耗时 τ(n)，这是状态依赖的处理器共享：稳态只依赖输出长度的均值，不依赖分布，直到并发封顶、有人排队为止。τ(n) 先平后斜时，脊点就是 Erlang C 里的服务台数。",
        status: "planned",
      },
      {
        no: 13,
        title: "GPTQ 是 Babai 的最近平面算法",
        note: "逐列量化、把误差往后传，就是在 Hessian 定出的格里做最近平面取整。误差的账：RTN 付 Hessian 的迹，GPTQ 付 LDL 分解里 D 的迹。往上是 QuIP# 的 E8 码本；往回是 FP4，格点不等距，这套还剩多少。",
        status: "planned",
      },
    ],
  },
  roofline: {
    title: "找边儿",
    en: "Roofline, model by model",
    path: "roofline/",
    unit: "",
    deck: "一块卡能跑多快，纸上先有一个上界。这里先把它算出来，再一个模型一个模型地量，看实测离边儿还差多少、差在哪儿。",
    planned: [
      { no: 0, title: "两个数画一条线", note: "内存带宽和峰值算力，定出脊点。", status: "planned" },
      {
        no: 1,
        title: "把 decode 放上去",
        note: "一步 decode 要读多少字节、做多少次浮点运算。batch 从 1 往上加，点沿着斜线往右上走；过了脊点，换成算力说了算。prefill 是另一个点。",
        status: "planned",
      },
      {
        no: 2,
        title: "先把量具焊好",
        note: "负载怎么发、token 怎么数、稳态怎么判。量具不可信，后面的差距都没法归因。",
        status: "planned",
      },
    ],
  },
};
