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
    deck: "每周挑推理系统里的一样东西，读到实现，拆开，讲清楚它为什么长这样。",
    planned: [
      { no: 1, title: "", note: "", status: "writing" },
      {
        no: 2,
        title: "投机解码：从拒绝采样到 DFlash、DSpark 的实现",
        note: "投机解码为什么不改变输出分布；再读 DFlash 和 DSpark 两份实现。",
        status: "planned",
      },
      {
        no: 3,
        title: "vLLM 和 SGLang：两套架构差在哪儿",
        note: "同一个请求在两边各走一遍，看调度、缓存和执行各自放在哪一层。",
        status: "planned",
      },
      {
        no: 4,
        title: "调度器的一步",
        note: "continuous batching 里，prefill 和 decode 怎么分同一份 token 预算；TTFT 和 TPOT 各自付了什么。",
        status: "planned",
      },
      {
        no: 5,
        title: "KV cache 的两种记账",
        note: "分页的块表和前缀的 radix 树；命中率怎么进吞吐的账。",
        status: "planned",
      },
      {
        no: 6,
        title: "约束解码：语法怎么变成 mask",
        note: "token mask 在 CPU 上算，这笔开销能不能藏到 forward 后面。",
        status: "planned",
      },
      {
        no: 7,
        title: "decode attention 的 kernel",
        note: "从 FlashAttention 到分页 KV 上的变长 batch，读 FlashInfer。",
        status: "planned",
      },
      {
        no: 8,
        title: "量化省下的是什么",
        note: "小 batch 的 decode 受内存带宽限制，权重字节数直接进吞吐的上界；FP8、INT4 各把上界抬了多少，实测又兑现了多少。",
        status: "planned",
      },
      {
        no: 9,
        title: "一张卡不够的时候",
        note: "tensor parallel 的通信账，以及什么时候不如多开副本。",
        status: "planned",
      },
      {
        no: 10,
        title: "压测怎么不骗自己",
        note: "开环和闭环负载，coordinated omission，按 SLO 算 goodput。",
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
