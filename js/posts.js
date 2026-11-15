// 笔记目录，标题、日期、摘要以这里为准。新笔记：在 posts/ 加一份 Markdown，在这里加一条，
// 再运行 node tools/build-posts.mjs 生成 blog/<slug>/。页面按 date 从新到旧排列。
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
