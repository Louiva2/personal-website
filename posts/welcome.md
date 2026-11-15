---
title: 写在前面
date: 2026-11-15
summary: 这个站点用来放笔记。这里说明它是什么，以及一篇笔记怎样写进来。
---

我是 Yihua Liu，GitHub 上的名字是 [Louiva2](https://github.com/Louiva2)。公开简介只有一句：I'm a math guy。

这些页面放在仓库 [personal-website](https://github.com/Louiva2/personal-website) 里。我想把读到的数学写成能回头翻的笔记，另外留一个自己仓库的索引。笔记用 Markdown 写，公式用 LaTeX，打开页面时在浏览器里排版。

## 公式怎么写

行内公式放在一对美元符号中间，例如勾股定理 $a^2+b^2=c^2$。单独成行的公式用两对：

$$
e^{i\pi}+1=0.
$$

这只是排版示例。拿来读的一篇是[调和级数为什么发散](harmonic-series.md)。

## 怎样加一篇

在 `posts/` 里新建 Markdown 文件，开头写成：

```yaml
---
title: 笔记标题
date: 2026-11-15
summary: 列表页上的一两句话。
---
```

然后在 `js/posts.js` 里登记。列表按日期从新到旧排列。本地预览和 GitHub Pages 的开法写在仓库 README 里。
