# personal-website

Yihua Liu（GitHub：[Louiva2](https://github.com/Louiva2)）的个人站点。页面是静态 HTML / CSS / JS，放在仓库根目录，笔记用 Markdown 存在 `posts/`，公式由浏览器里的 [KaTeX](https://katex.org/) 0.19.1 排版（Markdown 解析用 marked 18.2.0）。这两个库和 KaTeX 字体都放在 `assets/` 里，打开页面不需要构建，也不依赖外部 CDN。许可证随库文件放在对应目录。

公开简介只有一句：I'm a math guy。

发布地址是 <https://louiva2.github.io/personal-website/>。

## 本地预览

在仓库根目录启动一个静态服务器。不要直接双击 `index.html`：笔记是用 `fetch` 读取的，`file://` 下浏览器会拦住。

```bash
python3 -m http.server 8080
```

然后打开 <http://127.0.0.1:8080/>。首页、笔记、项目分别在：

- <http://127.0.0.1:8080/>
- <http://127.0.0.1:8080/blog/>
- <http://127.0.0.1:8080/projects/>

页面链接都是相对路径，所以从仓库根目录预览是通的。GitHub Pages 项目站点会多一层 `/personal-website/`。要按发布后的路径检查，在上一级目录把本仓库放成名为 `personal-website` 的文件夹（或做符号链接），再从那里启动服务器，然后打开 <http://127.0.0.1:8080/personal-website/>。

换任何别的静态服务器也可以。

## 启用 GitHub Pages

页面按项目站点来写，对应地址是 <https://louiva2.github.io/personal-website/>。

1. 打开本仓库的 **Settings → Pages**。
2. 在 **Build and deployment** 里，**Source** 选 **Deploy from a branch**。
3. **Branch** 选默认分支 `main`，文件夹选 **`/ (root)`**。
4. 保存。

等部署完成后再访问 <https://louiva2.github.io/personal-website/>。

请保留根目录的 `.nojekyll`。没有它的话，GitHub Pages 会用 Jekyll 处理仓库，`posts/` 里带开头信息的 Markdown 可能不会按原文件发布。

## 加一篇笔记

1. 在 `posts/` 新建 `your-slug.md`。文件名只用小写字母、数字和连字符。开头写成：

```yaml
---
title: 笔记标题
date: 2026-11-15
summary: 列表页上的一两句话。
---
```

2. 在 `js/posts.js` 的 `SITE_POSTS` 里加一条，`slug` 与文件名（不含 `.md`）一致。

列表按 `date` 从新到旧排列。行内公式用一对 `$`，独立成行的公式用一对 `$$`。文章采用 CC BY-NC 4.0，见下面的版权说明。

## 版权

代码、样式和页面设计版权归 Yihua Liu（Louiva2）所有，保留所有权利。没有授予开源许可。在 GitHub 上查看本仓库，并不表示可以复制或再利用这些代码和设计。

`posts/` 里的文章采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)：署名，并且仅限非商业性使用。KaTeX、marked 以及 KaTeX 字体仍适用各自的许可证，文本在 `assets/` 对应目录。

详见 [LICENSE](LICENSE) 与 [LICENSE-CONTENT](LICENSE-CONTENT)。

## English

Static personal site for Yihua Liu (Louiva2). Posts are Markdown files rendered in the browser, with LaTeX via vendored KaTeX. No build step.

Preview from the repository root:

```bash
python3 -m http.server 8080
```

Open <http://127.0.0.1:8080/>. Do not open `index.html` via `file://`; post loading uses `fetch`.

To mimic the GitHub Pages project-site subpath, serve this directory as `personal-website/` from its parent and open <http://127.0.0.1:8080/personal-website/>.

GitHub Pages: **Settings → Pages → Deploy from a branch → `main` → `/ (root)`**. Keep `.nojekyll`.

Expected URL: <https://louiva2.github.io/personal-website/>.

### Copyright

Code and design © 2026 Yihua Liu (Louiva2). All rights reserved. No open-source license is granted, and viewing this repository on GitHub does not grant reuse rights.

Posts in `posts/` are under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Bundled KaTeX, marked, and the KaTeX fonts keep their own licenses in `assets/`.

See [LICENSE](LICENSE) and [LICENSE-CONTENT](LICENSE-CONTENT).
