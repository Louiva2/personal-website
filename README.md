# personal-website

Yihua Liu（GitHub：[Louiva2](https://github.com/Louiva2)）的个人站点。页面是静态 HTML / CSS / JS，放在仓库根目录，笔记用 Markdown 存在 `posts/`，公式由浏览器里的 [KaTeX](https://katex.org/) 0.19.1 排版（Markdown 解析用 marked 18.2.0）。这两个库和 KaTeX 字体都放在 `assets/` 里，打开页面不需要构建，也不依赖外部 CDN。许可证随库文件放在对应目录。

公开简介只有一句：I'm a math guy。

发布地址是 <https://louiva2.github.io/personal-website/>。

## 本地预览

在仓库根目录启动一个静态服务器。不要直接双击 `index.html`：笔记是用 `fetch` 读取的，`file://` 下浏览器会拦住。

```bash
python3 -m http.server 8080
```

然后打开 <http://127.0.0.1:8080/>。几个页面分别在：

- 首页 <http://127.0.0.1:8080/>
- 每周一拆 <http://127.0.0.1:8080/weekly/>
- 找边儿 <http://127.0.0.1:8080/roofline/>
- 笔记（全部文章） <http://127.0.0.1:8080/blog/>
- 项目 <http://127.0.0.1:8080/projects/>
- 英文版 <http://127.0.0.1:8080/en/>，下面是同样的五个页面

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

1. 在 `posts/` 写 `your-slug.md`。文件名只用小写字母、数字和连字符。
2. 在 `js/posts.js` 的 `SITE_POSTS` 里加一条：`slug`（文件名去掉 `.md`）、`title`、`date`（写成 `2026-11-15`）、`summary`。标题、日期和摘要以这里为准。
3. 运行 `node tools/build-posts.mjs`（Node 18+，没有依赖），生成 `blog/your-slug/index.html`。
4. 把这三样一起提交。

列表按 `date` 从新到旧排列。行内公式用一对 `$`，独立成行的公式用一对 `$$`。文章采用 CC BY-NC 4.0，见下面的版权说明。

正文里可以用表格（GFM 写法，窄屏会横向滚动）、`<figure>` 配 `<figcaption>`、`<details>`。文章自己的图放在 `assets/img/` 下，按文章建子目录。

## 两个栏目

站点有两个栏目：**每周一拆**（`weekly/`）和**找边儿**（`roofline/`）。栏目的数据也在 `js/posts.js` 里，栏目页的目录、首页卡片上的"最新／下一篇"、文章页的栏目名和上下篇，都从这一个文件取。

### 发一期

和加一篇笔记一样，只是在 `SITE_POSTS` 那一条里多写两项：

```js
{
  slug: "speculative-decoding",
  title: "投机解码：从拒绝采样到 DFlash、DSpark 的实现",
  date: "2026-11-27",
  summary: "一两句话。",
  series: "weekly", // 或 "roofline"
  no: 2,            // 期号，同一栏目里不能重复
},
```

如果这一期原来列在 `SITE_SERIES.weekly.planned` 里，把那一条删掉（期号重复时生成脚本会报错）。然后运行 `node tools/build-posts.mjs`。文章地址仍是 `blog/<slug>/`，生成的页面里导航高亮的是它所在的栏目。栏目文章只出现在栏目页和 `blog/` 的总列表里，首页的"笔记"只列不属于栏目的文章。

### 改计划

`SITE_SERIES.<栏目>.planned` 是还没发的期，不需要 Markdown，也不生成页面：

```js
{ no: 14, title: "下一期的题目", note: "一两句话：这一期拆什么，背后是哪条数学。", status: "planned" },
```

`status` 写 `"writing"`（显示"在写"）或 `"planned"`（显示"计划"）。`title` 可以先留空，这时标题的位置显示状态。增、删、换顺序都只改这个数组。

找边儿页上的"怎么出发"也是这样来的：它列的是 roofline 栏目里不带 `ledger` 的条目（计划中的和已发的）。

### 往账本加一行

找边儿里写某一只模型的文章，在 `SITE_POSTS` 那一条上再带一项 `ledger`，栏目页"账本"那张表就多出一行，不用改 HTML：

```js
ledger: { model: "模型名", gpu: "卡", precision: "fp16", unit: "tok/s", bound: 150, measured: 120 },
```

`bound` 是纸上算出的上界，`measured` 是实测，两个都是数字、同一个单位；"到边"一列是 `measured / bound`。一篇文章要记多行时，`ledger` 写成数组。表里一行都没有时显示"第一只还在量。"

### 改栏目名或导语

栏目的名字、英文小标题和导语在两处：`js/posts.js` 的 `SITE_SERIES`，以及栏目页自己的 HTML（没有 JavaScript 时也要看得到）。两处不一致时 `node tools/build-posts.mjs` 会报错。导航和首页卡片上的栏目名是手写的，改名时一起改：`index.html`、`weekly/`、`roofline/`、`blog/index.html`、`blog/post.html`、`projects/index.html`、`404.html`、`tools/post-template.html`。英文那一套见下面的"英文版"。

`roofline/index.html` 里的 roofline 示意图是手写的内联 SVG，有两份：`.roof-narrow` 给手机，`.roof-wide` 给 34rem 以上的宽度，由 CSS 选一份显示。改图时两份都改。

## 英文版

英文是 `en/` 下的另一套页面，地址和中文一一对应：`en/`、`en/weekly/`、`en/roofline/`、`en/blog/`、`en/projects/`，文章在 `en/blog/<slug>/`。中文页的地址不变。每个页面导航的最后一项（`EN`／`中文`）通到对面的同一页。

两套页面读同一份 `js/posts.js`，用同一份 `css/site.css` 和 `js/site.js`。脚本看页面的 `<html lang>` 决定写哪种语言。

### 给一篇文章加英文版

1. 把译文写成 `posts/en/your-slug.md`，文件名和中文那篇一样。
2. 在 `js/posts.js` 里那一条上加一项：

   ```js
   en: { title: "English title", summary: "One or two sentences." },
   ```

3. 运行 `node tools/build-posts.mjs`，它会多生成一页 `en/blog/your-slug/`，并且让两页互相指（导航里的语言切换、`<head>` 里的 `hreflang`）。
4. 一起提交。

去掉 `en` 再运行脚本，那一页英文会被删掉。

### 没有英文版的文章

不需要每篇都译。没有 `en` 的文章，在英文的列表和目录里显示中文标题和摘要，后面标一个"中文"，点过去是中文页；中文页导航里的 `EN` 这时通到英文的笔记列表。英文文章里写 `[…](other-post.md)` 这样的链接，对方没有英文版时会自动指到中文页。

### 栏目和计划的英文

在 `SITE_SERIES` 里：

- 栏目上的 `titleEn`、`deckEn` 是英文页用的名字和导语，`unitEn` 是期号的量词（`"Week"` 写成 "Week 2"，留空写成 "02"）。英文栏目页自己的 HTML 里也写着这两句，和中文页一样，不一致时生成脚本会报错。
- `planned` 里每一期的 `titleEn`、`noteEn` 是英文的题目和说明。缺了就退回中文那两项。

英文页上，标题上面那行小字放的是栏目的中文名。

### 两边各手写一份的东西

导航、页脚、跳到正文的链接、每页的标题和导语、插图的 `alt`、`<head>` 里的 description 和 Open Graph 标签，都是每个页面自己写的，改一边时记得看另一边。英文文章页的外壳是 `tools/post-template.en.html`。`en/roofline/index.html` 里那两份内联 SVG 的文字是另写的，位置也按英文的长度挪过，改图时四份都要看。404 页只有一页，中文下面带一行英文。

英文页的分享图是 `assets/img/og-en.jpg`。

## 插图

页面上的六张插图放在 `assets/img/`，文件名和尺寸是固定的：

| 文件 | 尺寸 | 用在哪儿 |
|---|---|---|
| `hero.webp` | 698 × 698 | 首页 |
| `weekly.webp` | 1026 × 684 | 每周一拆的栏目页和首页卡片 |
| `roofline.webp` | 1050 × 700 | 找边儿的栏目页和首页卡片 |
| `notes.webp` | 1085 × 587 | 笔记页 |
| `projects.webp` | 1058 × 705 | 项目页 |
| `lost.webp` | 572 × 572 | 404 页 |

换图时覆盖同名文件即可。图要求是**纯白底、四周过渡到白**：亮色模式下用 `mix-blend-mode: multiply` 直接叠在页面上，白色的部分看不见，所以图没有边框；暗色模式下图放在一张浅色的"纸片"（`--plate`）上。底色不是纯白的图，亮色模式下会露出一个矩形。换了画面记得同时改 `alt`（`index.html`、两个栏目页、`blog/index.html`、`projects/index.html`、`404.html`）。

## 标志、图标和分享图

站点的标志是一朵花，花心坐着一只小猪。它只有一个源文件，根目录的 `favicon.svg`（64 × 64 的画板）：既是标签页图标，也是页眉里名字左边的小标（`.site-mark`）。另外三样是由它出的位图，改了标志要一起重出：

| 文件 | 尺寸 | 用在哪儿 |
|---|---|---|
| `favicon.ico` | 16、32、48 | 不认 SVG 图标的浏览器 |
| `apple-touch-icon.png` | 180 × 180 | 加到手机主屏时的图标，纸色底 |
| `assets/img/og.jpg` | 1200 × 630 | 链接预览里的大图：标志、名字、首页第一句和首页那张插图 |
| `assets/img/og-en.jpg` | 1200 × 630 | 同一张图的英文版，英文页用 |

三个图标的 `<link>` 手写在每个页面的 `<head>` 里；`og.jpg` 是中文页共用的一张（英文页用 `og-en.jpg`），手写在首页、两个栏目页、笔记、项目的 `og:image` 里，文章页的由生成脚本从 `BASE_URL` 拼出来。

生成的页面把标题和摘要直接写进 `<title>`、description 和 Open Graph 标签，链接预览不用跑 JavaScript。这些页面里的绝对地址来自脚本里的常量 `BASE_URL`（<https://louiva2.github.io/personal-website/>）；首页、两个栏目页、笔记、项目的 canonical 和 og:url 是手写的，站点换地址时一起改。加 `--check` 只检查不写入，页面缺失或过期时以非零状态退出。旧链接 `blog/post.html?p=<slug>` 会跳到新地址。

## 版权

代码、样式和页面设计版权归 Yihua Liu（Louiva2）所有，保留所有权利。没有授予开源许可。在 GitHub 上查看本仓库，并不表示可以复制或再利用这些代码和设计。

`posts/` 里的文章采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)：署名，并且仅限非商业性使用。KaTeX、marked 以及 KaTeX 字体仍适用各自的许可证，文本在 `assets/` 对应目录。

详见 [LICENSE](LICENSE) 与 [LICENSE-CONTENT](LICENSE-CONTENT)。

## English

Static personal site for Yihua Liu (Louiva2). Posts are Markdown files rendered in the browser, with LaTeX via vendored KaTeX. Viewing needs no build step. Each post also gets a generated page at `blog/<slug>/` that carries its title and share-preview tags: after adding the Markdown file and its entry in `js/posts.js`, run `node tools/build-posts.mjs` (Node 18+, no dependencies) and commit the output. Their absolute URLs come from `BASE_URL` in that script; the home, notes and projects pages hard-code theirs.

The site has two series, `weekly/` and `roofline/`. A post joins one by adding `series` and `no` to its entry in `js/posts.js`; issues not yet written live in `SITE_SERIES.<series>.planned` in the same file and need no Markdown. A roofline post with a `ledger` field adds a row to the ledger table on `roofline/`. The six illustrations in `assets/img/` have fixed names and sizes and must have a pure white ground, since they are multiplied onto the page. The site mark is `favicon.svg`, also shown beside the name in the header; `favicon.ico`, `apple-touch-icon.png` and the share-preview picture `assets/img/og.jpg` are rendered from it.

The English pages are a second set under `en/`, one for each Chinese page, reading the same `js/posts.js`. A post gets an English page at `en/blog/<slug>/` when its entry carries `en: { title, summary }` and its translation is at `posts/en/<slug>.md`; a post without one shows its Chinese title on the English lists and links to the Chinese page. The English names of the series and of the planned issues are the `titleEn`, `deckEn`, `unitEn` and `noteEn` fields. The English share-preview picture is `assets/img/og-en.jpg`.

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
