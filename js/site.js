(function () {
  const root = globalThis;
  // This file sits in js/, one level below the site root, wherever the site is served from.
  const siteRoot = new URL("../", document.currentScript.src);
  // The English pages are a second set under en/. The page says which set it belongs to.
  const EN = /^en\b/i.test(document.documentElement.lang || "");
  const langRoot = EN ? new URL("en/", siteRoot) : siteRoot;

  // Everything this script writes into a page, in the page's language.
  const TEXT = EN
    ? {
        status: { writing: "Writing", planned: "Planned" },
        latest: "Latest",
        noPosts: "No notes yet.",
        chinese: "中文",
        backToNotes: "Back to notes",
        backTo: (name) => `Back to ${name}`,
        previous: "Previous",
        next: "Next",
        pager: (name) => `The issues before and after this one in ${name}`,
        notFound: ["No such note", "There is no note at this address."],
        loadFailed: "Could not load",
        scriptsMissing: "The page’s scripts did not load completely. ",
        postMissing: "This note did not load. ",
        hintFile: "Open the site through a local static server, not by double-clicking the HTML file.",
        hintRetry: "Please refresh in a moment.",
      }
    : {
        status: { writing: "在写", planned: "计划" },
        latest: "最新",
        noPosts: "还没有笔记。",
        chinese: "中文",
        backToNotes: "返回笔记",
        backTo: (name) => `返回${name}`,
        previous: "上一篇",
        next: "下一篇",
        pager: (name) => `${name}的前后两篇`,
        notFound: ["找不到这篇笔记", "这个地址没有对应的笔记。"],
        loadFailed: "加载失败",
        scriptsMissing: "页面脚本没有载入完整。",
        postMissing: "这篇笔记没有加载成功。",
        hintFile: "请用本地静态服务器打开站点，不要直接双击 HTML 文件。",
        hintRetry: "请稍后刷新再试。",
      };

  function formatDate(iso) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
    if (!match) return iso || "";
    return `${match[1]}.${match[2]}.${match[3]}`;
  }

  const STATUS = TEXT.status;

  function pad2(no) {
    return String(no).padStart(2, "0");
  }

  function seriesOf(key) {
    return (root.SITE_SERIES || {})[key] || null;
  }

  // "第 2 周" where the series has a unit, "02" where it does not. In English, "Week 2".
  function issueLabel(series, no) {
    if (EN) return series.unitEn ? `${series.unitEn} ${no}` : pad2(no);
    return series.unit ? `第 ${no} ${series.unit}` : pad2(no);
  }

  function seriesName(series) {
    return (EN && series.titleEn) || series.title;
  }

  function seriesUrl(series) {
    return new URL(series.path, langRoot).href;
  }

  // A post has an English page only if its entry carries en. Without one, the English pages
  // show its Chinese title and link to the Chinese page.
  function hasPage(post) {
    return !EN || Boolean(post.en);
  }

  function postUrl(post) {
    const root = EN && post.en ? langRoot : siteRoot;
    return new URL(`blog/${encodeURIComponent(post.slug)}/`, root).href;
  }

  // Title and summary of a post as this page shows them. foreign marks Chinese text on an English page.
  function view(post) {
    if (EN && post.en) return { title: post.en.title, summary: post.en.summary, foreign: false };
    return { title: post.title, summary: post.summary, foreign: EN };
  }

  // A planned issue: the English fields on the English pages, the Chinese ones where they are missing.
  function planView(item) {
    if (!EN) return { title: item.title, note: item.note, titleForeign: false, noteForeign: false };
    return {
      title: item.titleEn || item.title,
      note: item.noteEn || item.note,
      titleForeign: !item.titleEn && Boolean(item.title),
      noteForeign: !item.noteEn && Boolean(item.note),
    };
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  // Chinese text set on an English page says so, for fonts, line breaking and screen readers.
  function zh(node) {
    node.lang = "zh-CN";
    return node;
  }

  // Published posts of one series, by issue number.
  function seriesPosts(key) {
    return (root.SITE_POSTS || [])
      .filter((post) => post.series === key)
      .sort((a, b) => a.no - b.no);
  }

  // Published and planned issues of one series in one list, by issue number.
  function seriesEntries(key) {
    const series = seriesOf(key);
    if (!series) return [];
    const published = seriesPosts(key).map((post) => {
      const shown = view(post);
      return { no: post.no, title: shown.title, note: shown.summary, titleForeign: shown.foreign, noteForeign: shown.foreign, post };
    });
    const planned = (series.planned || []).map((item) => ({ no: item.no, status: item.status, ...planView(item) }));
    return published.concat(planned).sort((a, b) => a.no - b.no);
  }

  function sortedPosts() {
    return (root.SITE_POSTS || [])
      .map((post, index) => ({ post, index }))
      .sort((a, b) => {
        if (a.post.date === b.post.date) return a.index - b.index;
        return a.post.date < b.post.date ? 1 : -1;
      })
      .map((item) => item.post);
  }

  function renderList(list) {
    // data-posts="notes" leaves out the posts that belong to a series.
    const notesOnly = list.getAttribute("data-posts") === "notes";
    const posts = sortedPosts().filter((post) => !(notesOnly && post.series));
    list.replaceChildren();
    if (!posts.length) {
      const item = document.createElement("li");
      item.textContent = TEXT.noPosts;
      list.append(item);
      return;
    }
    for (const post of posts) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = postUrl(post);
      const shown = view(post);
      const meta = el("span", "index-meta");
      const time = document.createElement("time");
      time.dateTime = post.date;
      time.textContent = formatDate(post.date);
      meta.append(time);
      const series = seriesOf(post.series);
      if (series) meta.append(el("span", "index-tag", `${seriesName(series)} · ${pad2(post.no)}`));
      const title = el("span", "index-title", shown.title);
      const summary = el("span", "index-summary", shown.summary);
      if (shown.foreign) {
        // No English version: the Chinese title, marked, and the link goes to the Chinese page.
        meta.append(zh(el("span", "index-tag", TEXT.chinese)));
        link.hreflang = "zh-CN";
        zh(title);
        zh(summary);
      }
      link.append(meta, title, summary);
      item.append(link);
      list.append(item);
    }
  }

  // The numbered contents of a series: published issues are links, the rest carry their status.
  // data-only="plain" keeps the entries without a ledger, i.e. the ones that are not about one model.
  function renderSeries(list) {
    const key = list.getAttribute("data-series");
    const plainOnly = list.getAttribute("data-only") === "plain";
    const entries = seriesEntries(key).filter((entry) => !(plainOnly && entry.post && entry.post.ledger));
    list.replaceChildren();
    for (const entry of entries) {
      const item = el("li", entry.post ? "toc-item" : `toc-item is-${entry.status}`);
      const row = entry.post ? el("a", "toc-row") : el("div", "toc-row");
      if (entry.post) row.href = postUrl(entry.post);
      const body = el("span", "toc-body");
      // An issue that has no title yet shows its status where the title would be.
      const untitled = !entry.title;
      const title = el("span", untitled ? "toc-title is-untitled" : "toc-title", untitled ? STATUS[entry.status] || "" : entry.title);
      if (entry.titleForeign) zh(title);
      body.append(title);
      if (entry.note) {
        const note = el("span", "toc-note", entry.note);
        if (entry.noteForeign) zh(note);
        body.append(note);
      }
      // A published issue with no English version leads to the Chinese page.
      if (entry.post && !hasPage(entry.post)) row.hreflang = "zh-CN";
      row.append(el("span", "toc-no", pad2(entry.no)), body);
      if (entry.post) {
        const status = el("span", "toc-status");
        const time = el("time", null, formatDate(entry.post.date));
        time.dateTime = entry.post.date;
        status.append(time);
        if (!hasPage(entry.post)) status.append(zh(el("span", "index-tag", TEXT.chinese)));
        row.append(status);
      } else if (!untitled) {
        row.append(el("span", "toc-status", STATUS[entry.status] || ""));
      }
      item.append(row);
      list.append(item);
    }
  }

  // One line on a home page card: the newest published issue, or else what comes first in the plan.
  function renderLatest(slot) {
    const key = slot.getAttribute("data-series-latest");
    const series = seriesOf(key);
    if (!series) return;
    const published = seriesPosts(key);
    let lead = "";
    let label = "";
    let title = "";
    let foreign = false;
    if (published.length) {
      const post = published[published.length - 1];
      const shown = view(post);
      lead = TEXT.latest;
      label = issueLabel(series, post.no);
      title = shown.title;
      foreign = shown.foreign;
    } else {
      const next = (series.planned || []).slice().sort((a, b) => a.no - b.no)[0];
      if (!next) return;
      const shown = planView(next);
      lead = STATUS[next.status] || "";
      label = issueLabel(series, next.no);
      title = shown.title;
      foreign = shown.titleForeign;
    }
    const parts = [el("span", "card-lead", lead), document.createTextNode(title ? `${label} · ` : label)];
    if (title) parts.push(foreign ? zh(el("span", null, title)) : document.createTextNode(title));
    slot.replaceChildren(...parts);
  }

  function formatNumber(value) {
    return Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  }

  // The ledger: one row per ledger entry on the posts of a series. The table already holds
  // the empty-state row; it is replaced once there is a row to show.
  function renderLedger(body) {
    const key = body.getAttribute("data-ledger");
    const rows = [];
    for (const post of seriesPosts(key)) {
      const entries = Array.isArray(post.ledger) ? post.ledger : post.ledger ? [post.ledger] : [];
      for (const entry of entries) rows.push({ post, entry });
    }
    if (!rows.length) return;
    body.replaceChildren();
    for (const { post, entry } of rows) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      const link = el("a", null, entry.model);
      link.href = postUrl(post);
      name.append(link, el("span", "ledger-rig", `${entry.gpu} · ${entry.precision}`));
      const share = Math.round((entry.measured / entry.bound) * 100);
      const reach = el("td", "ledger-reach");
      const bar = el("span", "ledger-bar");
      bar.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, share))}%`;
      bar.append(fill);
      reach.append(document.createTextNode(`${share}%`), bar);
      row.append(
        name,
        el("td", null, `${formatNumber(entry.bound)} ${entry.unit}`),
        el("td", null, `${formatNumber(entry.measured)} ${entry.unit}`),
        reach,
      );
      body.append(row);
    }
  }

  function showMessage(article, heading, text) {
    document.title = `${heading} · Yihua Liu`;
    article.replaceChildren();
    const title = document.createElement("h1");
    title.textContent = heading;
    const paragraph = document.createElement("p");
    paragraph.className = "deck";
    paragraph.textContent = text;
    const back = document.createElement("p");
    back.className = "back";
    const link = document.createElement("a");
    link.href = new URL("blog/", langRoot).href;
    link.textContent = TEXT.backToNotes;
    back.append(link);
    article.append(title, paragraph, back);
    article.setAttribute("aria-busy", "false");
  }

  function showNotFound(article) {
    showMessage(article, TEXT.notFound[0], TEXT.notFound[1]);
  }

  function showLoadError(article, reason) {
    // fetch() cannot read file:// pages, so opening the HTML directly always ends up here.
    const hint = location.protocol === "file:" ? TEXT.hintFile : TEXT.hintRetry;
    showMessage(article, TEXT.loadFailed, reason + hint);
  }

  // A display formula or a table wider than the column scrolls sideways. Only those get a tab stop,
  // so the keyboard can scroll them. Widths change once the KaTeX fonts arrive and on resize.
  function watchWideMath(scope) {
    const check = () => {
      for (const box of scope.querySelectorAll(".math-display, .table-scroll")) {
        if (box.scrollWidth > box.clientWidth) box.tabIndex = 0;
        else box.removeAttribute("tabindex");
      }
    };
    check();
    if (document.fonts) document.fonts.ready.then(check);
    let frame = 0;
    addEventListener("resize", () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(check);
    });
  }

  // A table wider than the column scrolls inside its own box instead of widening the page.
  function wrapTables(scope) {
    for (const table of scope.querySelectorAll("table")) {
      const box = el("div", "table-scroll");
      table.replaceWith(box);
      box.append(table);
    }
  }

  async function loadArticle(article) {
    // Pages generated under blog/<slug>/ carry their slug. blog/post.html?p=<slug> is the old
    // address: a known post moves on to its page, anything else gets the not-found message.
    const fixedSlug = article.getAttribute("data-slug");
    const slug = fixedSlug || new URLSearchParams(location.search).get("p") || "";
    // tools/build-posts.mjs applies the same rule; keep the two identical.
    if (!/^[a-z0-9-]+$/.test(slug)) {
      showNotFound(article);
      return;
    }
    const manifest = (root.SITE_POSTS || []).find((post) => post.slug === slug);
    if (!fixedSlug) {
      if (!Array.isArray(root.SITE_POSTS)) showLoadError(article, TEXT.scriptsMissing);
      else if (manifest) location.replace(new URL(`blog/${slug}/`, siteRoot).href + location.hash);
      else showNotFound(article);
      return;
    }
    if (typeof root.renderMarkdown !== "function" || typeof root.splitFrontMatter !== "function") {
      showLoadError(article, TEXT.scriptsMissing);
      return;
    }
    // An English page exists only for a post that has an English version.
    if (EN && !(manifest && manifest.en)) {
      showNotFound(article);
      return;
    }
    article.setAttribute("aria-busy", "true");
    try {
      const early = root.SITE_POST_FETCH;
      const source = EN ? `posts/en/${slug}.md` : `posts/${slug}.md`;
      const response = await (early && early.slug === slug ? early.response : fetch(new URL(source, siteRoot)));
      if (response.status === 404) {
        showNotFound(article);
        return;
      }
      if (!response.ok) throw new Error(String(response.status));
      const raw = await response.text();
      const { meta, body } = root.splitFrontMatter(raw);
      // js/posts.js wins over the front matter: the generated <title> and share tags come from it.
      const info = manifest || {};
      const title = (EN ? info.en.title : info.title) || meta.title || slug;
      const date = info.date || (/^\d{4}-\d{2}-\d{2}$/.test(meta.date || "") ? meta.date : "");
      document.title = `${title} · Yihua Liu`;
      article.replaceChildren();
      const series = seriesOf(info.series);
      const header = document.createElement("header");
      header.className = "page-head";
      if (series) {
        const kicker = el("p", "kicker");
        const home = el("a", null, seriesName(series));
        home.href = seriesUrl(series);
        kicker.append(home, document.createTextNode(` · ${issueLabel(series, info.no)}`));
        header.append(kicker);
      }
      if (date) {
        const time = document.createElement("time");
        time.dateTime = date;
        time.textContent = formatDate(date);
        header.append(time);
      }
      const heading = document.createElement("h1");
      heading.textContent = title;
      header.append(heading);
      const prose = document.createElement("div");
      prose.className = "prose";
      prose.innerHTML = root.renderMarkdown(body);
      wrapTables(prose);
      const back = document.createElement("p");
      back.className = "back";
      const link = document.createElement("a");
      link.href = series ? seriesUrl(series) : new URL("blog/", langRoot).href;
      link.textContent = series ? TEXT.backTo(seriesName(series)) : TEXT.backToNotes;
      back.append(link);
      if (series) {
        // The issue before and the issue after, among the published ones that have a page in this language.
        const siblings = seriesPosts(info.series).filter(hasPage);
        const at = siblings.findIndex((post) => post.slug === slug);
        const around = [
          { post: siblings[at - 1], cls: "post-prev", lead: TEXT.previous },
          { post: siblings[at + 1], cls: "post-next", lead: TEXT.next },
        ].filter((side) => at !== -1 && side.post);
        if (around.length) {
          const pager = el("nav", "post-nav");
          pager.setAttribute("aria-label", TEXT.pager(seriesName(series)));
          for (const side of around) {
            const step = el("a", side.cls);
            step.href = postUrl(side.post);
            step.append(el("span", "post-nav-lead", `${side.lead} · ${issueLabel(series, side.post.no)}`), el("span", "post-nav-title", view(side.post).title));
            pager.append(step);
          }
          article.append(header, prose, pager, back);
        } else {
          article.append(header, prose, back);
        }
      } else {
        article.append(header, prose, back);
      }
      article.setAttribute("aria-busy", "false");
      watchWideMath(prose);
    } catch (err) {
      showLoadError(article, TEXT.postMissing);
    }
  }

  document.querySelectorAll("[data-posts]").forEach(renderList);
  document.querySelectorAll("[data-series]").forEach(renderSeries);
  document.querySelectorAll("[data-series-latest]").forEach(renderLatest);
  document.querySelectorAll("[data-ledger]").forEach(renderLedger);
  const article = document.querySelector("[data-article]");
  if (article) loadArticle(article);
})();
