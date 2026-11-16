(function () {
  const root = globalThis;
  // This file sits in js/, one level below the site root, wherever the site is served from.
  const siteRoot = new URL("../", document.currentScript.src);

  function formatDate(iso) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
    if (!match) return iso || "";
    return `${match[1]}.${match[2]}.${match[3]}`;
  }

  const STATUS = { writing: "在写", planned: "计划" };

  function pad2(no) {
    return String(no).padStart(2, "0");
  }

  function seriesOf(key) {
    return (root.SITE_SERIES || {})[key] || null;
  }

  // "第 2 周" where the series has a unit, "02" where it does not.
  function issueLabel(series, no) {
    return series.unit ? `第 ${no} ${series.unit}` : pad2(no);
  }

  function seriesUrl(series) {
    return new URL(series.path, siteRoot).href;
  }

  function postUrl(post) {
    return new URL(`blog/${encodeURIComponent(post.slug)}/`, siteRoot).href;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
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
    const published = seriesPosts(key).map((post) => ({ no: post.no, title: post.title, note: post.summary, post }));
    const planned = (series.planned || []).map((item) => ({ no: item.no, title: item.title, note: item.note, status: item.status }));
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
    const base = list.getAttribute("data-base") || "./";
    // data-posts="notes" leaves out the posts that belong to a series.
    const notesOnly = list.getAttribute("data-posts") === "notes";
    const posts = sortedPosts().filter((post) => !(notesOnly && post.series));
    list.replaceChildren();
    if (!posts.length) {
      const item = document.createElement("li");
      item.textContent = "还没有笔记。";
      list.append(item);
      return;
    }
    for (const post of posts) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `${base}${encodeURIComponent(post.slug)}/`;
      const meta = el("span", "index-meta");
      const time = document.createElement("time");
      time.dateTime = post.date;
      time.textContent = formatDate(post.date);
      meta.append(time);
      const series = seriesOf(post.series);
      if (series) meta.append(el("span", "index-tag", `${series.title} · ${pad2(post.no)}`));
      link.append(meta, el("span", "index-title", post.title), el("span", "index-summary", post.summary));
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
      body.append(el("span", untitled ? "toc-title is-untitled" : "toc-title", untitled ? STATUS[entry.status] || "" : entry.title));
      if (entry.note) body.append(el("span", "toc-note", entry.note));
      row.append(el("span", "toc-no", pad2(entry.no)), body);
      if (entry.post) {
        const time = el("time", "toc-status", formatDate(entry.post.date));
        time.dateTime = entry.post.date;
        row.append(time);
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
    let text = "";
    if (published.length) {
      const post = published[published.length - 1];
      lead = "最新";
      text = `${issueLabel(series, post.no)} · ${post.title}`;
    } else {
      const next = (series.planned || []).slice().sort((a, b) => a.no - b.no)[0];
      if (!next) return;
      lead = STATUS[next.status] || "";
      text = next.title ? `${issueLabel(series, next.no)} · ${next.title}` : issueLabel(series, next.no);
    }
    slot.replaceChildren(el("span", "card-lead", lead), document.createTextNode(text));
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
    link.href = new URL("blog/", siteRoot).href;
    link.textContent = "返回笔记";
    back.append(link);
    article.append(title, paragraph, back);
    article.setAttribute("aria-busy", "false");
  }

  function showNotFound(article) {
    showMessage(article, "找不到这篇笔记", "这个地址没有对应的笔记。");
  }

  function showLoadError(article, reason) {
    // fetch() cannot read file:// pages, so opening the HTML directly always ends up here.
    const hint = location.protocol === "file:"
      ? "请用本地静态服务器打开站点，不要直接双击 HTML 文件。"
      : "请稍后刷新再试。";
    showMessage(article, "加载失败", reason + hint);
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
      if (!Array.isArray(root.SITE_POSTS)) showLoadError(article, "页面脚本没有载入完整。");
      else if (manifest) location.replace(new URL(`blog/${slug}/`, siteRoot).href + location.hash);
      else showNotFound(article);
      return;
    }
    if (typeof root.renderMarkdown !== "function" || typeof root.splitFrontMatter !== "function") {
      showLoadError(article, "页面脚本没有载入完整。");
      return;
    }
    article.setAttribute("aria-busy", "true");
    try {
      const early = root.SITE_POST_FETCH;
      const response = await (early && early.slug === slug ? early.response : fetch(new URL(`posts/${slug}.md`, siteRoot)));
      if (response.status === 404) {
        showNotFound(article);
        return;
      }
      if (!response.ok) throw new Error(String(response.status));
      const raw = await response.text();
      const { meta, body } = root.splitFrontMatter(raw);
      // js/posts.js wins over the front matter: the generated <title> and share tags come from it.
      const info = manifest || {};
      const title = info.title || meta.title || slug;
      const date = info.date || (/^\d{4}-\d{2}-\d{2}$/.test(meta.date || "") ? meta.date : "");
      document.title = `${title} · Yihua Liu`;
      article.replaceChildren();
      const series = seriesOf(info.series);
      const header = document.createElement("header");
      header.className = "page-head";
      if (series) {
        const kicker = el("p", "kicker");
        const home = el("a", null, series.title);
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
      link.href = series ? seriesUrl(series) : new URL("blog/", siteRoot).href;
      link.textContent = series ? `返回${series.title}` : "返回笔记";
      back.append(link);
      if (series) {
        // The issue before and the issue after, among the published ones.
        const siblings = seriesPosts(info.series);
        const at = siblings.findIndex((post) => post.slug === slug);
        const around = [
          { post: siblings[at - 1], cls: "post-prev", lead: "上一篇" },
          { post: siblings[at + 1], cls: "post-next", lead: "下一篇" },
        ].filter((side) => at !== -1 && side.post);
        if (around.length) {
          const pager = el("nav", "post-nav");
          pager.setAttribute("aria-label", `${series.title}的前后两篇`);
          for (const side of around) {
            const step = el("a", side.cls);
            step.href = postUrl(side.post);
            step.append(el("span", "post-nav-lead", `${side.lead} · ${issueLabel(series, side.post.no)}`), el("span", "post-nav-title", side.post.title));
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
      showLoadError(article, "这篇笔记没有加载成功。");
    }
  }

  document.querySelectorAll("[data-posts]").forEach(renderList);
  document.querySelectorAll("[data-series]").forEach(renderSeries);
  document.querySelectorAll("[data-series-latest]").forEach(renderLatest);
  document.querySelectorAll("[data-ledger]").forEach(renderLedger);
  const article = document.querySelector("[data-article]");
  if (article) loadArticle(article);
})();
