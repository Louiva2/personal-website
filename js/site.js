(function () {
  const root = globalThis;

  function formatDate(iso) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
    if (!match) return iso || "";
    return `${match[1]}.${match[2]}.${match[3]}`;
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
    const base = list.getAttribute("data-base") || "post.html?p=";
    const posts = sortedPosts();
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
      link.href = base + encodeURIComponent(post.slug);
      const time = document.createElement("time");
      time.dateTime = post.date;
      time.textContent = formatDate(post.date);
      const title = document.createElement("span");
      title.className = "index-title";
      title.textContent = post.title;
      const summary = document.createElement("span");
      summary.className = "index-summary";
      summary.textContent = post.summary;
      link.append(time, title, summary);
      item.append(link);
      list.append(item);
    }
  }

  function showMessage(article, text) {
    article.replaceChildren();
    const paragraph = document.createElement("p");
    paragraph.className = "deck";
    paragraph.textContent = text;
    const back = document.createElement("p");
    back.className = "back";
    const link = document.createElement("a");
    link.href = "./";
    link.textContent = "返回笔记";
    back.append(link);
    article.append(paragraph, back);
    article.setAttribute("aria-busy", "false");
  }

  async function loadArticle(article) {
    const slug = new URLSearchParams(location.search).get("p") || "";
    if (!/^[a-z0-9-]+$/.test(slug)) {
      showMessage(article, "没有这篇笔记。");
      return;
    }
    if (typeof root.renderMarkdown !== "function" || typeof root.splitFrontMatter !== "function") {
      showMessage(article, "页面脚本没有载入完整。");
      return;
    }
    article.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(`../posts/${slug}.md`);
      if (!response.ok) throw new Error(String(response.status));
      const raw = await response.text();
      const { meta, body } = root.splitFrontMatter(raw);
      const manifest = (root.SITE_POSTS || []).find((post) => post.slug === slug) || {};
      const title = meta.title || manifest.title || slug;
      const date = /^\d{4}-\d{2}-\d{2}$/.test(meta.date || "") ? meta.date : manifest.date || "";
      document.title = `${title} · Yihua Liu`;
      article.replaceChildren();
      const header = document.createElement("header");
      header.className = "page-head";
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
      const back = document.createElement("p");
      back.className = "back";
      const link = document.createElement("a");
      link.href = "./";
      link.textContent = "返回笔记";
      back.append(link);
      article.append(header, prose, back);
      article.setAttribute("aria-busy", "false");
    } catch (err) {
      showMessage(article, "这篇笔记没有加载成功。请用本地静态服务器打开站点，不要直接双击 HTML 文件。");
    }
  }

  document.querySelectorAll("[data-posts]").forEach(renderList);
  const article = document.querySelector("[data-article]");
  if (article) loadArticle(article);
})();
