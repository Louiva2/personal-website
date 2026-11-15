(function (root) {
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));
  }

  function splitFrontMatter(text) {
    const src = String(text).replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
    if (!src.startsWith("---\n")) return { meta: {}, body: src };
    const end = src.indexOf("\n---\n", 4);
    if (end === -1) return { meta: {}, body: src };
    const meta = {};
    for (const line of src.slice(4, end).split("\n")) {
      const match = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
      if (!match) continue;
      meta[match[1]] = match[2].trim().replace(/^["']|["']$/g, "");
    }
    return { meta, body: src.slice(end + 5) };
  }

  function rewriteLinks(html) {
    return html.replace(/href="([^"]*)"/g, (full, href) => {
      const match = href.match(/(?:^|\/)([a-z0-9-]+)\.md(#[^"]*)?$/);
      if (!match) return full;
      return `href="post.html?p=${match[1]}${match[2] || ""}"`;
    });
  }

  function renderMarkdown(markdown) {
    const markedApi = root.marked;
    const katexApi = root.katex;
    if (!markedApi || typeof markedApi.parse !== "function") {
      throw new Error("marked missing");
    }
    if (!katexApi || typeof katexApi.renderToString !== "function") {
      throw new Error("katex missing");
    }

    const tokens = [];
    const take = (value) => {
      const id = tokens.length;
      tokens.push(value);
      return `%%T${id}%%`;
    };

    let src = String(markdown).replace(/\r\n/g, "\n");
    src = src.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      return take({ type: "code", lang: lang.trim(), code: code.replace(/\n$/, "") });
    });
    src = src.replace(/`([^`\n]+)`/g, (_, code) => take({ type: "inline-code", code }));
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
      return take({ type: "math", display: true, tex: tex.trim() });
    });
    src = src.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => {
      return take({ type: "math", display: true, tex: tex.trim() });
    });
    src = src.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => {
      return take({ type: "math", display: false, tex: tex.trim() });
    });
    src = src.replace(/(^|[^\\])\$(?!\$)((?:\\.|[^\$\n])+?)\$(?!\$)/g, (match, pre, tex) => {
      return pre + take({ type: "math", display: false, tex: tex.trim() });
    });

    const parsed = markedApi.parse(src, { gfm: true, breaks: false });
    let html = typeof parsed === "string" ? parsed : String(parsed);

    const renderToken = (token) => {
      if (!token) return "";
      if (token.type === "code") {
        const lang = /^[A-Za-z0-9_-]+$/.test(token.lang) ? token.lang : "";
        const cls = lang ? ` class="language-${lang}"` : "";
        return `<pre><code${cls}>${escapeHtml(token.code)}</code></pre>`;
      }
      if (token.type === "inline-code") return `<code>${escapeHtml(token.code)}</code>`;
      if (token.type === "math") {
        try {
          return katexApi.renderToString(token.tex, {
            displayMode: token.display,
            throwOnError: false,
            strict: "ignore",
            trust: false,
          });
        } catch (err) {
          return `<span class="math-fallback">${escapeHtml(token.tex)}</span>`;
        }
      }
      return "";
    };

    html = html.replace(/<p>\s*%%T(\d+)%%\s*<\/p>/g, (_, id) => renderToken(tokens[Number(id)]));
    html = html.replace(/%%T(\d+)%%/g, (_, id) => renderToken(tokens[Number(id)]));
    return rewriteLinks(html);
  }

  root.splitFrontMatter = splitFrontMatter;
  root.renderMarkdown = renderMarkdown;
})(typeof globalThis !== "undefined" ? globalThis : this);
