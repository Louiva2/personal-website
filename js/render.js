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
      // Only links to other posts. Anything with a scheme (https:, mailto:) or a host (//) stays.
      if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href)) return full;
      const match = href.match(/(?:^|\/)([a-z0-9-]+)\.md(#[^"]*)?$/);
      if (!match) return full;
      // Posts live at blog/<slug>/, so another post is a sibling directory. On an English page,
      // a post that has no English version is in the Chinese set, three levels up.
      const english = /^en\b/i.test(document.documentElement.lang || "");
      const target = (root.SITE_POSTS || []).find((post) => post.slug === match[1]);
      const up = english && target && !target.en ? "../../../blog/" : "../";
      return `href="${up}${match[1]}/${match[2] || ""}"`;
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
    // Fenced code (``` or ~~~) is set aside first, so nothing inside it is read as math.
    // As in CommonMark, the fence starts a line and is closed by a line of at least as many
    // of the same character; a mid-line ~~~ (好的~~~) is just text.
    src = src.replace(/^([ \t]*)((`|~)\3{2,})([^\n`]*)\n([\s\S]*?)^[ \t]*\2\3*[ \t]*$/gm, (_, indent, fence, char, lang, code) => {
      return indent + take({ type: "code", lang: lang.trim(), code: code.replace(/\n$/, "") });
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
        let out;
        try {
          out = katexApi.renderToString(token.tex, {
            displayMode: token.display,
            throwOnError: false,
            strict: "ignore",
            trust: false,
          });
        } catch (err) {
          out = `<span class="math-fallback">${escapeHtml(token.tex)}</span>`;
        }
        return token.display ? `<div class="math-display">${out}</div>` : out;
      }
      return "";
    };

    // Code blocks and display math are blocks, and a <p> cannot hold a block. A paragraph
    // that contains one is cut around it; usually the block is all the paragraph holds.
    const blockToken = (part) => {
      const id = /^%%T(\d+)%%$/.exec(part);
      const token = id && tokens[Number(id[1])];
      return token && (token.type === "code" || token.display === true) ? token : null;
    };
    html = html.replace(/<p>([\s\S]*?)<\/p>/g, (paragraph, inner) => {
      const parts = inner.split(/(%%T\d+%%)/);
      if (!parts.some(blockToken)) return paragraph;
      let out = "";
      let text = "";
      for (const part of parts) {
        const block = blockToken(part);
        if (!block) {
          text += part;
          continue;
        }
        if (text.trim()) out += `<p>${text.trim()}</p>`;
        out += renderToken(block);
        text = "";
      }
      if (text.trim()) out += `<p>${text.trim()}</p>`;
      return out;
    });
    html = html.replace(/%%T(\d+)%%/g, (_, id) => renderToken(tokens[Number(id)]));
    return rewriteLinks(html);
  }

  root.splitFrontMatter = splitFrontMatter;
  root.renderMarkdown = renderMarkdown;
})(typeof globalThis !== "undefined" ? globalThis : this);
