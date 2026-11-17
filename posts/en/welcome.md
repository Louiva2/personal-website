---
title: Before anything else
date: 2026-11-15
summary: This site is for notes. Here is what it is, and how a note gets written into it.
---

I am Yihua Liu; on GitHub the name is [Louiva2](https://github.com/Louiva2). The public bio is one line: I’m a math guy.

These pages live in the repository [personal-website](https://github.com/Louiva2/personal-website). I want to write the mathematics I read into notes I can come back to, and to keep an index of my own repositories. Notes are written in Markdown, formulas in LaTeX, and the page is typeset in the browser when it is opened.

## Writing formulas

An inline formula goes between a pair of dollar signs, like the Pythagorean theorem $a^2+b^2=c^2$. A formula on a line of its own takes two pairs:

$$
e^{i\pi}+1=0.
$$

That is only a typesetting sample. The note meant for reading is [Why the harmonic series diverges](harmonic-series.md).

## Adding a note

Create a Markdown file in `posts/` that begins with:

```yaml
---
title: The title of the note
date: 2026-11-15
summary: A sentence or two for the list page.
---
```

Then register it in `js/posts.js` and run `node tools/build-posts.mjs` to give it a page of its own. Lists run from newest to oldest. Previewing locally and publishing on GitHub Pages are described in the repository’s README. An English version of a note is a second file under `posts/en/` with the same name.
