# Design brief: inovello.dev

## How to use this document

This is a complete brief for designing the front end of a small static site. Read all of it before producing anything. The visual direction is deliberately left open: choose typography, color, spacing, and layout yourself. The page inventory, the content each page holds, and the technical constraints in the final section are fixed and must be followed exactly, because the output will be cut into Hugo templates by a separate engineer and any deviation from the markup rules costs real rework.

Deliver the following files and nothing else:

- `index.html` (home)
- `writeup.html` (a single writeup, using the sample content below)
- `writeups.html` (the writeup list page)
- `projects.html`
- `hardware.html`
- `404.html`
- `site.css` (the single stylesheet shared by every page)

Every page must be a complete, valid HTML document that renders correctly when opened directly from disk with only `site.css` beside it.

## What the site is

A permanent home for benchmark writeups and failure notes about running large mixture-of-experts language models on consumer hardware. The author runs local inference on two RTX 3090 GPUs with 192 GB of host memory, offloading most expert layers to system RAM, and publishes measured results: decode throughput before and after a change, what broke, and why.

The site replaces Reddit posts, which expire and cannot be cited. It is not a blog for a general audience and has no traffic goals.

Two audiences, in priority order:

1. **Hiring managers and engineers at inference and infrastructure companies.** They receive the root URL in a job application. They need to understand within seconds who the author is, what hardware they work on, and that the writeups are rigorous. They will read one writeup closely.
2. **People in the local-inference community** who read a Reddit post and followed the link for the full version. They will read the writeup, check the setup details, and possibly subscribe to the RSS feed.

Tone of the content: precise, technical, first person, no marketing language. The design should feel like a place where measurements are taken seriously. Beyond that, the visual direction is yours.

## Identity and global elements

**Author identity.** The author is Evans, handle "Inovello". Both appear on the site. The handle is the site name and appears in the header. The real name appears on the home page and as the byline on writeups. Use `Evans` as a placeholder for the full name; the engineer will substitute it.

**Site header** (identical on every page): the site name "Inovello" linking to `/`, and a navigation with exactly these items in this order: Writeups (`/writeups/`), Projects (`/projects/`), Hardware (`/hardware/`). No search, no theme toggle, no icons required.

**Site footer** (identical on every page): outbound links to GitHub, X, and Reddit, and a link to the RSS feed at `/index.xml`. Use `#` placeholders for the three profile URLs. A short copyright or name line is fine. Nothing else.

There is no about page. The home page is the about page.

## Pages

### Home (`/`)

The home page must work as a one-screen introduction for a hiring manager and as the writeup index for a returning reader. Content, in order:

1. **Positioning block.** Name, handle, and a short statement of what the author does. Sample copy:

   > Evans (Inovello). I run large mixture-of-experts language models on consumer hardware and publish what I measure. Current bench: two RTX 3090s and 192 GB of DDR4, with most expert layers offloaded to host memory.

   The hardware sentence should link to `/hardware/`.

2. **Writeups list.** Every writeup, newest first, showing for each: title (linked), date, and the one-sentence summary. Entries of kind "note" should be visually lighter than entries of kind "writeup" so a reader can see at a glance which items are substantial, but they appear in the same list. Include a small kind label or equivalent treatment. Do not truncate the list or add pagination; assume it will hold up to about fifty items over time and should still read cleanly at that length.

3. **Projects, abbreviated.** The project names with one-line descriptions, linking to `/projects/`. Use the project data below.

4. Outbound links are in the footer and need not be repeated.

No hero image, no photo, no tagline animation.

### Writeup (`/writeups/<slug>/`)

The most important page on the site. A reader may spend ten minutes here. Content, in order:

1. **Title.**
2. **Metadata line:** byline (Evans), published date, and, when present, an "Updated" date. Also a kind label when the item is a note.
3. **Article body**, generated from Markdown. See the markup rules below for what elements will appear. The body must comfortably handle:
   - Long paragraphs of technical prose.
   - Headings at two levels below the title (h2 and h3).
   - Fenced code blocks containing shell commands that can run to 200 characters on one line. These must scroll horizontally inside the block, never wrap the page.
   - Tables with up to about eight columns of numbers. Wide tables scroll horizontally inside their wrapper.
   - Images with captions, mostly charts. Charts should be able to sit at full content width.
   - Inline code, links, bulleted and numbered lists, and blockquotes.
4. **Discussion link.** When present, a single link labelled "Discuss on Reddit" pointing at the original thread. It appears after the body, not in the header.
5. **Navigation back** to the writeup list.

Every benchmark writeup opens with a short setup block in the body: llama.cpp commit hash, model and quantization, the exact command used, and a link to the hardware page. This is ordinary Markdown (a heading and a short list or table), not a special component, but the design should make such a block easy to scan.

Sample content for `writeup.html`, to be used as-is with placeholder body text expanded to a realistic length of roughly 1,200 words including two code blocks, one table, and one image with a caption:

- Title: Qwen3 Flash Next on 2x3090: 17 to 41 t/s decode via expert-cache offload
- Date: 2026-09-06
- Kind: writeup
- Summary: Moving the expert cache to host memory and pinning the shared layers on GPU more than doubled decode throughput on UD-Q4_K_XL, at the cost of prompt processing.
- Discussion: `#` placeholder
- Setup block: llama.cpp commit `b7a3c1d` (placeholder), model Qwen3 Flash Next, quant UD-Q4_K_XL, one long `llama-server` command line as a code block, link to `/hardware/`.

### Writeups list (`/writeups/`)

The same list as the home page's writeups section, on its own page with a page title. Nothing else. This page exists mainly so the URL resolves; it can reuse the home list styling exactly.

### Projects (`/projects/`)

A list of projects. Each entry has a name, a one-line description, a link, and a status. Two groups: "Projects" and "Upstream contributions". Sample data:

Projects:

| Name | Description | Status |
|---|---|---|
| kv-sparsity-profiler | Profiles key-value cache sparsity across layers during decode to find safe eviction candidates. | active |
| bitrebuttal | Tooling for reproducing and checking published quantization quality claims. | active |
| lifeos | Personal operations system. Unrelated to inference. | maintenance |

Upstream contributions:

| Name | Description | Status |
|---|---|---|
| llama.cpp PR #28223 | Placeholder one-line description of the change. | open |

Descriptions above are placeholders and will be replaced. Status values are one of: active, maintenance, open, merged, archived. Treat status as a small label.

### Hardware (`/hardware/`)

The test bench specification that every writeup links to. Content: a short paragraph explaining that all published numbers come from this machine unless stated otherwise, then a specification table, then a short section on the offload configuration in prose. Sample data:

| Component | Value |
|---|---|
| GPU | 2x NVIDIA RTX 3090, 24 GB each, 48 GB total |
| Host memory | 192 GB DDR4 |
| CPU | [placeholder] |
| Motherboard | [placeholder] |
| Storage | [placeholder] |
| OS | [placeholder] |
| Driver / CUDA | [placeholder] |

Include a "Last updated" date line.

### 404

A short message, a link home, and a link to the writeups list. Same header and footer as every other page.

## Technical constraints (fixed)

These exist because the output is going into Hugo, a static site generator that renders Markdown into HTML the designer does not control. Violating any of these means the design cannot be used.

**Stack.**

- Plain HTML and one plain CSS file. No CSS framework, no Tailwind, no Sass, no PostCSS, no build step of any kind.
- No JavaScript is required to read the site. Do not include any JavaScript. No dark-mode toggle scripts, no smooth-scroll, no analytics, no icon libraries.
- No React, no components, no templating syntax in the HTML.
- Fonts: either a system font stack, or at most one web font family. If you use a web font, load it with a single `<link>` to Google Fonts and give it a full fallback stack; the engineer may switch it to self-hosted files.
- No external assets other than that optional font link. Icons, if any, are inline SVG.
- Dark mode is welcome but must be implemented with `prefers-color-scheme` in CSS only.
- The layout must be responsive down to 360 px wide with no horizontal page scrolling. Code blocks and tables scroll within their own containers.

**Shared shell.** Every page must use exactly this skeleton so it can be cut into a single Hugo base template:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Page title · Inovello</title>
  <meta name="description" content="...">
  <link rel="stylesheet" href="site.css">
  <link rel="alternate" type="application/rss+xml" title="Inovello" href="/index.xml">
</head>
<body>
  <header class="site-header"> ... </header>
  <main class="site-main"> ... page content ... </main>
  <footer class="site-footer"> ... </footer>
</body>
</html>
```

The header and footer markup must be byte-identical across all pages. Only the contents of `<main>`, the `<title>`, and the description change.

**Article body markup.** The writeup body is generated from Markdown, so you cannot add classes to its elements. Wrap the body in `<article class="writeup">` and style its contents by element selector (for example `.writeup h2`, `.writeup table`, `.writeup pre`). The generator will emit exactly these structures inside the body and nothing else:

- `<h2>`, `<h3>`, `<p>`, `<a>`, `<em>`, `<strong>`, `<ul>`, `<ol>`, `<li>`, `<blockquote>`, `<hr>`.
- Inline code as `<code>` inside a `<p>` or `<li>`.
- Fenced code blocks as:

  ```html
  <div class="highlight"><pre><code class="language-bash">...</code></pre></div>
  ```

  The `<code>` contents will contain `<span>` elements with token classes for syntax highlighting. Do not style those token classes; the generator produces its own highlighting CSS which the engineer will append to `site.css`. Style only the `.highlight`, `pre`, and `code` containers: background, padding, font, border, and horizontal overflow.

- Tables as:

  ```html
  <div class="table-wrap"><table><thead>...</thead><tbody>...</tbody></table></div>
  ```

  The wrapper handles horizontal overflow. Cells carry no classes, so numeric alignment cannot be targeted per column; choose a tabular-figure font treatment that reads well either way.

- Images as:

  ```html
  <figure><img src="..." alt="..." width="..." height="..."><figcaption>...</figcaption></figure>
  ```

  A figure may also appear without a caption. Images must never exceed the content width.

Use the same `.writeup` element rules for the hardware page body and the home page positioning text so prose looks consistent everywhere; the hardware and 404 pages are also Markdown-generated.

**List markup.** For the writeups list, use a structure like:

```html
<ol class="writeup-list">
  <li class="writeup-list-item is-note">   <!-- is-note only for notes -->
    <a href="/writeups/slug/">Title</a>
    <time datetime="2026-09-06">6 Sep 2026</time>
    <p>Summary sentence.</p>
  </li>
</ol>
```

Exact class names may differ, but every item must be representable with only those four pieces of data (title, URL, date, summary) plus the kind flag. Do not design list items that depend on reading time, tags, thumbnails, or excerpts, because that data does not exist.

For projects, each entry must be representable with only: name, description, URL, status.

**Class naming.** Keep class names short, lowercase, hyphenated, and used consistently across pages. Avoid deeply nested selectors; the engineer needs to be able to find and change any rule quickly. Put a short comment block at the top of `site.css` listing the color and spacing variables you define.

**Do not include:** hero images, background images, carousels, animations, newsletter forms, comment sections, share buttons, cookie banners, search boxes, reading-time estimates, author avatars, tag clouds, related-posts blocks, or any element that would require data the pages above do not have.

## What "done" looks like

The seven files listed at the top. Opening `writeup.html` from disk should show a complete, readable technical article with a long code block that scrolls horizontally, a wide table that scrolls horizontally, and a captioned image, all inside the shared header and footer. Resizing the window to 360 px should not produce a horizontal scrollbar on the page itself.
