# inovello.dev

Static site, built with [Hugo](https://gohugo.io/) (extended edition), hosted on Cloudflare Pages. No theme, no JavaScript, no Node. Four templates and one stylesheet.

## Publish a writeup

1. Make a folder under `content/writeups/` named after the slug. Put `index.md` in it, and any images next to it.
2. Frontmatter:

   ```yaml
   ---
   title: "Title as it appears on the page"
   slug: short-stable-slug          # the URL. Never change it after publishing.
   date: 2026-09-06
   kind: writeup                    # or: note
   part: 3                          # optional. Shows "Part 03" in lists.
   summary: "One sentence. Shown in the list and used as the meta description."
   draft: true                      # remove or set false to publish
   discussion: https://reddit.com/... # optional. "Discuss on Reddit" link.
   updated: 2026-10-01              # optional. Only when you materially revise.
   stats:                           # optional. Headline numbers strip under the title.
     - label: Decode
       value: "41 t/s"
       accent: true                 # one cell in orange
   config:                          # optional. Sticky "Run config" sidebar.
     - k: Model
       v: Qwen3.8-Flash-Next
   ---
   ```

3. Body is plain Markdown. A `## Setup` heading followed by a bullet list renders as a boxed setup block. Images: `![alt](file.png "Caption")` renders a figure with the caption. Tables scroll sideways on narrow screens.
4. Preview: `hugo server -D` in this folder, then open http://localhost:1313. `-D` shows drafts.
5. Commit and push to `main`. Cloudflare builds and deploys in about a minute.

If you ever must move a post, add `aliases: ["/writeups/old-slug/"]` to its frontmatter. Hugo writes a redirect page at the old URL.

## Home page data

- `data/now.yaml`: the stats strip and the "Currently open" panel. Edit `updated` when you edit the rest.
- `data/projects.yaml`: the project cards.
- `content/_index.md`: the intro paragraph and the tagline.
- `content/hardware.md`: the box.

## Cloudflare Pages settings

| Setting | Value |
|---|---|
| Build command | `hugo --minify` |
| Build output directory | `public` |
| Environment variable | `HUGO_VERSION` = the version in `hugo version` locally (currently `0.165.0`) |
| Custom domains | `inovello.dev`, plus `www.inovello.dev` redirected to the apex |

Pin `HUGO_VERSION`. Cloudflare's default can be old or change without notice.

## Local install

Hugo extended via winget: `winget install Hugo.Hugo.Extended`. Git is the only other requirement.
