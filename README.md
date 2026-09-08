# inovello.dev

Static site, built with [Hugo](https://gohugo.io/) (extended edition), hosted on Cloudflare Workers (static assets). No theme, no JavaScript, no Node. Four templates and one stylesheet.

## Publish a writeup

1. Make a folder under `content/writeups/` named after the slug. Put `index.md` in it, and any images next to it.
2. Frontmatter:

   ```yaml
   ---
   title: "Short title. The series name supplies the context, so don't repeat it here."
   series: flash-next               # a key from data/series.yaml
   slug: short-stable-slug          # the URL. Never change it after publishing.
   date: 2026-09-20
   kind: writeup                    # or: note
   part: 4                          # optional. Numbers within the series, not across the site.
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

   A new model, or a new line of work, gets a new entry in `data/series.yaml` and its own
   part numbering starting at 1. Writeups group by series on the home and writeups pages,
   and the previous/next links at the foot of an article stay inside the series.

3. Body is plain Markdown. A `## Setup` heading followed by a bullet list renders as a boxed setup block. Images: `![alt](file.png "Caption")` renders a figure with the caption. Tables scroll sideways on narrow screens.
4. Preview: `hugo server -D` in this folder, then open http://localhost:1313. `-D` shows drafts.
5. Commit and push to `main`. Cloudflare builds and deploys in about a minute.

If you ever must move a post, add `aliases: ["/writeups/old-slug/"]` to its frontmatter. Hugo writes a redirect page at the old URL.

## Home page data

- `data/now.yaml`: the daily-driver strip and the "Currently open" panel. Open work only in that panel; anything finished belongs in a writeup. Edit `updated` when you edit the rest.
- `data/series.yaml`: the writeup groups, in display order.
- `data/projects.yaml`: the project cards.
- `content/_index.md`: the intro paragraph and the tagline.
- `content/hardware.md`: the box.

## Cloudflare settings

| Setting | Value |
|---|---|
| Build command | `bash cloudflare-build.sh` (downloads the pinned Hugo, then builds) |
| Build output directory | `public` |
| Deploy command | `npx wrangler deploy` (reads `wrangler.jsonc`, serves `public/`) |
| Custom domains | `inovello.dev`, plus `www.inovello.dev` redirected to the apex |

The Hugo version is pinned in `cloudflare-build.sh`. Change it there when you upgrade locally.

## Local install

Hugo extended via winget: `winget install Hugo.Hugo.Extended`. Git is the only other requirement.
