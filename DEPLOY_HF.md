# Deploy to Hugging Face Spaces (free) + link from the website

This hosts the Gradio app (`app.py`) on a free Hugging Face Space and links it
from the **Interactive Apps** section of `academics.html`
(`Desktop/Admin/Website/web-azan/`).

Why HF Spaces: the keychain engine needs native Python wheels
(`vtracer`, `manifold3d`, `shapely`, `trimesh`) that can't run as static
browser files. HF Spaces installs them from `requirements.txt` automatically —
no Dockerfile, no system packages needed.

---

## 1. One-time setup

1. Create a free account at <https://huggingface.co/join>.
2. New Space: <https://huggingface.co/new-space>
   - **Owner:** your username
   - **Space name:** `image-to-keychain`
   - **SDK:** **Gradio**
   - **Hardware:** **CPU basic** (free)
   - **Visibility:** **Public** (so the website's "Launch Maker" button works
     for visitors). The default font is now the open-license **DejaVu Sans
     Bold** — the personal-use Waltograph font is *not* shipped, so Public is
     fine. (See the font note below.)
3. Create a **write** access token: <https://huggingface.co/settings/tokens>
   (New token → type "Write" → copy it).

## 2. Push this folder to the Space

From this directory (`image_to_keychain/`):

```bash
# Commit the current code (text-plate feature, bundled font, etc.)
git add -A
git commit -m "feat: name plate + HF Spaces config"

# Point a 'hf' remote at your Space and push
git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/image-to-keychain
git push hf HEAD:main
```

When git asks for credentials: **username** = your HF username,
**password** = the **write token** from step 1.3.

> If `git push` is rejected because the Space already has a commit, run
> `git pull hf main --allow-unrelated-histories` first, or push with
> `git push hf HEAD:main -f` (force) since the Space starts empty.

The Space then builds for ~3–5 min (installing the wheels). When it shows
**Running**, it's live at:

- App page: `https://huggingface.co/spaces/<YOUR_USERNAME>/image-to-keychain`
- Bare app:  `https://<YOUR_USERNAME>-image-to-keychain.hf.space`

## 3. Wire the website button

In `Desktop/Admin/Website/web-azan/academics.html`, find **AZAN-HF-USERNAME**
(2 places, in the new "Image → 3D Keychain Maker" card) and replace it with
your real HF username. Commit + push the site repo as usual (GitHub Pages
auto-deploys to `azan.biomechemical.com`).

```bash
# in web-azan/
git add academics.html
git commit -m "feat: add Image to Keychain interactive app card"
git push
```

---

## Notes

- **Free-tier sleep:** the Space sleeps after ~48 h idle and cold-starts in
  ~30 s on the next visit. Fine for a portfolio demo. Paid "always-on" exists
  if you ever want it.
- **Font licensing — already handled.** The public app's default font is
  **DejaVu Sans Bold** (open/redistributable), installed on the Space via the
  `fonts-dejavu-core` line in `packages.txt` — no font binary is committed to
  the repo (Hugging Face rejects committed binaries that aren't in Git LFS). The
  personal-use **Waltograph** font is git-ignored, so it stays on your machine
  for local `disney` use but is never pushed. Nothing to do here.
- **Updating later:** edit code locally, then
  `git commit -am "..." && git push hf HEAD:main` — the Space rebuilds itself.
