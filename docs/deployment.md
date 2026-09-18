# Deploying without Streamlit Community Cloud

Streamlit Community Cloud is the least-effort option, but it is not the only one. Nothing in this project depends on it — the app is a normal long-lived Python process, so any host that runs one will serve it.

## What a host must provide

| Requirement | Why | Who fails it |
| --- | --- | --- |
| A **long-lived process** | Streamlit keeps a server and per-visitor WebSocket sessions alive | PHP/shared hosts, plain S3-style static hosting |
| **WebSocket** upgrade support | every interaction is a socket message | some serverless/edge platforms, badly configured proxies |
| **~1 GB RAM** | CPU-only PyTorch plus pandas and Plotly | 512 MB containers are tight — it fits, barely |
| **Outbound HTTPS** | `services.swpc.noaa.gov` for live X-ray flux | egress-restricted sandboxes |
| A **stable HTTPS URL** | the whole point | — |
| A writable directory *(optional)* | otherwise `data/prediction_history.csv` and `logs/` reset | every free tier |

## Pick your host in 30 seconds

| If you want… | Go to | Setup time | What you end up with |
| --- | --- | --- | --- |
| A live link *right now* | [Option 0 — Streamlit Community Cloud](#option-0--streamlit-community-cloud-fastest) | ~2 min, 6 clicks | `https://<name>.streamlit.app` |
| Free and unlikely to be asleep when someone opens it | [Option 1 — Hugging Face Spaces](#option-1--hugging-face-spaces-docker-sdk) | ~3 min + one command | `https://<user>-<space>.hf.space` |
| A free custom domain | [Option 2 — Render](#option-2--render) | ~5 min | `https://<name>.onrender.com` |
| No wake-up at all, ever | [Option 3 — Always-on VM](#option-3--always-on-vm) | ~15 min | `https://your-domain` |
| Instant by construction | [Option 4 — No server](#option-4--no-server-at-all) | a day (rewrite) | static site, cannot sleep |

Everything below the tables is per-host detail; the repository already contains the config each path needs.

## Options at a glance

Limits verified September 2026 — check the providers, they change.

| Host | Free? | Sleeps after | RAM | Custom domain | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Hugging Face Spaces** (Docker SDK) | yes | **48 h** idle | 2 vCPU / 16 GB | paid | best free ratio: 4× Streamlit Cloud's idle window, generous RAM, direct `*.hf.space` URL |
| **Render** | 750 instance-hours + 5 GB bandwidth/month | **15 min** idle | 512 MB | **free** | only free tier with a custom domain, but it wakes like a cold start (~30–60 s) |
| **Streamlit Community Cloud** | yes | 12 h idle | ~1 GB | no | simplest; see the main README |
| **Oracle Cloud Always Free** (Ampere A1) | yes, forever | **never** | 4 OCPU / 24 GB | yes | the only true "no wake-up"; card required for signup, no charges |
| **Google Cloud e2-micro** (free tier) | yes | **never** | 1 vCPU / 1 GB | yes | always-on but RAM-tight; add swap |
| **Cloudflare Pages / GitHub Pages** | yes | n/a | n/a | yes | static only — cannot run Streamlit (see [the no-server option](#option-4--no-server-at-all)) |
| PHP shared hosts (InfinityFree, iFreeDomains…) | yes | n/a | n/a | yes | cannot run the app; fine as a landing page or redirect |

## Option 0 — Streamlit Community Cloud (fastest)

The simplest path, and everything it needs is already committed: the CPU-only PyTorch index in `requirements.txt`, `.streamlit/config.toml` and `runtime.txt`.

1. Sign in at <https://share.streamlit.io> with GitHub.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repository `sandilya-bit/solar-flare-prediction`, branch `main`, **Main file path** `app.py`.
4. *Advanced settings → Python version*: `3.12`.
5. **Deploy** — the first build takes 3–6 minutes.
6. Rename the link: **⋮ → Settings → General → App URL**.

Afterwards every push to `main` redeploys automatically. It sleeps after 12 hours without traffic (see [keeping any host warm](#keeping-any-host-warm)) and its filesystem is ephemeral, so prediction history resets on rebuild.

## Option 1 — Hugging Face Spaces (Docker SDK)

Free CPU Basic gives 2 vCPU / 16 GB RAM and sleeps only after **48 hours** of inactivity, so an occasional visit keeps it permanently warm. You get both a project page and a direct URL, `https://<user>-<space>.hf.space`, which is the one to hand in as your live link.

1. Create a Space at <https://huggingface.co/new-space> → **Docker** SDK → *Blank* template. Note the name; the repo id is `<user>/<space>`.
2. Create a **write** token at <https://huggingface.co/settings/tokens>.
3. From the repository root, publish:

   ```bash
   HF_TOKEN=hf_xxxxxxxx scripts/deploy_hf.sh <user>/<space>
   ```

   The script exports the tracked files, swaps in the Space front-matter (see `docs/hf/README.md`), and force-pushes. Re-run it after any change to redeploy.
4. First build takes a few minutes; then the Space is live at `https://<user>-<space>.hf.space`.

> The Space's `README.md` **must** carry the YAML front-matter that declares the SDK and port. That is exactly why `docs/hf/README.md` exists separately: pushing this repository's own README over it would strip the metadata and break the Space.

## Option 2 — Render

`render.yaml` in the repository root is a complete blueprint.

1. <https://dashboard.render.com> → **New → Blueprint** → pick this repository → **Apply**. Render builds the `Dockerfile` and starts the service.
2. You get `https://<name>.onrender.com`. Add your own domain under **Settings → Custom Domains** (free): create the `CNAME` it shows at your DNS provider.
3. Health checks use Streamlit's own endpoint, `/_stcore/health`.

Free-plan caveats worth designing around:

- It **spins down after 15 minutes** without inbound traffic. Traffic means HTTP requests *and* WebSocket messages, so a 10-minute ping keeps it genuinely warm — that consumes ~730 of the 750 monthly instance-hours, leaving no headroom for a second service.
- **5 GB of bandwidth per month.** A Streamlit page load moves a few MB, so expect roughly a thousand visits. Beyond that, the service is suspended until the next month.

If a 30–60 second wake on the first visit is acceptable, a single hourly ping is enough and the quota is never a concern.

## Option 3 — Always-on VM

The only way to remove the concept of sleeping. On Oracle's Always Free Ampere A1 (4 OCPU / 24 GB) this dashboard uses a rounding error's worth of the machine; Google's free `e2-micro` works too if you add swap.

The repository ships a `docker-compose.yml` and a `Caddyfile`, so the whole deployment is three commands:

```bash
# on the VM
git clone https://github.com/sandilya-bit/solar-flare-prediction.git
cd solar-flare-prediction
cp .env.example .env        # then set DOMAIN to your hostname
docker compose up -d --build
```

That starts two containers: the dashboard (deliberately **not** published on the host) and **Caddy**, which terminates TLS and obtains plus renews a Let's Encrypt certificate for `DOMAIN` automatically. Once the containers are up, every push to your fork can be deployed with `git pull && docker compose up -d --build`.

Prerequisites: the hostname's `A`/`AAAA` record already points at the machine, and ports 80 and 443 are reachable (`sudo ufw allow 80,443/tcp`). Caddy cannot obtain a certificate for a name that resolves elsewhere.

**No hostname yet?** Either set `DOMAIN=:80` in `.env` and reach the box by IP over plain HTTP — no authority issues certificates for bare IP addresses — or skip domains entirely with a Cloudflare Tunnel, which needs no open ports and gives free HTTPS:

```bash
cloudflared tunnel --url http://127.0.0.1:7860
```

Operational notes:

- `docker compose logs -f caddy` shows certificate and proxy errors; `docker compose pull && docker compose up -d --build` updates the app.
- Prediction history and logs live in named volumes, so `docker compose down` keeps them and `docker compose down -v` erases them.
- Keep the VM patched, and allow outbound traffic to `services.swpc.noaa.gov`; the app needs nothing else.

## Option 4 — No server at all

The most "always live" architecture is one with nothing to wake. Instead of a Python server rendering pages on demand, a scheduled job runs the model and writes a static snapshot (`predictions.json` plus the chart series), and a static site reads it:

```
GitHub Actions (cron, every 10 min)
   └─ fetch NOAA → clean → 60-min window → CNN → predictions.json
Cloudflare Pages / GitHub Pages
   └─ index.html + Plotly.js reads predictions.json   ← instant, forever
```

- **Pros:** no sleep, no cold start, no monthly hours, survives a front-page hug, free custom domain, HTTPS included.
- **Cons:** it is a rewrite of the presentation layer. `assets/style.css` and `assets/mobile.css` carry over unchanged, and the education content ports directly, but the cards, chart and tables become HTML/JS backed by Plotly.js instead of `st.plotly_chart`.
- **Freshness:** bounded by the job interval. For flare nowcasting, 5–15 minutes is comfortably inside the physics.

## Keeping any host warm

| Host | Idle window | Ping interval that keeps it awake |
| --- | --- | --- |
| Streamlit Community Cloud | 12 h | 6 h — shipped in [`keepalive-workflow.yml`](keepalive-workflow.yml) |
| Hugging Face Spaces | 48 h | daily is plenty |
| Render | 15 min | ~10 min (accept the wake if you would rather not) |
| VM | never | not needed |

Any of them can be kept warm without touching the repository, using a free [UptimeRobot](https://uptimerobot.com/) HTTP monitor on the app URL with the interval from the table above.
