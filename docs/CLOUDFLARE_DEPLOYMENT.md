# Deploy Veilcrean to Cloudflare Containers

This setup deploys Veilcrean as a **dynamic Cloudflare Container behind a Cloudflare Worker**. It is not a static Cloudflare Pages deployment.

## What was added

```text
Dockerfile.cloudflare                 # Cloudflare production container image
requirements.cloudflare.txt           # Production Python deps without test/dev tools
cloudflare/entrypoint.py              # Health/status HTTP server + brain supervisor
src/index.ts                          # Cloudflare Worker that routes to the container
wrangler.jsonc                        # Cloudflare Worker + Container config
package.json                          # Wrangler / @cloudflare/containers scripts
.env.cloudflare.example               # Secret reference, no real values
.github/workflows/cloudflare-deploy.yml
```

## Important production reality for Veilcrean

Veilcrean is a trading brain that currently depends on **ZMQ + MT5 EA**:

```text
MT5 EA  <->  ZMQ tcp://127.0.0.1:5555/5556  <->  Python Brain
```

Cloudflare Containers only expose HTTP traffic through the Worker. They do **not** expose arbitrary inbound TCP/UDP ports directly to end users. That means an MT5 EA running on your laptop/VPS cannot simply connect to `tcp://your-worker.workers.dev:5555`.

For a truly live Cloudflare-only production deployment, use one of these patterns:

1. **Direct broker API mode** — adapt Veilcrean to trade via Deriv/WebSocket directly from the container.
2. **Broker bridge/gateway** — run a separate reachable service that converts MT5/ZMQ traffic to HTTPS/WebSocket.
3. **Run the brain beside MT5** — keep the Python brain on the same VPS/machine as MT5 and put Cloudflare in front of a dashboard/API only.

This Cloudflare setup is production-ready for the container/runtime layer, health checks, singleton routing, and deployment pipeline, but the trading connectivity still needs one of the patterns above.

## Filesystem persistence warning

Cloudflare Container disk is ephemeral. Do not rely on container-local files for live production persistence:

```text
data/trade_journal.db
models/*.pt
logs/*.log
training outputs
```

For production durability, move these to durable storage:

- D1 / external Postgres for journal and trades
- R2 for model files, logs, training outputs, and historical datasets
- KV / Durable Objects for small state and coordination

## Why `max_instances = 1`

`wrangler.jsonc` intentionally sets:

```jsonc
"max_instances": 1
```

For a trading bot, multiple active containers could duplicate decisions and trades. Do **not** increase this unless you intentionally shard by account/symbol and add idempotency controls.

## Prerequisites

Install locally:

- Node.js 22+
- Docker Desktop / Colima / Docker Engine
- Cloudflare account with Workers + Containers enabled
- Wrangler login

```bash
npm install
npx wrangler login
```

## Configure secrets

The Worker blocks `/status`, `/logs`, `/metrics`, and `/__worker` unless a status API key is configured.

```bash
openssl rand -hex 32
npx wrangler secret put VEIL_STATUS_API_KEY
```

Optional secrets:

```bash
npx wrangler secret put DERIV_API_TOKEN
npx wrangler secret put GROQ_API_KEY
npx wrangler secret put GEMINI_API_KEY
npx wrangler secret put VEIL_TG_TOKEN
npx wrangler secret put VEIL_TG_CHAT
npx wrangler secret put VEIL_DISCORD_HOOK
```

Non-secret values live in `wrangler.jsonc`:

```jsonc
"DERIV_APP_ID": "",
"DERIV_ENABLED": "false",
"DERIV_IS_DEMO": "true",
"VEIL_LLM_PROVIDER": "groq",
"VEIL_LLM_ENABLED": "false"
```

## Deploy

```bash
npm install
npx wrangler deploy
```

First deployment may take a few minutes because Wrangler builds and uploads the Docker image.

Check container rollout:

```bash
npx wrangler containers list
npx wrangler containers images list
```

## Test endpoints

Public liveness:

```bash
curl https://veilcrean-production.<your-subdomain>.workers.dev/healthz
```

Authenticated status:

```bash
export VEIL_STATUS_API_KEY="your-value"

curl \
  -H "Authorization: Bearer $VEIL_STATUS_API_KEY" \
  https://veilcrean-production.<your-subdomain>.workers.dev/status
```

Logs:

```bash
curl \
  -H "Authorization: Bearer $VEIL_STATUS_API_KEY" \
  "https://veilcrean-production.<your-subdomain>.workers.dev/logs?lines=200"
```

Worker-only diagnostics:

```bash
curl \
  -H "Authorization: Bearer $VEIL_STATUS_API_KEY" \
  https://veilcrean-production.<your-subdomain>.workers.dev/__worker
```

## Safe first deployment

If you want to deploy without starting the trading brain yet, set this in `wrangler.jsonc` before deploying:

```jsonc
"VEIL_START_BRAIN": "false"
```

Deploy, verify `/healthz` and `/status`, then switch it back to:

```jsonc
"VEIL_START_BRAIN": "true"
```

## GitHub Actions deployment

The workflow `.github/workflows/cloudflare-deploy.yml` deploys on pushes to `main` and manual runs.

Create GitHub repository secrets:

```text
CLOUDFLARE_API_TOKEN
CLOUDFLARE_ACCOUNT_ID
```

The Cloudflare API token should have enough permission to deploy Workers and Containers for your account.

## Custom domain

After deploy, add a route/custom domain in Cloudflare dashboard or extend `wrangler.jsonc` with your route once DNS is on Cloudflare.

Example:

```jsonc
"routes": [
  { "pattern": "veilcrean.example.com/*", "custom_domain": true }
]
```

## Operational checklist

Before live trading:

- [ ] Revoke any temporary GitHub token used during setup.
- [ ] Confirm only one singleton container can trade.
- [ ] Decide how MT5/ZMQ or Deriv connectivity will work from the cloud.
- [ ] Move journal/model persistence out of the container filesystem.
- [ ] Keep `VEIL_REQUIRE_API_KEY=true`.
- [ ] Set Cloudflare WAF/rate limiting on status routes if exposed publicly.
- [ ] Start on a demo account first.
- [ ] Monitor Cloudflare container logs and `/status` after deployment.
