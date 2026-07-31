# Veilcrean Vercel + Supabase API

This folder is the Vercel project root for Veilcrean's serverless API layer.

It is intentionally separate from the full trading brain. The repo root is also Vercel-ready through `api/index.py`, but this folder can be used as the Vercel Root Directory if you prefer a cleaner Vercel-only project root.

## What this deploys

A FastAPI serverless API that stores Veilcrean data in Supabase:

- `GET /api/health`
- `POST /api/status`
- `GET /api/status/latest`
- `POST /api/events`
- `GET /api/events`
- `POST /api/signals`
- `GET /api/signals/latest`
- `POST /api/trades`
- `GET /api/trades`
- `POST /api/ingest`
- `POST /api/commands`
- `GET /api/commands`
- `POST /api/telegram/webhook`
- `GET /api/telegram/info`
- `GET /api/docs`

## Vercel settings

When importing the GitHub repo into Vercel, set:

```text
Root Directory: vercel_app
Framework Preset: Other
Build Command: leave empty
Output Directory: leave empty
Install Command: pip install -r requirements.txt
```

## Environment variables

Set these in Vercel:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
VEIL_API_KEY=long-random-secret
SUPABASE_SCHEMA=public
VEIL_PUBLIC_READ=false
CORS_ORIGINS=*

# Optional Telegram bot
TELEGRAM_BOT_TOKEN=token-from-botfather
TELEGRAM_WEBHOOK_SECRET=long-random-webhook-secret
TELEGRAM_ALLOWED_CHAT_IDS=your-telegram-chat-id
```

Generate `VEIL_API_KEY` with:

```bash
openssl rand -hex 32
```

## Supabase schema

Run the SQL in:

```text
../supabase/schema.sql
```

or the copy at:

```text
supabase/schema.sql
```

inside the Supabase SQL Editor.

## Test after deploy

```bash
curl https://your-vercel-project.vercel.app/api/health

curl -X POST https://your-vercel-project.vercel.app/api/events \
  -H "Authorization: Bearer $VEIL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"event_type":"deploy_test","payload":{"ok":true}}'
```

## Connect the Veilcrean Python brain

Wherever the actual long-running brain runs, set:

```text
VEIL_VERCEL_API_URL=https://your-vercel-project.vercel.app
VEIL_API_KEY=the-same-key-set-in-vercel
```

Then Veilcrean will post status/trade events to this API automatically.
