# Veilcrean on Vercel + Supabase

This repo is now prepared for a **Vercel + Supabase** architecture.

## Architecture

```text
Veilcrean Python Brain
  runs on your machine/VPS near MT5 or broker access
  ↓ HTTPS
Vercel FastAPI Serverless API
  lightweight authenticated API
  ↓ Supabase REST
Supabase Postgres
  persistent status/events/signals/trades
```

Vercel does **not** run the full trading brain. Vercel is serverless and is not suited for long-running PyTorch/ZMQ trading loops. The full brain still runs wherever broker/MT5 connectivity exists. Vercel hosts the public/private API layer and Supabase stores the data.

## Added files

```text
api/index.py                 # repo-root Vercel entrypoint wrapper
vercel.json                  # repo-root Vercel config
pyproject.toml               # explicit Vercel Python entrypoint
requirements.txt             # lightweight Vercel dependencies
requirements.brain.txt       # full trading-brain dependencies

vercel_app/
  api/index.py              # FastAPI serverless API
  requirements.txt          # lightweight Vercel-only dependencies
  vercel.json               # Vercel routing/build config
  .env.example              # Vercel env reference
  README.md
  supabase/schema.sql       # schema copy

supabase/schema.sql         # canonical Supabase schema
python_brain/integrations/vercel_supabase.py
python_brain/integrations/__init__.py
```

`python_brain/main.py` now initializes `VercelSupabaseBridge`. If `VEIL_VERCEL_API_URL` and `VEIL_API_KEY` are set, it posts status and trade-open events to the Vercel API. If those variables are missing, it stays disabled and the bot still works normally.

## Step 1 — Create Supabase project

1. Go to Supabase.
2. Create a project.
3. Open **SQL Editor**.
4. Paste and run:

```text
supabase/schema.sql
```

This creates:

```text
bot_events
bot_status
trade_signals
trade_journal
latest_bot_status view
latest_trade_signals view
```

RLS is enabled with no public policies. The Vercel API uses the Supabase service role key server-side.

## Step 2 — Get Supabase keys

In Supabase:

```text
Project Settings → API
```

Copy:

```text
Project URL
service_role key
```

Never expose the service role key in browser/client code.

## Step 3 — Deploy to Vercel

Import the GitHub repo into Vercel.

Set project settings:

```text
Root Directory: vercel_app
Framework Preset: Other
Build Command: leave empty
Output Directory: leave empty
Install Command: pip install -r requirements.txt
```

The repo root is also Vercel-ready now. If you deploy from `vercel_app`, use this folder's `requirements.txt`. If you deploy from the repo root, the root `api/index.py` wrapper and root `requirements.txt` route to the same lightweight Vercel API.

## Step 4 — Add Vercel environment variables

In Vercel:

```text
Project → Settings → Environment Variables
```

Add:

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
TELEGRAM_TIMEOUT_SECONDS=8
```

Generate `VEIL_API_KEY`:

```bash
openssl rand -hex 32
```

## Step 5 — Test Vercel API

Health endpoint:

```bash
curl https://your-vercel-project.vercel.app/api/health
```

Insert test event:

```bash
curl -X POST https://your-vercel-project.vercel.app/api/events \
  -H "Authorization: Bearer YOUR_VEIL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"event_type":"deploy_test","payload":{"ok":true}}'
```

Read events:

```bash
curl https://your-vercel-project.vercel.app/api/events \
  -H "Authorization: Bearer YOUR_VEIL_API_KEY"
```

If you want read endpoints public, set:

```text
VEIL_PUBLIC_READ=true
```

For personal trading data, keep it false.

## Step 6 — Telegram bot setup

Create a bot:

1. Open Telegram.
2. Message `@BotFather`.
3. Run `/newbot`.
4. Copy the bot token into Vercel as `TELEGRAM_BOT_TOKEN`.

Generate a webhook secret:

```bash
openssl rand -hex 32
```

Set it in Vercel as:

```text
TELEGRAM_WEBHOOK_SECRET=the-generated-secret
```

Set the Telegram webhook:

```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=https://your-vercel-project.vercel.app/api/telegram/webhook" \
  -d "secret_token=$TELEGRAM_WEBHOOK_SECRET"
```

Find your chat ID:

1. Temporarily leave `TELEGRAM_ALLOWED_CHAT_IDS` empty.
2. Send `/id` to your bot.
3. Copy the returned chat id.
4. Set `TELEGRAM_ALLOWED_CHAT_IDS` in Vercel to that value.
5. Redeploy/restart the Vercel project.

Recommended: keep `TELEGRAM_ALLOWED_CHAT_IDS` set. Otherwise anyone who finds your bot can query or queue commands.

Telegram commands:

```text
/start or /help       Show help
/id                   Show current chat id
/health               API health
/status               Latest Veilcrean status
/signals [n]          Latest signals
/trades [n]           Latest trades
/events [n]           Latest events
/queue                Recent command queue
/pause [reason]       Queue PAUSE command
/resume [reason]      Queue RESUME command
/flatten [symbol]     Queue FLATTEN_ALL command
/command NAME [args]  Queue custom command
```

`/pause` and `/resume` are processed by the Python brain if command polling is enabled. `/flatten` is only executed by the Python brain if this is set on the brain host:

```text
VEIL_ENABLE_REMOTE_TRADE_COMMANDS=true
```

Keep it `false` unless you intentionally want Telegram to be able to send flatten-all to the EA.

## Step 7 — Connect the Python brain

Wherever the real Veilcrean brain runs, set:

```text
VEIL_VERCEL_API_URL=https://your-vercel-project.vercel.app
VEIL_API_KEY=the-same-key-from-vercel
```

Then run the brain normally:

```bash
python -m python_brain.main
```

When the brain publishes status or opens a trade, it will post to Vercel, and Vercel will write to Supabase.

## API endpoints

### Public

```text
GET /api/health
GET /
GET /api/docs
```

### Authenticated writes

Use one of:

```text
Authorization: Bearer YOUR_VEIL_API_KEY
x-api-key: YOUR_VEIL_API_KEY
```

Endpoints:

```text
POST /api/events
POST /api/status
POST /api/signals
POST /api/trades
POST /api/ingest
POST /api/commands
POST /api/commands/{command_id}/complete
POST /api/telegram/webhook
```

### Authenticated reads unless `VEIL_PUBLIC_READ=true`

```text
GET /api/events
GET /api/status/latest
GET /api/signals/latest
GET /api/trades
GET /api/config
GET /api/commands
GET /api/commands/pending
GET /api/telegram/info
```

## Limitations

Vercel is serverless. It should not run:

```text
python -m python_brain.main
MT5/ZMQ listener
PyTorch model training
long-running trading loops
background workers
```

Use Vercel for the API/dashboard layer and Supabase for persistent storage. Run the actual trading brain on a VPS/local machine/broker-connected host.
