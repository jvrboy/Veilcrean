-- Veilcrean Supabase schema
-- Run this in Supabase SQL Editor before deploying the Vercel API.

create extension if not exists pgcrypto;

create table if not exists public.bot_events (
  id uuid primary key default gen_random_uuid(),
  source text not null default 'veilcrean',
  event_type text not null default 'event',
  symbol text,
  severity text not null default 'info',
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.bot_status (
  id uuid primary key default gen_random_uuid(),
  source text not null default 'veilcrean',
  status jsonb not null default '{}'::jsonb,
  win_rate numeric,
  profit_factor numeric,
  sharpe numeric,
  max_dd_pct numeric,
  trades integer,
  threshold numeric,
  kill_switch boolean,
  regime text,
  confidence numeric,
  created_at timestamptz not null default now()
);

create table if not exists public.trade_signals (
  id uuid primary key default gen_random_uuid(),
  signal_id text,
  source text not null default 'veilcrean',
  symbol text not null,
  timeframe text,
  action text not null,
  confidence numeric,
  price numeric,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.trade_journal (
  id uuid primary key default gen_random_uuid(),
  trade_id text,
  source text not null default 'veilcrean',
  symbol text not null,
  direction text,
  status text not null default 'open',
  opened_at timestamptz,
  closed_at timestamptz,
  entry_price numeric,
  exit_price numeric,
  sl numeric,
  tp numeric,
  lots numeric,
  pnl numeric,
  pnl_pct numeric,
  confidence numeric,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists bot_events_created_at_idx on public.bot_events (created_at desc);
create index if not exists bot_events_type_idx on public.bot_events (event_type);
create index if not exists bot_events_symbol_idx on public.bot_events (symbol);

create index if not exists bot_status_created_at_idx on public.bot_status (created_at desc);
create index if not exists trade_signals_created_at_idx on public.trade_signals (created_at desc);
create index if not exists trade_signals_symbol_idx on public.trade_signals (symbol);
create index if not exists trade_journal_created_at_idx on public.trade_journal (created_at desc);
create index if not exists trade_journal_symbol_idx on public.trade_journal (symbol);
create index if not exists trade_journal_trade_id_idx on public.trade_journal (trade_id);

create or replace view public.latest_bot_status as
select *
from public.bot_status
order by created_at desc
limit 1;

create or replace view public.latest_trade_signals as
select *
from public.trade_signals
order by created_at desc
limit 50;

-- RLS is enabled and no public policies are created. The Vercel API should use
-- SUPABASE_SERVICE_ROLE_KEY server-side. Do not expose the service role key in a browser.
alter table public.bot_events enable row level security;
alter table public.bot_status enable row level security;
alter table public.trade_signals enable row level security;
alter table public.trade_journal enable row level security;
