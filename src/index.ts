import { Container, getContainer } from "@cloudflare/containers";

export interface Env {
  VEILCREAN_CONTAINER: DurableObjectNamespace<VeilcreanContainer>;

  // Non-secret vars from wrangler.jsonc
  PORT?: string;
  VEIL_ENV?: string;
  VEIL_START_BRAIN?: string;
  VEIL_EXIT_ON_BRAIN_STOP?: string;
  VEIL_BRAIN_AUTO_RESTART?: string;
  VEIL_BRAIN_CMD?: string;
  VEIL_REQUIRE_API_KEY?: string;
  VEIL_ZMQ_PUB?: string;
  VEIL_ZMQ_PULL?: string;
  VEIL_ZMQ_STATUS?: string;
  VEIL_ZMQ_RECV_TIMEOUT_MS?: string;
  VEIL_ZMQ_SEND_TIMEOUT_MS?: string;
  DERIV_APP_ID?: string;
  DERIV_ENABLED?: string;
  DERIV_IS_DEMO?: string;
  VEIL_LLM_PROVIDER?: string;
  VEIL_LLM_ENABLED?: string;

  // Secrets set with `wrangler secret put ...`
  VEIL_STATUS_API_KEY?: string;
  DERIV_API_TOKEN?: string;
  GROQ_API_KEY?: string;
  GEMINI_API_KEY?: string;
  VEIL_TG_TOKEN?: string;
  VEIL_TG_CHAT?: string;
  VEIL_DISCORD_HOOK?: string;
}

const SINGLETON_ID = "production-singleton";
const PUBLIC_PATHS = new Set(["/health", "/healthz", "/livez", "/favicon.ico", "/robots.txt"]);

function envOr(env: Env, key: keyof Env, fallback = ""): string {
  const value = env[key];
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json; charset=utf-8");
  headers.set("cache-control", "no-store");
  return new Response(JSON.stringify(body, null, 2), { ...init, headers });
}

function getSingleton(env: Env) {
  return getContainer(env.VEILCREAN_CONTAINER, SINGLETON_ID);
}

function unauthorized(message = "Unauthorized"): Response {
  return jsonResponse(
    {
      error: "unauthorized",
      message,
      usage: "Send Authorization: Bearer <VEIL_STATUS_API_KEY> or x-api-key: <VEIL_STATUS_API_KEY>.",
    },
    { status: 401, headers: { "www-authenticate": "Bearer" } },
  );
}

function checkAuth(request: Request, env: Env): Response | null {
  const url = new URL(request.url);
  if (PUBLIC_PATHS.has(url.pathname)) {
    return null;
  }

  const requireKey = envOr(env, "VEIL_REQUIRE_API_KEY", "true").toLowerCase() !== "false";
  if (!requireKey) {
    return null;
  }

  const expected = env.VEIL_STATUS_API_KEY;
  if (!expected) {
    return jsonResponse(
      {
        error: "missing_secret",
        message: "VEIL_STATUS_API_KEY is required because VEIL_REQUIRE_API_KEY is enabled.",
        fix: "Run: npx wrangler secret put VEIL_STATUS_API_KEY",
      },
      { status: 503 },
    );
  }

  const auth = request.headers.get("authorization") || "";
  const bearer = auth.match(/^Bearer\s+(.+)$/i)?.[1];
  const apiKey = request.headers.get("x-api-key");
  if (bearer === expected || apiKey === expected) {
    return null;
  }
  return unauthorized();
}

async function proxyToContainer(request: Request, env: Env): Promise<Response> {
  const container = getSingleton(env);
  return container.fetch(request);
}

async function keepWarm(env: Env): Promise<void> {
  const container = getSingleton(env);
  const res = await container.fetch(new Request("http://veilcrean.internal/healthz"));
  console.log("Veilcrean keep-warm", res.status);
}

export class VeilcreanContainer extends Container {
  defaultPort = 8080;

  // Keep the singleton brain alive between requests. The cron trigger in
  // wrangler.jsonc pings /healthz every five minutes as an extra guard.
  sleepAfter = "24h";

  envVars = {
    PORT: "8080",
    VEIL_ENV: envOr(this.env as Env, "VEIL_ENV", "production"),
    VEIL_CF_MODE: "true",
    VEIL_START_BRAIN: envOr(this.env as Env, "VEIL_START_BRAIN", "true"),
    VEIL_EXIT_ON_BRAIN_STOP: envOr(this.env as Env, "VEIL_EXIT_ON_BRAIN_STOP", "true"),
    VEIL_BRAIN_AUTO_RESTART: envOr(this.env as Env, "VEIL_BRAIN_AUTO_RESTART", "false"),
    VEIL_BRAIN_CMD: envOr(this.env as Env, "VEIL_BRAIN_CMD", "python -m python_brain.main"),

    // ZMQ defaults work only if the EA bridge is inside the same container.
    // For a remote MT5/EA bridge, set these to reachable TCP endpoints.
    VEIL_ZMQ_PUB: envOr(this.env as Env, "VEIL_ZMQ_PUB", "tcp://127.0.0.1:5555"),
    VEIL_ZMQ_PULL: envOr(this.env as Env, "VEIL_ZMQ_PULL", "tcp://127.0.0.1:5556"),
    VEIL_ZMQ_STATUS: envOr(this.env as Env, "VEIL_ZMQ_STATUS", "tcp://127.0.0.1:5557"),
    VEIL_ZMQ_RECV_TIMEOUT_MS: envOr(this.env as Env, "VEIL_ZMQ_RECV_TIMEOUT_MS", "1000"),
    VEIL_ZMQ_SEND_TIMEOUT_MS: envOr(this.env as Env, "VEIL_ZMQ_SEND_TIMEOUT_MS", "1000"),

    DERIV_APP_ID: envOr(this.env as Env, "DERIV_APP_ID", ""),
    DERIV_API_TOKEN: envOr(this.env as Env, "DERIV_API_TOKEN", ""),
    DERIV_ENABLED: envOr(this.env as Env, "DERIV_ENABLED", "false"),
    DERIV_IS_DEMO: envOr(this.env as Env, "DERIV_IS_DEMO", "true"),

    VEIL_LLM_PROVIDER: envOr(this.env as Env, "VEIL_LLM_PROVIDER", "groq"),
    VEIL_LLM_ENABLED: envOr(this.env as Env, "VEIL_LLM_ENABLED", "false"),
    GROQ_API_KEY: envOr(this.env as Env, "GROQ_API_KEY", ""),
    GEMINI_API_KEY: envOr(this.env as Env, "GEMINI_API_KEY", ""),

    VEIL_TG_TOKEN: envOr(this.env as Env, "VEIL_TG_TOKEN", ""),
    VEIL_TG_CHAT: envOr(this.env as Env, "VEIL_TG_CHAT", ""),
    VEIL_DISCORD_HOOK: envOr(this.env as Env, "VEIL_DISCORD_HOOK", ""),
  };

  onStart() {
    console.log("Veilcrean container started");
  }

  onStop() {
    console.log("Veilcrean container stopped");
  }

  onError(error: unknown) {
    console.error("Veilcrean container error", error);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/__worker") {
      const authError = checkAuth(request, env);
      if (authError) return authError;
      return jsonResponse({
        app: "Veilcrean",
        worker: "ok",
        container_id: SINGLETON_ID,
        routes: ["/healthz", "/readyz", "/status", "/logs", "/metrics"],
      });
    }

    const authError = checkAuth(request, env);
    if (authError) return authError;

    if (url.pathname === "/robots.txt") {
      return new Response("User-agent: *\nDisallow: /\n", {
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    return proxyToContainer(request, env);
  },

  async scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(keepWarm(env));
  },
};
