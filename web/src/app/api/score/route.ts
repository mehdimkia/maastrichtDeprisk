// web/src/app/api/score/route.ts
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

export const runtime = "nodejs";          // run on Node (not edge)
export const dynamic = "force-dynamic";   // no caching of responses

// ---- Env (server-only) -----------------------------------------------
const BASE =
  (process.env.MODEL_API_URL || process.env.NEXT_PUBLIC_API_URL || "")
    .trim()
    .replace(/\/+$/, ""); // strip trailing slash

const API_KEY = process.env.MODEL_API_KEY ?? "";

// ---- Types ------------------------------------------------------------
type ScoreRequest = {
  sleep_minutes: number;
  frag_index: number;
  age: number;
  sex: "M" | "F";
  [k: string]: unknown;
};

type ScoreResult = { prob: number };

// Small helper to make a timeout signal without using `any`
function timeoutSignal(ms: number): AbortSignal | undefined {
  if (typeof AbortController === "undefined") return undefined;
  const ctrl = new AbortController();
  setTimeout(() => ctrl.abort(), ms);
  return ctrl.signal;
}

function fail(message: string, status = 500) {
  console.error(`[api/score] ${message}`);
  return NextResponse.json({ error: message }, { status });
}

// ---- Route handler ----------------------------------------------------
export async function POST(req: NextRequest) {
  if (!BASE) return fail("Missing API base URL. Set MODEL_API_URL in Vercel.");

  let payload: ScoreRequest;
  try {
    payload = (await req.json()) as ScoreRequest;
  } catch {
    return fail("Invalid JSON body", 400);
  }

  try {
    const resp = await fetch(`${BASE}/score`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(API_KEY ? { "x-api-key": API_KEY } : {}),
      },
      cache: "no-store",
      signal: timeoutSignal(10_000),
      body: JSON.stringify(payload),
    });

    const text = await resp.text();

    if (!resp.ok) {
      console.error("[api/score] upstream error:", resp.status, text);
      return NextResponse.json(
        { error: text || `Upstream ${resp.status}` },
        { status: 502 }
      );
    }

    const data = JSON.parse(text) as ScoreResult;
    return NextResponse.json(data);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[api/score] fetch failed:", msg);
    return NextResponse.json({ error: `Proxy error: ${msg}` }, { status: 502 });
  }
}

// Optional: block other verbs
export function GET() {
  return NextResponse.json({ error: "Method Not Allowed" }, { status: 405 });
}
