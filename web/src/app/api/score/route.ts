import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

// ── Environment variables (never leak to client) ────────────────
//   .env.local
//   MODEL_API_URL=https://deprisk.limburg.ai
//   MODEL_API_KEY=   # optional
const FASTAPI_URL = process.env.MODEL_API_URL!;
const FASTAPI_KEY = process.env.MODEL_API_KEY ?? "";

/**
 * POST /api/score
 * Body: { sleep_minutes, frag_index, age, sex, … }
 * → { prob: 0.23 }
 */
export async function POST(req: NextRequest) {
  try {
    // 1 · parse incoming JSON
    const payload = await req.json();

    // Basic sanity guard
    if (typeof payload !== "object" || payload == null) {
      return NextResponse.json(
        { error: "Invalid JSON body" },
        { status: 400 }
      );
    }

    // 2 · forward to FastAPI
    const upstream = await fetch(`${FASTAPI_URL}/score`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(FASTAPI_KEY ? { "x-api-key": FASTAPI_KEY } : {}),
      },
      body: JSON.stringify(payload),
      // 10‑s safety timeout via AbortController
      signal: AbortSignal.timeout?.(10_000),
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      return NextResponse.json(
        { error: `Upstream ${upstream.status}: ${text.slice(0, 120)}` },
        { status: 502 }
      );
    }

    // 3 · relay probability to the client
    const { prob } = (await upstream.json()) as { prob: number };
    return NextResponse.json({ prob });
  } catch (err: any) {
    return NextResponse.json(
      { error: err.message || "Internal error" },
      { status: 500 }
    );
  }
}

/* Optional: block every other verb */
export const GET = () =>
  NextResponse.json({ error: "Method Not Allowed" }, { status: 405 });
