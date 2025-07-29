/* app/predict/page.tsx
   ─────────────────────────────────────────────────────────────
   Minimal “Try the predictor” page
   • Collects inputs
   • POSTS JSON to /api/score
   • Renders probability + basic interpretation
   • Everything is client‑side except the fetch()
*/

"use client";

import { useState } from "react";
import Link from "next/link";

type ApiResponse = { prob: number };

export default function PredictPage() {
  // ── Controlled inputs ───────────────────────────────
  const [sleepMin, setSleepMin] = useState("");
  const [fragmentation, setFragmentation] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<"M" | "F" | "">("");

  // ── UI state ─────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [prob, setProb]       = useState<number | null>(null);

  // ── Helpers ──────────────────────────────────────────
  function valid() {
    return (
      sleepMin !== "" &&
      fragmentation !== "" &&
      age !== "" &&
      sex !== ""
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid()) return;

    setLoading(true);
    setError(null);
    setProb(null);

    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sleep_minutes: Number(sleepMin),
          frag_index:    Number(fragmentation),
          age:           Number(age),
          sex
        }),
      });

      if (!res.ok) throw new Error(`API ${res.status}`);

      const data: ApiResponse = await res.json();
      setProb(data.prob);
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // ── Render ───────────────────────────────────────────
  return (
    <main className="mx-auto max-w-xl px-4 py-16">
      <Link href="/" className="text-sm text-indigo-600 hover:underline">
        ← Home
      </Link>

      <h1 className="mt-4 text-3xl font-bold tracking-tight text-zinc-800 dark:text-zinc-100">
        Depression‑risk predictor
      </h1>
      <p className="mt-2 text-zinc-600 dark:text-zinc-300">
        Enter the most recent weekly averages from your wearable. <br />
        We’ll estimate the probability of developing clinically
        relevant depressive symptoms.
      </p>

      <form
        onSubmit={handleSubmit}
        className="mt-8 grid gap-6 rounded-xl bg-white/80 p-6 shadow-lg
                   dark:bg-zinc-900/40 dark:shadow-zinc-800"
      >
        {/* Sleep minutes */}
        <div>
          <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Average nightly in‑bed minutes
          </label>
          <input
            type="number"
            required
            min={240}
            max={960}
            value={sleepMin}
            onChange={(e) => setSleepMin(e.target.value)}
            className="mt-1 w-full rounded-md border border-zinc-300
                       bg-white/70 p-2 text-zinc-900 shadow-sm
                       focus:border-indigo-500 focus:ring-indigo-500
                       dark:border-zinc-700 dark:bg-zinc-800
                       dark:text-zinc-100"
          />
        </div>

        {/* Fragmentation index */}
        <div>
          <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Sleep fragmentation index&nbsp;(%) 
            <span className="text-xs text-zinc-400">(0–100)</span>
          </label>
          <input
            type="number"
            required
            min={0}
            max={100}
            step={0.1}
            value={fragmentation}
            onChange={(e) => setFragmentation(e.target.value)}
            className="mt-1 w-full rounded-md border border-zinc-300
                       bg-white/70 p-2 text-zinc-900 shadow-sm
                       focus:border-indigo-500 focus:ring-indigo-500
                       dark:border-zinc-700 dark:bg-zinc-800
                       dark:text-zinc-100"
          />
        </div>

        {/* Age */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-200">
              Age
            </label>
            <input
              type="number"
              required
              min={18}
              max={90}
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="mt-1 w-full rounded-md border border-zinc-300
                         bg-white/70 p-2 text-zinc-900 shadow-sm
                         focus:border-indigo-500 focus:ring-indigo-500
                         dark:border-zinc-700 dark:bg-zinc-800
                         dark:text-zinc-100"
            />
          </div>

          {/* Sex */}
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-200">
              Sex
            </label>
            <select
              required
              value={sex}
              onChange={(e) => setSex(e.target.value as "M" | "F")}
              className="mt-1 w-full rounded-md border border-zinc-300
                         bg-white/70 p-2 text-zinc-900 shadow-sm
                         focus:border-indigo-500 focus:ring-indigo-500
                         dark:border-zinc-700 dark:bg-zinc-800
                         dark:text-zinc-100"
            >
              <option value="">–</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={loading || !valid()}
          className="inline-flex items-center justify-center rounded-md
                     bg-indigo-600 px-4 py-2 font-medium text-white
                     hover:bg-indigo-700 disabled:opacity-40
                     dark:bg-indigo-500 dark:hover:bg-indigo-400"
        >
          {loading ? "Scoring…" : "Predict risk"}
        </button>
      </form>

      {/* Result / error */}
      {prob !== null && (
        <div
          className="mt-8 rounded-lg border-l-4 border-indigo-600 bg-indigo-50
                     p-4 text-indigo-800 dark:border-indigo-500
                     dark:bg-indigo-950/40 dark:text-indigo-300"
        >
          <p className="text-lg font-semibold">
            Estimated 2‑year risk: {Math.round(prob * 1000) / 10}%
          </p>
          <p className="text-sm">
            <span className="font-medium">Interpretation:</span>{" "}
            {prob < 0.10
              ? "Low"
              : prob < 0.25
              ? "Moderate"
              : prob < 0.40
              ? "Elevated"
              : "High"} risk versus cohort baseline.
          </p>
        </div>
      )}

      {error && (
        <div
          className="mt-8 rounded-lg border-l-4 border-red-600 bg-red-50
                     p-4 text-sm text-red-800 dark:border-red-500
                     dark:bg-red-900/40 dark:text-red-300"
        >
          {error}
        </div>
      )}
    </main>
  );
}
