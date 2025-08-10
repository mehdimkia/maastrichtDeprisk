/* app/predict/page.tsx — fact-checked + advanced fields + working tooltip */

"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";

import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type ApiResponse = { prob: number };

export default function PredictPage() {
  // --- core inputs ----------------------------------------------------------
  const [sleepMin, setSleepMin] = useState<number>(480); // 8h default
  const [fragQ, setFragQ] = useState<"1" | "2" | "3" | "4" | "">("");
  const [age, setAge] = useState<number>(60);
  const [sex, setSex] = useState<"M" | "F" | "">("");

  // --- advanced predictors --------------------------------------------------
  const [historyDepression, setHistoryDepression] = useState(false);
  const [antidepressantUse, setAntidepressantUse] = useState(false);
  const [neuropathy, setNeuropathy] = useState(false);

  // --- ui state -------------------------------------------------------------
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prob, setProb] = useState<number | null>(null);

  const anyVuln = historyDepression || antidepressantUse;

  const formValid = useMemo(
    () => sleepMin > 0 && !!fragQ && age >= 40 && age <= 75 && !!sex,
    [sleepMin, fragQ, age, sex]
  );

  function resetForm() {
    setSleepMin(480);
    setFragQ("");
    setAge(60);
    setSex("");
    setHistoryDepression(false);
    setAntidepressantUse(false);
    setNeuropathy(false);
    setProb(null);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!formValid) return;

    setLoading(true);
    setError(null);
    setProb(null);

    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sleep_minutes: sleepMin,
          frag_index: Number(fragQ), // 1–4 (Q1–Q4)
          age,
          sex,
          any_vuln: Number(anyVuln),    // prior depression OR antidepressant use
          neuropathy: Number(neuropathy),
        }),
      });

      if (!res.ok) throw new Error(`API ${res.status}`);
      const data: ApiResponse = await res.json();
      setProb(data.prob);
    } catch (err: any) {
      setError(err?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const pct = useMemo(
    () => (prob !== null ? Math.round(prob * 1000) / 10 : null),
    [prob]
  );

  return (
    <div className="relative isolate">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-br from-indigo-50 via-white to-sky-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-900" />
      <TooltipProvider delayDuration={150}>
        <motion.main
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="mx-auto max-w-2xl px-4 py-16 sm:py-20"
        >
          <Link href="/" className="text-sm text-indigo-600 hover:underline">
            ← Home
          </Link>

          <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-zinc-100">
            Depression-risk predictor
          </h1>
          <p className="mt-2 text-zinc-600 dark:text-zinc-300">
            Enter your recent weekly averages. We’ll estimate your{" "}
            <strong>~4-year risk</strong> of developing{" "}
            <strong>clinically relevant depressive symptoms (PHQ-9 ≥ 10)</strong>.
          </p>

          <form
            onSubmit={handleSubmit}
            className="mt-8 space-y-7 rounded-2xl border border-white/60 bg-white/70 p-6 shadow-xl backdrop-blur dark:border-white/10 dark:bg-zinc-900/60"
          >
            {/* Sleep minutes */}
            <div>
              <label className="mb-2 flex items-center gap-2 text-sm font-medium text-zinc-800 dark:text-zinc-200">
                Average nightly in-bed minutes
                <span className="text-xs text-zinc-400">({sleepMin} min)</span>
              </label>
              <Slider
                min={240}
                max={960}
                step={10}
                value={[sleepMin]}
                onValueChange={(v) => setSleepMin(v[0])}
              />
              <p className="mt-1 text-xs text-zinc-500">
                Derived from your wearable (total time in bed per night).
              </p>
            </div>

            {/* Fragmentation quartile */}
            <div>
              <div className="mb-2 flex items-center gap-2">
                <label className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                  Sleep fragmentation (quartile)
                </label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-zinc-300 text-[11px] font-semibold text-zinc-500 hover:text-zinc-700 dark:border-zinc-600 dark:text-zinc-300"
                      aria-label="What is sleep fragmentation?"
                    >
                      i
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Average number of ≥1-minute posture-transition breaks per
                    night. Approx. cohort cut-points: Q1 ≤ 1.0 · Q2 ≈ 1.0–1.86 ·
                    Q3 ≈ 1.86–2.83 · Q4 ≥ 2.83 breaks/night.
                  </TooltipContent>
                </Tooltip>
              </div>

              <Select
                value={fragQ}
                onValueChange={(v: "1" | "2" | "3" | "4") => setFragQ(v)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select quartile (Q1–Q4)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Q1 — least fragmented</SelectItem>
                  <SelectItem value="2">Q2</SelectItem>
                  <SelectItem value="3">Q3</SelectItem>
                  <SelectItem value="4">Q4 — most fragmented</SelectItem>
                </SelectContent>
              </Select>

              <details className="mt-2 text-xs text-zinc-500 open:pb-2">
                <summary className="cursor-pointer select-none text-zinc-600 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200">
                  What does “fragmentation” mean?
                </summary>
                <div className="mt-1">
                  It’s the average count of ≥1-minute posture-transition breaks
                  per night measured by your wearable. Higher quartile = more fragmented sleep.
                </div>
              </details>
            </div>

            {/* Age */}
            <div>
              <label className="mb-2 flex items-center gap-2 text-sm font-medium text-zinc-800 dark:text-zinc-200">
                Age
                <span className="text-xs text-zinc-400">({age} y)</span>
              </label>
              <Slider
                min={40}
                max={75}
                step={1}
                value={[age]}
                onValueChange={(v) => setAge(v[0])}
              />
              <p className="mt-1 text-xs text-zinc-500">
                Trained on ages 40–75 (The Maastricht Study).
              </p>
            </div>

            {/* Sex */}
            <div>
              <label className="mb-2 block text-sm font-medium text-zinc-800 dark:text-zinc-200">
                Sex
              </label>
              <Select value={sex} onValueChange={(v: "M" | "F") => setSex(v)}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="M">Male</SelectItem>
                  <SelectItem value="F">Female</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Advanced predictors */}
            <details className="rounded-xl border border-zinc-200/70 p-4 dark:border-zinc-700/60">
              <summary className="cursor-pointer select-none text-sm font-semibold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300">
                Advanced predictors (optional)
              </summary>
              <div className="mt-4 space-y-4">
                <CheckRow
                  label="History of depression"
                  checked={historyDepression}
                  onChange={setHistoryDepression}
                />
                <CheckRow
                  label="Antidepressant use (current or recent)"
                  checked={antidepressantUse}
                  onChange={setAntidepressantUse}
                />
                <CheckRow
                  label="History of neuropathy"
                  checked={neuropathy}
                  onChange={setNeuropathy}
                />
                <p className="pt-1 text-xs text-zinc-500">
                  We compute an internal “any vulnerability” flag if either
                  history of depression or antidepressant use is selected.
                </p>
              </div>
            </details>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-2">
              <motion.button
                whileHover={{
                  scale: 1.02,
                  boxShadow: "0 6px 24px rgba(99,102,241,.35)",
                }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading || !formValid}
                className="inline-flex flex-1 items-center justify-center rounded-xl bg-indigo-600 px-5 py-2.5 font-semibold text-white shadow hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-400"
              >
                {loading ? "Scoring…" : "Predict risk"}
              </motion.button>

              <button
                type="button"
                onClick={resetForm}
                className="rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800/60"
              >
                Reset
              </button>
            </div>
          </form>

          {/* Result */}
          {pct !== null && (
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 rounded-2xl border border-indigo-100 bg-white/70 p-6 shadow-lg backdrop-blur dark:border-indigo-900/40 dark:bg-indigo-900/30"
            >
              <p className="text-sm font-medium text-zinc-700 dark:text-zinc-200">
                Estimated ~4-year risk
              </p>

              <div className="mt-3">
                <Progress value={pct} className="h-3" />
              </div>

              <div className="mt-2 flex items-end justify-between">
                <p className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
                  {pct.toFixed(1)}%
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Cohort baseline: ~14.7% over ~4 years.
                </p>
              </div>

              <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
                Research prototype for education/research — not a medical device.
                Results should not be used for diagnosis or treatment.
              </p>
            </motion.div>
          )}

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 rounded-lg border-l-4 border-red-600 bg-red-50 p-4 text-sm text-red-800 dark:border-red-500 dark:bg-red-900/40 dark:text-red-200"
            >
              {error}
            </motion.div>
          )}
        </motion.main>
      </TooltipProvider>
    </div>
  );
}

/* tiny checkbox row */
function CheckRow({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer select-none items-center justify-between rounded-lg border border-zinc-200/70 px-3 py-2 text-sm dark:border-zinc-700/60">
      <span className="text-zinc-800 dark:text-zinc-200">{label}</span>
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-zinc-300 text-indigo-600 focus:ring-indigo-500 dark:border-zinc-600"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
