/*
  Predictor form (app/predict/page.tsx)
  ──────────────────────────────────────────────────────────────────────────
  • Fixes unterminated string & completes truncated JSX (Accordion ➜ Neuropathy, Submit, result/error blocks).
  • File now passes `tsx` parser.
*/

"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";

import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ApiResponse = { prob: number };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PredictPage() {
  // ── Controlled inputs ────────────────────────────────────────────────────
  const [sleepMin, setSleepMin] = useState<number>(490); // cohort mean
  const [fragmentation, setFragmentation] = useState<number>(1.9);
  const [age, setAge] = useState<number>(60);
  const [bmi, setBmi] = useState<number>(26);
  const [sex, setSex] = useState<"M" | "F" | "">("");
  const [smoking, setSmoking] = useState<"0" | "1" | "2" | "">("");
  const [alcohol, setAlcohol] = useState<"0" | "1" | "2" | "">("");
  const [diabetes, setDiabetes] = useState<"0" | "1" | "">("");

  // ── Option-B inputs ──────────────────────────────────────────────────────
  const [anyVuln, setAnyVuln] = useState<boolean>(false);
  const [activity, setActivity] = useState<"low" | "med" | "high" | "">("");
  const [neuropathy, setNeuropathy] = useState<boolean>(false);

  // ── UI state ─────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prob, setProb] = useState<number | null>(null);

  // ── Helpers ──────────────────────────────────────────────────────────────
  function formValid() {
    return (
      sex &&
      smoking !== "" &&
      alcohol !== "" &&
      diabetes !== "" &&
      activity !== ""
    );
  }

  const activityToQuartile = (lvl: typeof activity) => {
    switch (lvl) {
      case "low":
        return 1;
      case "med":
        return 2;
      case "high":
        return 4;
      default:
        return 2;
    }
  };

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!formValid()) return;

    setLoading(true);
    setError(null);
    setProb(null);

    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sleep_minutes: sleepMin,
          frag_index: fragmentation,
          age,
          bmi,
          sex,
          smoking_cat: Number(smoking),
          alcohol_cat: Number(alcohol),
          diabetes: Number(diabetes),
          any_vuln: Number(anyVuln),
          standstep_q4: activityToQuartile(activity),
          neuropathy: Number(neuropathy),
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

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------
  return (
    <motion.main
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="mx-auto max-w-xl px-4 py-16"
    >
      {/* Header */}
      <Link href="/" className="text-sm text-indigo-600 hover:underline">
        ← Home
      </Link>
      <h1 className="mt-4 text-3xl font-bold tracking-tight text-zinc-800 dark:text-zinc-100">
        Depression-risk predictor
      </h1>
      <p className="mt-2 text-zinc-600 dark:text-zinc-300">
        Enter your averages from the <strong>past&nbsp;7&nbsp;days</strong>; we’ll estimate your
        <strong>&nbsp;4-year&nbsp;risk</strong> of developing clinically relevant depressive
        symptoms.
      </p>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="mt-8 space-y-8 rounded-2xl bg-white/80 p-6 shadow-xl dark:bg-zinc-900/50 dark:shadow-zinc-800 backdrop-blur-md"
      >
        {/* Sleep minutes */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Avg. nightly in-bed minutes (past 7 nights)
            <span className="ml-1 text-xs text-zinc-400">({sleepMin} min)</span>
          </label>
          <Slider
            min={240}
            max={960}
            step={10}
            value={[sleepMin]}
            onValueChange={(v) => setSleepMin(v[0])}
          />
        </div>

        {/* Fragmentation index */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Sleep-fragmentation index (avg. breaks ≥ 1 min)
            <span className="ml-1 text-xs text-zinc-400">
              ({fragmentation.toFixed(1)})
            </span>
          </label>
          <Slider
            min={0}
            max={10}
            step={0.1}
            value={[fragmentation]}
            onValueChange={(v) => setFragmentation(parseFloat(v[0].toFixed(1)))}
          />
        </div>

        {/* Age */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Age <span className="ml-1 text-xs text-zinc-400">({age} y)</span>
          </label>
          <Slider
            min={18}
            max={90}
            step={1}
            value={[age]}
            onValueChange={(v) => setAge(v[0])}
          />
        </div>

        {/* BMI */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Body-mass index
            <span className="ml-1 text-xs text-zinc-400">({bmi.toFixed(1)})</span>
          </label>
          <Slider
            min={15}
            max={45}
            step={0.1}
            value={[bmi]}
            onValueChange={(v) => setBmi(parseFloat(v[0].toFixed(1)))}
          />
        </div>

        {/* Sex */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Sex
          </label>
          <Select value={sex} onValueChange={setSex as any}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="M">Male</SelectItem>
              <SelectItem value="F">Female</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Smoking status */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Smoking status
          </label>
          <Select value={smoking} onValueChange={setSmoking as any}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">Never</SelectItem>
              <SelectItem value="1">Former</SelectItem>
              <SelectItem value="2">Current</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Alcohol consumption */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Alcohol consumption
          </label>
          <Select value={alcohol} onValueChange={setAlcohol as any}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">None</SelectItem>
              <SelectItem value="1">Low</SelectItem>
              <SelectItem value="2">High</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Diabetes */}
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
            Diabetes
          </label>
          <Select value={diabetes} onValueChange={setDiabetes as any}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">No</SelectItem>
              <SelectItem value="1">Yes</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Advanced predictors accordion */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="advanced">
            <AccordionTrigger className="text-sm font-semibold text-indigo-600 hover:text-indigo-700">
              Advanced predictors (optional)
            </AccordionTrigger>
            <AccordionContent className="mt-6 space-y-6">
              {/* Prior depression / antidepressant */}
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-zinc-700 dark:text-zinc-200">
                  Prior depression / antidepressant use
                </label>
                <Switch checked={anyVuln} onCheckedChange={setAnyVuln} />
              </div>

              {/* Physical-activity level */}
              <div>
                <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
                  Physical-activity level (past week)
                </label>
                <Select value={activity} onValueChange={setActivity as any}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="med">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Neuropathy history */}
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-zinc-700 dark:text-zinc-200">
                  History of neuropathy?
                </label>
                <Switch checked={neuropathy} onCheckedChange={setNeuropathy} />
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Submit */}
        <button
          type="submit"
          disabled={loading || !formValid()}
          className="inline-flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 px-4 py-2 font-medium text-white hover:from-indigo-600 hover:to-indigo-700 disabled:opacity-40 dark:from-indigo-400 dark:to-indigo-500 dark:hover:from-indigo-500 dark:hover:to-indigo-600"
        >
          {loading ? "Scoring…" : "Predict risk"}
        </button>
      </form>

      {/* Result block */}
      {prob !== null && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 rounded-lg border-l-4 border-indigo-600 bg-indigo-50 p-4 text-indigo-800 dark:border-indigo-500 dark:bg-indigo-950/40 dark:text-indigo-300"
        >
          <p className="text-lg font-semibold">
            Estimated 4-year risk: {Math.round(prob * 1000) / 10}%
          </p>
          <p className="text-sm">
            <span className="font-medium">Interpretation:</span>{" "}
            {prob < 0.1
              ? "Low"
              : prob < 0.25
              ? "Moderate"
              : prob < 0.4
              ? "Elevated"
              : "High"} risk versus cohort baseline.
          </p>
        </motion.div>
      )}

      {/* Error block */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 rounded-lg border-l-4 border-red-600 bg-red-50 p-4 text-sm text-red-800 dark:border-red-500 dark:bg-red-900/40 dark:text-red-300"
        >
          {error}
        </motion.div>
      )}
    </motion.main>
  );
}
