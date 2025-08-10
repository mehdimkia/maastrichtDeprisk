/* app/predict/page.tsx — no Low/Moderate/High badges */
"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import {
  Select, SelectTrigger, SelectContent, SelectItem, SelectValue,
} from "@/components/ui/select";
import {
  Accordion, AccordionItem, AccordionTrigger, AccordionContent,
} from "@/components/ui/accordion";

type ApiResponse = { prob: number };

export default function PredictPage() {
  // form state
  const [sleepMin, setSleepMin] = useState<number>(480);
  const [fragmentation, setFragment] = useState<number>(1.9);
  const [age, setAge] = useState<number>(60);
  const [bmi, setBmi] = useState<number>(26);
  const [sex, setSex] = useState<"M" | "F" | "">("");
  const [smoking, setSmoking] = useState<"0" | "1" | "2" | "">("");
  const [alcohol, setAlcohol] = useState<"0" | "1" | "2" | "">("");
  const [diabetes, setDiabetes] = useState<"0" | "1" | "">("");

  // advanced (optional)
  const [anyVuln, setAnyVuln] = useState(false);
  const [activity, setActivity] = useState<"low" | "med" | "high" | "">("");
  const [neuropathy, setNeuropathy] = useState(false);

  // ui state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prob, setProb] = useState<number | null>(null);

  const formValid = () =>
    sex && smoking !== "" && alcohol !== "" && diabetes !== "";

  const activityToQuartile = (lvl: typeof activity) =>
    lvl === "low" ? 1 : lvl === "med" ? 2 : lvl === "high" ? 4 : 2;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!formValid()) return;

    setLoading(true);
    setError(null);
    setProb(null);

    const payload: Record<string, any> = {
      sleep_minutes: sleepMin,
      frag_index: fragmentation,
      age,
      bmi,
      sex,
      smoking_cat: Number(smoking),
      alcohol_cat: Number(alcohol),
      diabetes: Number(diabetes),
      any_vuln: Number(anyVuln),
      neuropathy: Number(neuropathy),
    };
    if (activity) payload.standstep_q4 = activityToQuartile(activity);

    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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

  return (
    <div className="relative isolate">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-br from-indigo-50 to-white dark:from-zinc-900 dark:to-zinc-800" />
      <motion.main
        initial={{ opacity: 0, y: 28 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="mx-auto max-w-2xl px-4 py-20"
      >
        <Link href="/" className="text-sm text-indigo-600 hover:underline">← Home</Link>

        <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-zinc-800 dark:text-zinc-100">
          Depression-risk predictor
        </h1>
        <p className="mt-2 text-lg text-zinc-600 dark:text-zinc-300">
          Enter your averages from the <strong>past 7 days</strong>; we’ll estimate your
          <strong> 4-year risk</strong> of developing clinically relevant depressive symptoms.
        </p>

        <form
          onSubmit={handleSubmit}
          className="mt-10 space-y-8 rounded-3xl border border-white/60 bg-white/70 p-8 shadow-2xl backdrop-blur dark:border-white/10 dark:bg-zinc-900/60"
        >
          <SliderRow label="Avg. nightly in-bed minutes" value={sleepMin} unit="min"
            min={240} max={960} step={10} onChange={setSleepMin} />
          <SliderRow
            label="Sleep-fragmentation (interruptions ≥ 1 min)"
            value={fragmentation} unit="" min={0} max={10} step={0.1}
            onChange={(v) => setFragment(parseFloat(v.toFixed(1)))}
          />
          <SliderRow label="Age" value={age} unit="y" min={18} max={90} step={1} onChange={setAge} />
          <SliderRow label="Body-mass index" value={bmi} unit="" min={15} max={45} step={0.1}
            onChange={(v) => setBmi(parseFloat(v.toFixed(1)))} />

          <SelectField label="Sex" value={sex} onChange={setSex}
            items={[{ value: "M", label: "Male" }, { value: "F", label: "Female" }]} />
          <SelectField label="Smoking status" value={smoking} onChange={setSmoking}
            items={[{ value: "0", label: "Never" }, { value: "1", label: "Former" }, { value: "2", label: "Current" }]} />
          <SelectField label="Alcohol consumption" value={alcohol} onChange={setAlcohol}
            items={[{ value: "0", label: "None" }, { value: "1", label: "Low" }, { value: "2", label: "High" }]} />
          <SelectField label="Diabetes" value={diabetes} onChange={setDiabetes}
            items={[{ value: "0", label: "No" }, { value: "1", label: "Yes" }]} />

          <Accordion type="single" collapsible>
            <AccordionItem value="adv">
              <AccordionTrigger className="text-sm font-semibold text-indigo-600 hover:text-indigo-700">
                Advanced predictors (optional)
              </AccordionTrigger>
              <AccordionContent className="mt-6 space-y-6">
                <SwitchRow label="Prior depression / antidepressant use" checked={anyVuln} onChange={setAnyVuln} />
                <SelectField
                  label="Physical-activity level (past week) (optional)"
                  value={activity} onChange={setActivity} placeholder="Select or skip"
                  items={[{ value: "low", label: "Low" }, { value: "med", label: "Medium" }, { value: "high", label: "High" }]}
                />
                <SwitchRow label="History of neuropathy?" checked={neuropathy} onChange={setNeuropathy} />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <motion.button
            whileHover={{ scale: 1.02, boxShadow: "0 4px 20px rgba(99,102,241,.4)" }}
            type="submit" disabled={loading || !formValid()}
            className="inline-flex w-full items-center justify-center rounded-xl
                       bg-gradient-to-r from-indigo-500 to-indigo-600 px-5 py-2.5
                       font-semibold text-white shadow hover:from-indigo-600 hover:to-indigo-700
                       disabled:opacity-40 dark:from-indigo-400 dark:to-indigo-500
                       dark:hover:from-indigo-500 dark:hover:to-indigo-600"
          >
            {loading ? "Scoring…" : "Predict risk"}
          </motion.button>
        </form>

        {prob !== null && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-10 rounded-2xl border border-indigo-100 bg-white/70 p-6 shadow-lg backdrop-blur dark:border-indigo-900/40 dark:bg-indigo-900/30"
          >
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-200">Estimated 4-year risk</p>
            <div className="mt-3">
              <Progress value={prob * 100} className="h-3" />
            </div>
            <p className="mt-2 text-lg font-semibold text-zinc-800 dark:text-zinc-100">
              {(prob * 100).toFixed(1)}%
            </p>
          </motion.div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-10 rounded-lg border-l-4 border-red-600 bg-red-50 p-4 text-sm text-red-800 dark:border-red-500 dark:bg-red-900/40 dark:text-red-300"
          >
            {error}
          </motion.div>
        )}
      </motion.main>
    </div>
  );
}

/* ——— tiny components ——— */

function SliderRow({
  label, value, unit, min, max, step, onChange,
}: {
  label: string; value: number; unit: string; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">
        {label}
        <span className="ml-1 text-xs text-zinc-400">
          ({value.toFixed(step < 1 ? 1 : 0)} {unit})
        </span>
      </label>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={(v) => onChange(v[0])} />
    </div>
  );
}

function SelectField({
  label, value, onChange, items, placeholder = "Select",
}: {
  label: string; value: string; onChange: (v: string) => void; items: { value: string; label: string }[]; placeholder?: string;
}) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-200">{label}</label>
      <Select value={value} onValueChange={onChange as any}>
        <SelectTrigger className="w-full"><SelectValue placeholder={placeholder} /></SelectTrigger>
        <SelectContent>
          {items.map((it) => (
            <SelectItem key={it.value} value={it.value}>{it.label}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function SwitchRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void; }) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-sm font-medium text-zinc-700 dark:text-zinc-200">{label}</label>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  );
}
