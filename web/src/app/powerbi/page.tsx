"use client";
/**
 * File: web/src/app/powerbi/page.tsx (Next.js App Router)
 *
 * Purpose
 * - Client-side cohort dashboard using a static CSV in /public/data/
 * - Slicers (sex, age group, diabetes), KPIs, heatmap, Top-3 by lift, narrative
 * - Robust CSV loading (null/undefined guards)
 *
 * Deps
 *   pnpm add echarts echarts-for-react papaparse
 */

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
// @ts-ignore — Vercel typecheck may lack papaparse types in this project; runtime import is fine
import Papa from "papaparse";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

// -------------------------
// Types & helpers
// -------------------------

type Row = {
  id: string;
  incident_depression: number | string | null; // 0/1 (numeric or numeric-like string)
  sleep_duration_cat: string; // "<7h" | "7–9h" | "≥9h"
  frag_quartile: string; // "Q1".."Q4"
  age: number | string;
  sex: "M" | "F" | string;
  diabetes: string; // e.g., "None" | "Prediabetes" | "Type 2"
};

type AgeBand = "40–49" | "50–59" | "60–69" | "70–75";

type WithBand = Omit<Row, "age"> & { age: number; age_band: AgeBand; incident_depression: number };

type Filters = {
  sex: "All" | "M" | "F";
  ageBand: "All" | AgeBand;
  diabetes: "All" | string;
};

type HeatmapMetric = "Incident Rate" | "Case Share";

const SLEEP_CATS = ["<7h", "7–9h", "≥9h"] as const;
const FRAG_QS = ["Q1", "Q2", "Q3", "Q4"] as const;
const AGE_BANDS: AgeBand[] = ["40–49", "50–59", "60–69", "70–75"];

const band = (a: number): AgeBand => (a < 50 ? "40–49" : a < 60 ? "50–59" : a < 70 ? "60–69" : "70–75");
const toNum = (v: unknown, fallback = 0): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
};
const to01 = (v: unknown): number => (toNum(v, 0) ? 1 : 0);
const pct = (x: number) => new Intl.NumberFormat("en-GB", { style: "percent", maximumFractionDigits: 1 }).format(x || 0);
const num = (x: number) => new Intl.NumberFormat("en-GB").format(Math.round(x || 0));

// -------------------------
// Page
// -------------------------

export default function CohortKPIPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [filters, setFilters] = useState<Filters>({ sex: "All", ageBand: "All", diabetes: "All" });
  const [metric, setMetric] = useState<HeatmapMetric>("Incident Rate");

  // Robust CSV loader: try multiple candidate paths (env override → /data/... → /deprisk_synth_6004.csv)
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      const CSV_PATH = process.env.NEXT_PUBLIC_CSV_PATH ?? "/data/deprisk_synth_6004.csv";
      const candidates = [CSV_PATH, "/deprisk_synth_6004.csv"];
      const attempts: string[] = [];

      try {
        let text: string | null = null;
        for (const url of candidates) {
          attempts.push(url);
          try {
            const res = await fetch(url, { cache: "no-store" });
            if (res.ok) {
              text = await res.text();
              break;
            }
          } catch {
            // continue to next candidate
          }
        }
        if (!text) throw new Error(`HTTP error — tried ${attempts.join(" → ")}`);

        const parsed = Papa.parse<Row>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
        const data = Array.isArray(parsed.data) ? parsed.data.filter(Boolean) : [];

        // Validate minimal columns exist
        const required = [
          "incident_depression",
          "sleep_duration_cat",
          "frag_quartile",
          "age",
          "sex",
          "diabetes",
        ];
        const hasCols = required.every((c) => Object.prototype.hasOwnProperty.call(data[0] || {}, c));
        if (!hasCols) throw new Error("CSV is missing required columns");

        if (!cancelled) setRows(data);
      } catch (e: any) {
        if (!cancelled)
          setError(
            `${e?.message || "Failed to load CSV"}\nEnsure the CSV is available at web/public/data/deprisk_synth_6004.csv → served at /data/deprisk_synth_6004.csv.`
          );
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // Derive constant cohort aggregates & age bands (normalize types first)
  const all = useMemo(() => {
    const withBands: WithBand[] = rows.map((r) => {
      const ageNum = toNum(r.age);
      return {
        ...r,
        age: ageNum,
        age_band: band(ageNum),
        incident_depression: to01(r.incident_depression),
        sex: (r.sex as string) as WithBand["sex"],
        diabetes: String(r.diabetes),
      } as WithBand;
    });
    const totalPop = withBands.length;
    const totalCases = withBands.reduce((s, r) => s + to01(r.incident_depression), 0);
    const overallIR = totalCases / Math.max(1, totalPop);
    return { withBands, totalPop, totalCases, overallIR };
  }, [rows]);

  // Build diabetes options dynamically
  const diabetesOptions = useMemo(
    () => ["All", ...Array.from(new Set(all.withBands.map((r) => String(r.diabetes))))],
    [all.withBands]
  );

  // Apply slicers
  const filtered = useMemo(() => {
    let r = all.withBands;
    if (filters.sex !== "All") r = r.filter((x) => x.sex === filters.sex);
    if (filters.ageBand !== "All") r = r.filter((x) => x.age_band === filters.ageBand);
    if (filters.diabetes !== "All") r = r.filter((x) => String(x.diabetes) === filters.diabetes);
    return r;
  }, [all.withBands, filters]);

  // KPIs
  const population = filtered.length;
  const cases = filtered.reduce((s, r) => s + to01(r.incident_depression), 0);
  const ir = cases / Math.max(1, population);
  const popShare = population / Math.max(1, all.totalPop);
  const caseShare = cases / Math.max(1, all.totalCases);
  const lift = ir / Math.max(1e-9, all.overallIR);
  const longSleepRate =
    filtered.filter((r) => r.sleep_duration_cat === "≥9h").length / Math.max(1, population);

  // Matrix cells (Incident Rate or Case Share per cell), with extra dims for tooltip:
  // value shape: [xIdx, yIdx, metricVal, cellPop, cellCases, cellInc, cellPopShare, cellCaseShare]
  const matrix = useMemo(() => {
    return FRAG_QS.flatMap((fq, i) =>
      SLEEP_CATS.map((sc, j) => {
        const cell = filtered.filter((r) => r.frag_quartile === fq && r.sleep_duration_cat === sc);
        const cPop = cell.length;
        const cCases = cell.reduce((s, r) => s + to01(r.incident_depression), 0);
        const cInc = cPop ? cCases / cPop : 0;
        const metricVal =
          metric === "Incident Rate" ? cInc : cCases / Math.max(1, all.totalCases);
        const safe = Number.isFinite(metricVal) && !Number.isNaN(metricVal) ? metricVal : 0;
        const cPopShare = cPop / Math.max(1, all.totalPop);
        const cCaseShare = cCases / Math.max(1, all.totalCases);
        return [j, i, safe, cPop, cCases, cInc, cPopShare, cCaseShare] as [
          number,
          number,
          number,
          number,
          number,
          number,
          number,
          number
        ];
      })
    );
  }, [filtered, metric, all.totalCases, all.totalPop]);

  // Top-3 segments by Lift (based on IR)
  const top3 = useMemo(() => {
    const cells = FRAG_QS.flatMap((fq) =>
      SLEEP_CATS.map((sc) => {
        const cell = filtered.filter((r) => r.frag_quartile === fq && r.sleep_duration_cat === sc);
        const p = cell.length;
        const c = cell.reduce((s, r) => s + to01(r.incident_depression), 0);
        const irc = p ? c / p : 0;
        return {
          sleep: sc,
          frag: fq,
          ir: irc,
          lift: all.overallIR ? irc / all.overallIR : 0,
          popShare: p / Math.max(1, all.totalPop),
          caseShare: c / Math.max(1, all.totalCases),
        };
      })
    )
      .sort((a, b) => b.lift - a.lift)
      .slice(0, 3);
    return cells;
  }, [filtered, all.overallIR, all.totalPop, all.totalCases]);

  const narrative = top3[0]
    ? `Highest relative risk: ${top3[0].sleep} × ${top3[0].frag} — IR ${pct(top3[0].ir)} (lift ${top3[0].lift.toFixed(
        2
      )}×; ${pct(top3[0].popShare)} of cohort; ${pct(top3[0].caseShare)} of cases).`
    : "";

 // ECharts config
const MAX_IR = 0.35;
const heatmapOption = useMemo(
  () => ({
    tooltip: {
      trigger: "item",
      formatter: (p: any) => {
        const sc = SLEEP_CATS[p.value[0]];
        const fq = FRAG_QS[p.value[1]];
        const cPop = p.value[3] as number;
        const cCases = p.value[4] as number;
        const inc = p.value[5] as number;
        const cPopShare = p.value[6] as number;
        const cCaseShare = p.value[7] as number;
        return [
          `${fq} × ${sc}`,
          `Incidence: ${pct(inc)}`,
          `Cases: ${num(cCases)} / ${num(cPop)}`,
          `Cohort share: ${pct(cPopShare)}`,
          `Share of cases: ${pct(cCaseShare)}`,
        ].join("<br/>");
      },
    },
    grid: { left: 70, right: 90, bottom: 60, top: 40 },
    xAxis: {
      type: "category",
      data: SLEEP_CATS,
      axisLabel: { interval: 0 },
      name: "Sleep duration",
      nameLocation: "middle",
      nameGap: 35,
      nameTextStyle: { fontSize: 12, color: "#4b5563" },
    },
    yAxis: {
      type: "category",
      data: FRAG_QS,
      name: "Sleep fragmentation (Q1 low → Q4 high)",
      nameLocation: "middle",
      nameGap: 55,
      nameTextStyle: { fontSize: 12, color: "#4b5563" },
    },
    visualMap: {
      min: 0,
      max: metric === "Incident Rate" ? MAX_IR : 0.35,
      // ⬇️ map color to the metric at index 2 (not x/y)
      dimension: 2,
      orient: "vertical",
      right: 10,
      top: 20,
      // ⬇️ ensure higher = darker
      inRange: {
        color: ["#e8ecff", "#b2c4ff", "#6b8cff", "#3b82f6", "#1e40af"],
      },
      inverse: false,
    },
    series: [
      {
        type: "heatmap",
        data: matrix,
        label: {
          show: true,
          formatter: (p: any) => pct(p.value[2]),
          color: "#111827",
          fontWeight: 600,
          textBorderColor: "rgba(255,255,255,0.95)",
          textBorderWidth: 2,
        },
        emphasis: {
          itemStyle: { shadowBlur: 10 },
          label: { show: true },
        },
      },
    ],
  }),
  [matrix, metric]
);


  // -------------------------
  // Dev sanity tests (non-breaking)
  // -------------------------
  useEffect(() => {
    if (loading || error || !all.totalPop) return;

    // Existing tests (kept):
    console.assert(all.totalPop === 6004, `Expected 6004 rows, got ${all.totalPop}`);
    console.assert(all.overallIR > 0.13 && all.overallIR < 0.16, `Overall IR out of range: ${all.overallIR}`);
    if (top3[0]) {
      const seg = `${top3[0].sleep} × ${top3[0].frag}`;
      console.assert(seg === "≥9h × Q4", `Top segment expected "≥9h × Q4", got "${seg}"`);
    }

    // Additional tests:
    // 1) Required columns exist
    const cols = Object.keys(all.withBands[0] || {});
    ["incident_depression", "sleep_duration_cat", "frag_quartile", "age", "sex", "diabetes"].forEach((c) =>
      console.assert(cols.includes(c), `Missing column: ${c}`)
    );
    // 2) Matrix size is FRAG_QS x SLEEP_CATS
    console.assert(
      matrix.length === FRAG_QS.length * SLEEP_CATS.length,
      `Matrix size mismatch: ${matrix.length}`
    );
    // 3) IR with no filters equals overall IR (within tolerance)
    const tol = 1e-6;
    console.assert(
      Math.abs(ir - all.overallIR) < tol ||
        filters.sex !== "All" ||
        filters.ageBand !== "All" ||
        filters.diabetes !== "All",
      "IR should equal overall IR when no filters are applied"
    );
    // 4) Diabetes options include at least 3 categories
    console.assert(diabetesOptions.length >= 4, `Unexpected diabetes options: ${diabetesOptions.join(", ")}`);
  }, [loading, error, all.totalPop, all.overallIR, top3, matrix.length, ir, filters, diabetesOptions]);

  // ---------------
  // UI
  // ---------------

  if (loading) {
    return (
      <main className="mx-auto max-w-6xl p-6 space-y-6">
        <h1 className="text-2xl font-semibold">Cohort Dashboard</h1>
        <p className="text-neutral-600">Loading data…</p>
        <div className="h-64 animate-pulse rounded-2xl bg-neutral-100" />
      </main>
    );
  }

  if (error) {
    return (
      <main className="mx-auto max-w-6xl p-6 space-y-6">
        <h1 className="text-2xl font-semibold">Cohort Dashboard</h1>
        <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-red-700 whitespace-pre-line">{error}</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold">Cohort Dashboard</h1>
        <p className="text-sm text-neutral-700">
          Explore new depression cases by sleep duration and sleep fragmentation. Filter by sex, age, and diabetes to see how rates change.
        </p>
        <details className="rounded-lg border bg-neutral-50 p-3 text-sm text-neutral-700">
          <summary className="cursor-pointer select-none font-medium">How to read this chart</summary>
          <ul className="ml-5 list-disc space-y-1 pt-2">
            <li><b>Incidence rate</b>: share of people who developed depression during follow-up.</li>
            <li><b>Relative risk (lift)</b>: a cell’s incidence divided by the overall rate.</li>
            <li><b>Fragmentation quartile</b>: Q1 is least fragmented; Q4 is most fragmented sleep.</li>
          </ul>
        </details>
      </header>

      {/* Controls */}
      <section className="flex flex-wrap items-end gap-3 rounded-2xl bg-neutral-50 p-4">
        <div className="flex flex-col">
          <label className="text-xs text-neutral-600" htmlFor="sex">Sex</label>
          <select id="sex" className="rounded-lg border bg-white px-3 py-2" value={filters.sex} onChange={(e) => setFilters((v) => ({ ...v, sex: e.target.value as Filters["sex"] }))}>
            {["All", "M", "F"].map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col">
          <label className="text-xs text-neutral-600" htmlFor="ageBand">Age group</label>
          <select id="ageBand" className="rounded-lg border bg-white px-3 py-2" value={filters.ageBand} onChange={(e) => setFilters((v) => ({ ...v, ageBand: e.target.value as Filters["ageBand"] }))}>
            {["All", ...AGE_BANDS].map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col">
          <label className="text-xs text-neutral-600" htmlFor="diabetes">Diabetes</label>
          <select id="diabetes" className="rounded-lg border bg-white px-3 py-2" value={filters.diabetes} onChange={(e) => setFilters((v) => ({ ...v, diabetes: e.target.value as Filters["diabetes"] }))}>
            {diabetesOptions.map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col">
          <label className="text-xs text-neutral-600" htmlFor="metric">Heatmap shows</label>
          <select id="metric" className="rounded-lg border bg-white px-3 py-2" value={metric} onChange={(e) => setMetric(e.target.value as HeatmapMetric)}>
            {(["Incident Rate", "Case Share"] as HeatmapMetric[]).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </div>

        <button className="ml-auto rounded-lg border px-3 py-2 text-sm hover:bg-neutral-100" onClick={() => setFilters({ sex: "All", ageBand: "All", diabetes: "All" })}>
          Reset filters
        </button>
      </section>

      {/* KPIs */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KPI title="Incidence rate (overall)" value={pct(all.overallIR)} />
        <KPI title="Incidence rate (filtered)" value={pct(ir)} />
        <KPI title="People in view" value={num(population)} />
        <KPI title="Long sleepers (≥9h)" value={pct(longSleepRate)} />
      </section>

      {/* Heatmap */}
      <section className="rounded-2xl bg-white p-3 shadow space-y-3">
        {all.totalPop > 0 ? (
          <ReactECharts option={heatmapOption as any} style={{ height: 520 }} notMerge lazyUpdate />
        ) : (
          <div className="h-64" />
        )}
        <p className="px-2 text-xs text-neutral-600">
          Rows: sleep fragmentation quartile (<b>Q1</b> = least fragmented, <b>Q4</b> = most). Columns: sleep duration bands. Darker cells mean higher values.
        </p>
      </section>

      {/* Top segments */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Top segments (highest relative risk)</h2>
          <ExportTop3Button top3={top3} />
        </div>
        <div className="overflow-hidden rounded-2xl border">
          <table className="w-full text-sm">
            <thead className="bg-neutral-50 text-left">
              <tr>
                <Th>Sleep duration</Th>
                <Th>Fragmentation quartile</Th>
                <Th>Incidence rate</Th>
                <Th>Relative risk (lift)</Th>
                <Th>Cohort share</Th>
                <Th>Share of new cases</Th>
              </tr>
            </thead>
            <tbody>
              {top3.map((t, i) => (
                <tr key={i} className="border-t">
                  <Td>{t.sleep}</Td>
                  <Td>{t.frag}</Td>
                  <Td>{pct(t.ir)}</Td>
                  <Td>{t.lift.toFixed(2)}×</Td>
                  <Td>{pct(t.popShare)}</Td>
                  <Td>{pct(t.caseShare)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {narrative && <p className="text-sm text-neutral-700">{narrative}</p>}
      </section>
    </main>
  );
}

// -------------------------
// Small UI primitives
// -------------------------

function KPI({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="text-xs text-neutral-600">{title}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-3 py-2 text-xs font-medium text-neutral-600">{children}</th>;
}

function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-3 py-2 align-middle">{children}</td>;
}

function ExportTop3Button({
  top3,
}: {
  top3: { sleep: string; frag: string; ir: number; lift: number; popShare: number; caseShare: number }[];
}) {
  const exportCsv = () => {
    const header = ["sleep", "frag", "ir", "lift", "popShare", "caseShare"].join(",");
    const rows = top3
      .map(
        (t) =>
          `${t.sleep},${t.frag},${t.ir.toFixed(4)},${t.lift.toFixed(4)},${t.popShare.toFixed(4)},${t.caseShare.toFixed(
            4
          )}`
      )
      .join("\n");
    const blob = new Blob([header + "\n" + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "top3_segments.csv";
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <button onClick={exportCsv} className="rounded-lg border px-3 py-1.5 text-sm hover:bg-neutral-100">
      Download CSV
    </button>
  );
}
