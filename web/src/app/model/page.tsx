import SiteHeader from "@/components/site-header";

export const metadata = { title: "Deprisk – Model card" };

export default function ModelCard() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50 dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900">
      <SiteHeader />
      <div className="mx-auto max-w-3xl px-6 py-12">
        <h1 className="text-3xl font-bold">Model card</h1>
        <p className="mt-3 text-zinc-600 dark:text-zinc-300">
          Tuned gradient-boosted trees (XGBoost) estimating ~4-year risk of
          clinically relevant depressive symptoms (PHQ-9 ≥ 10).
        </p>

        <section className="mt-10 space-y-3">
          <h2 className="text-xl font-semibold">Data</h2>
          <ul className="list-disc pl-6 text-zinc-600 dark:text-zinc-300">
            <li>The Maastricht Study; baseline 2010–2020; ages 40–75.</li>
            <li>
              Analytic cohort: <strong>6,004</strong> participants;{" "}
              <strong>880</strong> incident cases.
            </li>
            <li>
              Outcome: first PHQ-9 ≥ 10 over ~4 years (annual assessments).
            </li>
          </ul>
        </section>

        <section className="mt-8 space-y-3">
          <h2 className="text-xl font-semibold">Features</h2>
          <ul className="list-disc pl-6 text-zinc-600 dark:text-zinc-300">
            <li>Sleep minutes (weekly average; spline transform)</li>
            <li>
              Sleep fragmentation (quartile Q1–Q4; ≥1-min posture-break count)
            </li>
            <li>Age, sex</li>
            <li>
              Advanced flags: prior depression / antidepressant use (
              <code>any_vuln</code>), neuropathy
            </li>
          </ul>
        </section>

        <section className="mt-8 space-y-3">
          <h2 className="text-xl font-semibold">Model & training</h2>
          <ul className="list-disc pl-6 text-zinc-600 dark:text-zinc-300">
            <li>
              XGBoost, Optuna-tuned; <code>tree_method="hist"</code>
            </li>
            <li>One-hot encoding; train n = 11,153; test n = 2,789</li>
          </ul>
        </section>

        <section className="mt-8 space-y-2">
          <h2 className="text-xl font-semibold">Performance (hold-out)</h2>
          <p className="text-zinc-600 dark:text-zinc-300">
            AUROC <strong>0.71</strong> · AUPRC <strong>0.29</strong>.
          </p>
        </section>

        <section className="mt-8 space-y-2">
          <h2 className="text-xl font-semibold">Limitations & ethics</h2>
          <p className="text-zinc-600 dark:text-zinc-300">
            Research prototype; not a medical device. Trained on ages 40–75; no
            external validation yet. Results should not be used for diagnosis or
            treatment.
          </p>
        </section>

        <section className="mt-10 flex gap-3">
          <a
            href="/predict"
            className="rounded-lg bg-indigo-600 px-4 py-2 font-semibold text-white hover:bg-indigo-700"
          >
            Try the predictor
          </a>
          <a
            href="https://github.com/mehdimkia/maastrichtDeprisk"
            target="_blank"
            className="rounded-lg border border-zinc-300 px-4 py-2 font-semibold text-zinc-700 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800/40"
          >
            View code on GitHub
          </a>
        </section>
      </div>
    </main>
  );
}
