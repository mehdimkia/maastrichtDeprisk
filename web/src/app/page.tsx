/* web/src/app/page.tsx */
import Link from "next/link";

export const metadata = {
  title: "Maastricht Deprisk – Depression-risk Predictor",
  description:
    "Research prototype that estimates ~4-year depression risk (PHQ-9 ≥ 10) from sleep, activity and lifestyle data.",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50 dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900">
      {/* --- NAVBAR ----------------------------------------------------------- */}
      <header className="mx-auto flex max-w-6xl items-center justify-between p-6">
        <Link
          href="/"
          className="text-2xl font-extrabold tracking-tight text-indigo-600 dark:text-indigo-400"
        >
          Deprisk<span className="text-indigo-400">•</span>
        </Link>

        <nav className="hidden gap-6 md:flex">
          <Link
            href="/predict"
            className="font-medium text-zinc-600 hover:text-indigo-600 dark:text-zinc-300 dark:hover:text-indigo-400"
          >
            Predictor
          </Link>
          <Link
            href="/powerbi"
            className="font-medium text-zinc-600 hover:text-indigo-600 dark:text-zinc-300 dark:hover:text-indigo-400"
          >
            Dashboard
          </Link>
          <a
            href="https://github.com/mehdimkia/maastrichtDeprisk"
            target="_blank"
            className="font-medium text-zinc-600 hover:text-indigo-600 dark:text-zinc-300 dark:hover:text-indigo-400"
          >
            GitHub
          </a>
        </nav>
      </header>

      {/* --- HERO ------------------------------------------------------------- */}
      <section className="relative isolate overflow-hidden py-24 sm:py-32">
        <div
          className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,theme(colors.indigo.200)_0%,transparent_70%)] dark:bg-[radial-gradient(ellipse_at_top,theme(colors.indigo.800)_0%,transparent_70%)]"
          aria-hidden="true"
        />

        <div className="mx-auto max-w-4xl px-6 text-center lg:px-8">
          <h1 className="text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-white sm:text-6xl">
            Predict depression risk before it manifests
          </h1>
          <p className="mx-auto mt-6 max-w-prose text-lg leading-8 text-zinc-600 dark:text-zinc-300">
            Our model, a tuned gradient-boosted-trees (XGBoost) pipeline built on{" "}
            <span className="font-semibold text-indigo-600 dark:text-indigo-400">
              longitudinal data from The Maastricht Study (2010–2020 baseline)
            </span>
            , estimates the ~4-year risk of{" "}
            <span className="font-medium">clinically relevant depressive symptoms (PHQ-9 ≥ 10)</span>{" "}
            from wearable-derived sleep and activity metrics, plus clinical and sociodemographic
            covariates.
          </p>

          <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row sm:gap-6">
            <Link
              href="/predict"
              className="rounded-full bg-indigo-600 px-8 py-3 text-base font-semibold text-white shadow hover:bg-indigo-700 focus:outline-none focus-visible:ring focus-visible:ring-indigo-500/75"
            >
              Try the predictor
            </Link>
            <Link
              href="/powerbi"
              className="rounded-full border border-zinc-300 px-8 py-3 text-base font-semibold text-zinc-700 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800/40"
            >
              View cohort insights
            </Link>
          </div>
        </div>
      </section>

      {/* --- FEATURE STRIP ---------------------------------------------------- */}
      <section className="mx-auto grid max-w-6xl gap-8 px-6 py-20 sm:grid-cols-3 lg:px-8">
        {[
          {
            emoji: "⚡️",
            title: "Low-latency scoring",
            text: "FastAPI + XGBoost ‘hist’ for fast, production-style inference.",
          },
          {
            emoji: "🛡️",
            title: "Privacy-first",
            text: "Only derived fields leave your browser; raw Maastricht data remain on secure servers.",
          },
          {
            emoji: "📊",
            title: "Transparent metrics",
            text: "AUROC 0.71 · AUPRC 0.29 on a hold-out test set of 2,789 records (train = 11,153).",
          },
        ].map((f) => (
          <div key={f.title} className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-600 text-2xl text-white">
              {f.emoji}
            </div>
            <h3 className="mb-2 text-lg font-semibold text-zinc-900 dark:text-white">
              {f.title}
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">{f.text}</p>
          </div>
        ))}
      </section>

      {/* --- FACT BAR --------------------------------------------------------- */}
      <section className="mx-auto -mt-8 max-w-6xl px-6 pb-10 text-center text-xs text-zinc-500 dark:text-zinc-400">
        Analytic cohort: <span className="font-medium">6,004 participants</span> ·{" "}
        <span className="font-medium">880 incident cases</span>.
      </section>

      {/* --- FOOTER ----------------------------------------------------------- */}
      <footer className="border-t border-zinc-200 bg-white py-6 text-center dark:border-zinc-700 dark:bg-zinc-900">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          © {new Date().getFullYear()} Maastricht Deprisk · MIT License
        </p>
        <p className="text-xs mt-1 text-zinc-400">Research prototype — not a medical device.</p>
      </footer>
    </main>
  );
}
