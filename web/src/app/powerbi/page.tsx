/* web/src/app/powerbi/page.tsx — Under-construction placeholder */

import Link from "next/link";
import SiteHeader from "@/components/site-header";

export const metadata = { title: "Deprisk – Cohort dashboard (Power BI)" };

export default function PowerBIPlaceholder() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50 dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900">
      <SiteHeader />

      <section className="mx-auto max-w-3xl px-6 py-16 text-center">
        <div className="mx-auto w-full rounded-2xl border border-dashed border-zinc-300 bg-white/70 p-10 shadow-md backdrop-blur dark:border-zinc-700 dark:bg-zinc-900/60">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-indigo-600 text-3xl text-white">
            🏗️
          </div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            Cohort dashboard (Power BI)
          </h1>
          <p className="mx-auto mt-3 max-w-prose text-zinc-600 dark:text-zinc-300">
            This page is <strong>under construction</strong>. We’re preparing an interactive
            Power&nbsp;BI dashboard with cohort-level sleep and mental-health insights.
          </p>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            Check back soon—or explore the predictor in the meantime.
          </p>

          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href="/predict"
              className="rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white hover:bg-indigo-700"
            >
              Try the predictor
            </Link>
            <Link
              href="/model"
              className="rounded-lg border border-zinc-300 px-4 py-2.5 font-semibold text-zinc-700 hover:bg-zinc-50 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800/40"
            >
              Read the model card
            </Link>
          </div>

          {/* TODO: Replace this block with the Power BI embed when ready.
              Example:
              <iframe
                title="Deprisk Cohort Dashboard"
                width="100%"
                height="720"
                src="https://app.powerbi.com/view?r=<report-id>"
                allowFullScreen
                className="mt-8 rounded-xl border border-zinc-200 dark:border-zinc-700"
              />
          */}
        </div>
      </section>
    </main>
  );
}
