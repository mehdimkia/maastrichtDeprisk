import SiteHeader from "@/components/site-header";

export const metadata = { title: "Deprisk – API quickstart" };

const sample = {
  sleep_minutes: 480,
  frag_index: 2, // Q1–Q4 → 1–4
  age: 60,
  sex: "M",
  any_vuln: 1,
  neuropathy: 0,
};

export default function ApiDocs() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50 dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900">
      <SiteHeader />
      <div className="mx-auto max-w-3xl px-6 py-12">
        <h1 className="text-3xl font-bold">API quickstart</h1>
        <p className="mt-3 text-zinc-600 dark:text-zinc-300">
          Score ~4-year depression risk with a single POST to{" "}
          <code>/api/score</code>.
        </p>

        <h2 className="mt-8 text-xl font-semibold">Request</h2>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100">
{`curl -X POST https://<your-host>/api/score \\
  -H 'Content-Type: application/json' \\
  -d '${JSON.stringify(sample, null, 2)}'`}
        </pre>

        <h2 className="mt-8 text-xl font-semibold">JS (fetch)</h2>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100">
{`const res = await fetch("/api/score", {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify(${JSON.stringify(sample, null, 2)})
});
const { prob } = await res.json(); // number in [0,1]`}
        </pre>

        <h2 className="mt-8 text-xl font-semibold">Response</h2>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100">
{`{ "prob": 0.304 }`}
        </pre>

        <p className="mt-8 text-xs text-zinc-500">
          Research prototype — not a medical device.
        </p>
      </div>
    </main>
  );
}
