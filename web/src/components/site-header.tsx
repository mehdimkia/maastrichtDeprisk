"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function cn(...parts: Array<string | boolean | undefined>) {
  return parts.filter(Boolean).join(" ");
}

const links = [
  { href: "/predict", label: "Predictor" },
  { href: "/powerbi", label: "Dashboard" },
  { href: "/model", label: "Model" },
  { href: "/api-docs", label: "API" },
];

export default function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="mx-auto flex max-w-6xl items-center justify-between p-6">
      <Link
        href="/"
        className="text-2xl font-extrabold tracking-tight text-indigo-600 dark:text-indigo-400"
      >
        Deprisk<span className="text-indigo-400">•</span>
      </Link>

      <nav className="hidden gap-6 md:flex">
        {links.map((l) => {
          const active =
            l.href === "/"
              ? pathname === "/"
              : pathname?.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "font-medium hover:text-indigo-600 dark:hover:text-indigo-400",
                active
                  ? "text-indigo-600 dark:text-indigo-400"
                  : "text-zinc-600 dark:text-zinc-300"
              )}
            >
              {l.label}
            </Link>
          );
        })}
        <a
          href="https://github.com/mehdimkia/maastrichtDeprisk"
          target="_blank"
          rel="noreferrer"
          className="font-medium text-zinc-600 hover:text-indigo-600 dark:text-zinc-300 dark:hover:text-indigo-400"
        >
          GitHub
        </a>
      </nav>
    </header>
  );
}
