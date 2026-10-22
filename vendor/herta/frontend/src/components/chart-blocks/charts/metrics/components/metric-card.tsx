import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { chartTitle } from "@/components/primitives";
import { cn } from "@/lib/utils";

export default function MetricCard({ title, value, change }: { title: string; value: string; change: number }) {
  return (
    <section className="top-beam glow-card flex flex-col rounded-lg border border-border bg-card px-4 py-3">
      <h2 className={cn(chartTitle({ color: "mute", size: "sm" }), "mb-1")}>{title}</h2>
      <div className="flex items-center gap-2">
        <span className="text-2xl font-semibold tabular-nums tracking-tight">{value}</span>
        {change !== 0 && (
          <span
            className={cn(
              "flex items-center rounded-sm px-1 py-0.5 text-xs",
              change > 0 ? "bg-emerald-50 text-emerald-600 dark:bg-emerald-950" : "bg-red-50 text-red-500 dark:bg-red-950",
            )}
          >
            {change > 0 ? "+" : ""}
            {Math.round(change)}%
            {change > 0 ? <ArrowUpRight className="ml-0.5 h-3 w-3" /> : <ArrowDownRight className="ml-0.5 h-3 w-3" />}
          </span>
        )}
      </div>
    </section>
  );
}
