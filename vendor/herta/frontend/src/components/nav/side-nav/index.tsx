"use client";

import { ArrowLeftToLine, ArrowRightToLine } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import Navigation from "./components/navigation";
import Brand from "./components/brand";

export default function SideNav() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        className={cn(
          "fixed left-0 top-12 z-50 rounded-r-md bg-secondary px-2 py-1.5 text-secondary-foreground shadow-soft-md hover:bg-secondary/70 tablet:hidden",
          "transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-60" : "translate-x-0",
        )}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? "收起导航" : "展开导航"}
      >
        {isOpen ? <ArrowLeftToLine size={16} /> : <ArrowRightToLine size={16} />}
      </button>
      <aside
        className={cn(
          "fixed bottom-0 left-0 top-0 z-40 flex h-[100dvh] w-60 shrink-0 flex-col border-r border-border bg-card/80 backdrop-blur tablet:sticky tablet:translate-x-0",
          "transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <Brand />
        <Navigation />
        <footer className="space-y-2.5 border-t border-border px-4 py-3.5">
          <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
            <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span>SYSTEM ONLINE</span>
          </div>
          <div className="flex items-center justify-between font-mono text-[10px] text-muted-foreground/60">
            <span>v0.1.0</span>
            <span>RAG PIPELINE</span>
          </div>
        </footer>
      </aside>
    </>
  );
}
