"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Database, LayoutGrid, Settings } from "lucide-react";
import Logo from "@/components/Logo";
import SettingsModal from "@/components/SettingsModal";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function TopBar() {
  const pathname = usePathname();
  const [llm, setLlm] = useState<boolean | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const load = () => api.settings().then((s) => setLlm(s.llm_available)).catch(() => setLlm(false));
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, []);

  const nav = [
    { name: "工作台", href: "/", icon: LayoutGrid },
    { name: "知识库", href: "/knowledge", icon: Database },
  ];

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-border bg-card/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-3 px-6">
          <Link href="/" className="flex items-center gap-2.5">
            <Logo size={30} className="shadow-soft-sm rounded-lg" />
            <div className="text-[14px] font-bold leading-tight tracking-tight">
              领域知识增强的无人系统代码智能适配平台
            </div>
          </Link>

          <nav className="ml-8 flex items-center gap-1">
            {nav.map((item) => {
              const Icon = item.icon;
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] transition-colors",
                    active
                      ? "bg-primary/10 font-medium text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Icon size={14} strokeWidth={1.9} />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <span
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px]",
                llm
                  ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-600"
                  : "border-border bg-muted text-muted-foreground",
              )}
            >
              <span className={cn("h-1.5 w-1.5 rounded-full", llm ? "pulse-dot bg-emerald-500" : "bg-muted-foreground/50")} />
              {llm === null ? "LLM …" : llm ? "LLM 已配置" : "LLM 未配置 · 离线演示"}
            </span>
            <button
              onClick={() => setShowSettings(true)}
              className="flex h-8 w-8 items-center justify-center rounded-lg border border-border text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              title="LLM 设置"
            >
              <Settings size={15} />
            </button>
          </div>
        </div>
      </header>
      {showSettings && <SettingsModal onClose={() => { setShowSettings(false); api.settings().then((s) => setLlm(s.llm_available)).catch(() => {}); }} />}
    </>
  );
}
