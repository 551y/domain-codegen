"use client";

import { useEffect, useState, useCallback } from "react";
import Container from "@/components/container";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { TargetInfo, KBInfo } from "@/types/types";

export default function GeneratePage() {
  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [kbs, setKbs] = useState<KBInfo[]>([]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedKb, setSelectedKb] = useState("");
  const [topK, setTopK] = useState(30);
  const [step, setStep] = useState(1);
  const [progress, setProgress] = useState("");
  const [progressPct, setProgressPct] = useState(0);
  const [configResult, setConfigResult] = useState<string | null>(null);
  const [handlerResult, setHandlerResult] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"config" | "handler">("config");

  useEffect(() => {
    Promise.all([api.targets(), api.kbList()])
      .then(([tRes, kRes]) => {
        setTargets(tRes.targets || []);
        setKbs(kRes.knowledge_bases || []);
      })
      .catch(() => {});
  }, []);

  // 选中目标后加载已有生成结果（仅展示用；点击生成按钮始终会重新生成）
  useEffect(() => {
    if (!selectedTarget) return;
    setConfigResult(null);
    setHandlerResult(null);
    api.generateResult(selectedTarget)
      .then((res) => {
        if (res.exists) {
          if (res.config) setConfigResult(JSON.stringify(res.config, null, 2));
          if (res.handler_source) setHandlerResult(res.handler_source);
        }
      })
      .catch(() => {});
  }, [selectedTarget]);

  const pollTask = useCallback(async (taskId: string, onDone: () => void) => {
    const poll = async () => {
      try {
        const status = await api.taskStatus(taskId);
        if (status.status === "completed") {
          setProgressPct(100);
          setTimeout(onDone, 500);
        } else if (status.status === "failed") {
          setProgress(`失败: ${status.error || "未知错误"}`);
          setStep(1);
        } else {
          setProgress(status.message || "处理中...");
          setProgressPct(Math.min(status.progress || 50, 90));
          setTimeout(poll, 2000);
        }
      } catch {
        setProgress("状态查询失败");
        setStep(1);
      }
    };
    setTimeout(poll, 2000);
  }, []);

  const handleGenConfig = async () => {
    if (!selectedTarget) return;
    setStep(2);
    setProgress("正在生成 Config...");
    setProgressPct(10);
    try {
      const res = await api.generateConfig(selectedTarget, selectedKb, topK);
      await pollTask(res.task_id, async () => {
        try {
          const result = await api.generateResult(selectedTarget);
          setConfigResult(JSON.stringify(result.config, null, 2));
          setStep(3);
          setProgress("");
        } catch {
          setProgress("获取结果失败");
          setStep(1);
        }
      });
    } catch (e: unknown) {
      setProgress(`生成失败: ${e instanceof Error ? e.message : "未知错误"}`);
      setStep(1);
    }
  };

  const handleGenHandler = async () => {
    if (!selectedTarget) return;
    setProgress("正在生成 Handler...");
    setProgressPct(10);
    try {
      const res = await api.generateHandler(selectedTarget, selectedKb, topK);
      await pollTask(res.task_id, async () => {
        try {
          const result = await api.generateResult(selectedTarget);
          setHandlerResult(result.handler_source || "");
          setProgress("");
        } catch {
          setProgress("获取结果失败");
        }
      });
    } catch (e: unknown) {
      setProgress(`生成失败: ${e instanceof Error ? e.message : "未知错误"}`);
    }
  };

  return (
    <div>
      {/* Target selection */}
      <Container className="border-b border-border py-4">
        <div className="grid gap-4 laptop:grid-cols-4">
          <div>
            <label className="mb-1 block text-sm font-medium">目标协议</label>
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={selectedTarget}
              onChange={(e) => setSelectedTarget(e.target.value)}
            >
              <option value="">-- 选择目标 --</option>
              {targets.map((t) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">源知识库</label>
            <select
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={selectedKb}
              onChange={(e) => setSelectedKb(e.target.value)}
            >
              <option value="">-- 选择知识库 --</option>
              {kbs.map((kb) => (
                <option key={kb.source} value={kb.source}>{kb.source}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Top-K</label>
            <input
              type="number"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <div className="flex items-end gap-2">
            <Button onClick={handleGenConfig} disabled={!selectedTarget || step === 2}>
              生成 Config
            </Button>
            <Button
              variant="outline"
              onClick={handleGenHandler}
              disabled={!selectedTarget || !configResult}
            >
              生成 Handler
            </Button>
          </div>
        </div>
      </Container>

      {/* Progress */}
      {progress && (
        <Container className="border-b border-border py-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-primary" />
            {progress}
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </Container>
      )}

      {/* Result */}
      {(configResult || handlerResult) && (
        <Container className="py-4">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex gap-1 rounded-lg border border-border bg-muted p-1">
              <button
                onClick={() => setViewMode("config")}
                className={`rounded-md px-4 py-1.5 text-sm transition-colors ${
                  viewMode === "config" ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Config（JSON）
              </button>
              <button
                onClick={() => setViewMode("handler")}
                className={`rounded-md px-4 py-1.5 text-sm transition-colors ${
                  viewMode === "handler" ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Handler（C）
              </button>
            </div>
            <button
              onClick={() => navigator.clipboard.writeText((viewMode === "config" ? configResult : handlerResult) || "")}
              className="ml-auto rounded-md border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
            >
              复制内容
            </button>
          </div>
          <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft-sm">
            <div className="flex items-center gap-2 border-b border-border bg-muted/50 px-4 py-2">
              <span className="h-2 w-2 rounded-full bg-primary/60" />
              <span className="font-mono text-xs text-muted-foreground">
                {viewMode === "config" ? `${selectedTarget}_config.json` : `${selectedTarget}_handler.c`}
              </span>
            </div>
            <pre className="code-scroll max-h-[600px] overflow-auto p-4 font-mono text-xs leading-relaxed">
              <code>{viewMode === "config" ? configResult : handlerResult}</code>
            </pre>
          </div>
        </Container>
      )}
    </div>
  );
}
