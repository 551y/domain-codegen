"use client";

import { useEffect, useState, useCallback } from "react";
import { ChevronDown, FileText } from "lucide-react";
import Container from "@/components/container";
import { ConfigScores } from "@/components/chart-blocks";
import MarkdownView from "@/components/markdown-view";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { TargetInfo, CompareMetrics, FunctionMatch } from "@/types/types";

export default function ComparePage() {
  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [metrics, setMetrics] = useState<CompareMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [report, setReport] = useState<string | null>(null);
  const [showReport, setShowReport] = useState(false);

  useEffect(() => {
    api.targets().then((res) => setTargets(res.targets || [])).catch(() => {});
  }, []);

  // 选择 target 后自动加载已有评测结果
  useEffect(() => {
    if (!selectedTarget) return;
    setReport(null);
    api.compareResult(selectedTarget)
      .then((res) => {
        if (res.exists && res.metrics) setMetrics(res.metrics);
      })
      .catch(() => {});
    api.compareReport(selectedTarget)
      .then((res) => {
        if (res.exists && res.content) setReport(res.content);
      })
      .catch(() => {});
  }, [selectedTarget]);

  const runCompare = async () => {
    if (!selectedTarget) return;
    setLoading(true);
    setProgress("评测中...");
    setMetrics(null);
    try {
      const res = await api.runCompare(selectedTarget);
      const poll = async () => {
        try {
          const status = await api.taskStatus(res.task_id);
          if (status.status === "completed") {
            const result = await api.compareResult(selectedTarget);
            if (result.exists && result.metrics) {
              setMetrics(result.metrics);
            }
            // 重跑后同步刷新 Markdown 报告，避免展示上一轮内容
            api.compareReport(selectedTarget)
              .then((rep) => {
                if (rep.exists && rep.content) setReport(rep.content);
              })
              .catch(() => {});
            setProgress("");
            setLoading(false);
          } else if (status.status === "failed") {
            setProgress(`失败: ${status.error || "未知错误"}`);
            setLoading(false);
          } else {
            setProgress(status.message || "评测中...");
            setTimeout(poll, 2000);
          }
        } catch {
          setProgress("状态查询失败");
          setLoading(false);
        }
      };
      setTimeout(poll, 2000);
    } catch (e: unknown) {
      setProgress(`评测失败: ${e instanceof Error ? e.message : "未知错误"}`);
      setLoading(false);
    }
  };

  const fmt = (v: number) => `${v.toFixed(1)}%`;
  const matches = metrics?.details?.function_matches || [];
  const unmatchedGt = metrics?.details?.unmatched_gt_functions || [];
  const unmatchedGen = metrics?.details?.unmatched_gen_functions || [];

  return (
    <div>
      {/* Toolbar */}
      <Container className="flex items-center gap-4 border-b border-border py-3">
        <select
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={selectedTarget}
          onChange={(e) => setSelectedTarget(e.target.value)}
        >
          <option value="">-- 选择目标协议 --</option>
          {targets.map((t) => (
            <option key={t.name} value={t.name}>{t.name}</option>
          ))}
        </select>
        <Button onClick={runCompare} disabled={!selectedTarget || loading}>
          {loading ? "评测中..." : "运行评测"}
        </Button>
      </Container>

      {progress && !metrics && (
        <Container className="py-3">
          <div className="text-sm text-muted-foreground">{progress}</div>
        </Container>
      )}

      {metrics && (
        <>
          {/* Score Cards */}
          <Container className="grid grid-cols-2 gap-3 border-b border-border py-4 phone:grid-cols-4 laptop:grid-cols-7">
            {[
              { label: "拓扑", value: fmt(metrics.topology_similarity) },
              { label: "命令表", value: fmt(metrics.cmd_list_similarity) },
              { label: "遥测表", value: fmt(metrics.tele_handler_similarity) },
              { label: "内部处理", value: fmt(metrics.internal_handler_similarity) },
              { label: "Config", value: fmt(metrics.config_similarity), highlight: true },
              { label: "Handler", value: fmt(metrics.handler_similarity) },
              { label: "整体", value: fmt(metrics.overall_similarity), highlight: true },
            ].map((item) => (
              <div
                key={item.label}
                className={`top-beam glow-card rounded-lg border bg-card p-3 text-center ${item.highlight ? "border-primary/40 bg-primary/5" : "border-border"}`}
              >
                <div className="text-xs text-muted-foreground">{item.label}</div>
                <div className={`mt-1 text-lg font-semibold tabular-nums ${item.highlight ? "text-primary" : ""}`}>{item.value}</div>
              </div>
            ))}
          </Container>

          {/* Chart + Details */}
          <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-2 laptop:divide-x laptop:divide-y-0">
            <Container className="py-4">
              <ConfigScores data={{
                topology: metrics.topology_similarity,
                cmd_list: metrics.cmd_list_similarity,
                tele_handler: metrics.tele_handler_similarity,
                internal_handler: metrics.internal_handler_similarity,
              }} />
            </Container>
            <Container className="py-4">
              <h3 className="mb-3 text-sm font-medium">Handler 匹配统计</h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">真实函数</span>
                  <span className="font-mono">{metrics.details?.function_matches?.length ?? 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">成功匹配</span>
                  <span className="font-mono text-primary">{matches.filter((m) => m.is_matched).length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">未匹配 (GT)</span>
                  <span className="font-mono text-destructive">{unmatchedGt.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">多余生成</span>
                  <span className="font-mono text-muted-foreground">{unmatchedGen.length}</span>
                </div>
              </div>
            </Container>
          </div>

          {/* Function Match Table */}
          {matches.length > 0 && (
            <Container className="py-4">
              <h3 className="mb-3 text-sm font-medium">函数匹配列表</h3>
              <div className="overflow-x-auto rounded-lg border border-border bg-card shadow-soft-sm">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">真实函数</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">生成函数</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">同名</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">AST 综合</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">TED</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">语义</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matches.map((m: FunctionMatch, i: number) => (
                      <tr key={i} className="border-t border-border transition-colors hover:bg-muted/30">
                        <td className="px-3 py-2 font-mono text-xs">{m.gt_func_name}</td>
                        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{m.gen_func_name || "-"}</td>
                        <td className="px-3 py-2">{m.is_name_match ? (
                          <span className="text-emerald-600">✓</span>
                        ) : (
                          <span className="text-muted-foreground/50">✗</span>
                        )}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
                              <div
                                className={cn(
                                  "h-full rounded-full",
                                  m.ast_similarity >= 0.6 ? "bg-primary" : m.ast_similarity >= 0.4 ? "bg-accent" : "bg-muted-foreground/40",
                                )}
                                style={{ width: `${m.ast_similarity * 100}%` }}
                              />
                            </div>
                            <span className="font-mono text-xs tabular-nums">{(m.ast_similarity * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums">{(m.tree_edit_similarity * 100).toFixed(1)}%</td>
                        <td className="px-3 py-2 font-mono text-xs tabular-nums">{(m.semantic_feature_similarity * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Container>
          )}

          {/* 评测报告 */}
          {report && (
            <Container className="py-4">
              <div className="top-beam overflow-hidden rounded-lg border border-border bg-card shadow-soft-sm">
                <button
                  className="flex w-full items-center gap-2 px-5 py-3.5 text-left transition-colors hover:bg-muted/40"
                  onClick={() => setShowReport(!showReport)}
                >
                  <FileText size={15} strokeWidth={1.8} className="text-primary" />
                  <span className="text-sm font-medium">完整评测报告</span>
                  <span className="font-mono text-[10.5px] text-muted-foreground">Markdown</span>
                  <ChevronDown
                    size={15}
                    className={cn("ml-auto text-muted-foreground transition-transform", showReport && "rotate-180")}
                  />
                </button>
                {showReport && (
                  <div className="border-t border-border px-5 py-4">
                    <MarkdownView content={report} />
                  </div>
                )}
              </div>
            </Container>
          )}
        </>
      )}

      {!metrics && !loading && (
        <Container className="py-8 text-center text-sm text-muted-foreground">
          选择目标协议并运行评测
        </Container>
      )}
    </div>
  );
}
