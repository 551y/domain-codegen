"use client";

import { useEffect, useState } from "react";
import { Database } from "lucide-react";
import Container from "@/components/container";
import { Metrics, ConfigScores } from "@/components/chart-blocks";
import type { MetricData, ConfigScoreData } from "@/components/chart-blocks";
import PipelineFlow from "@/components/pipeline-flow";
import ChunkDistribution from "@/components/chunk-distribution";
import EvalWeights from "@/components/eval-weights";
import { api } from "@/lib/api";
import type { CompareMetrics, SettingsData } from "@/types/types";

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<MetricData[]>([
    { title: "知识库", value: "-", change: 0 },
    { title: "Config 相似度", value: "-", change: 0 },
    { title: "Handler 相似度", value: "-", change: 0 },
    { title: "整体相似度", value: "-", change: 0 },
  ]);
  const [configScores, setConfigScores] = useState<ConfigScoreData | null>(null);
  const [settings, setSettings] = useState<SettingsData | null>(null);
  const [kbSource, setKbSource] = useState<string | null>(null);
  const [kbChunks, setKbChunks] = useState<number | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [kbRes, settingsRes] = await Promise.all([
          api.kbList().catch(() => ({ knowledge_bases: [] })),
          api.getSettings().catch(() => null),
        ]);
        const kbs = kbRes.knowledge_bases || [];
        setSettings(settingsRes);
        if (kbs.length > 0) {
          setKbSource(kbs[0].source);
          setKbChunks(kbs[0].chunk_count ?? null);
        }

        setMetrics([
          { title: "知识库", value: `${kbs.length} 个`, change: 0 },
          { title: "Config 相似度", value: "-", change: 0 },
          { title: "Handler 相似度", value: "-", change: 0 },
          { title: "整体相似度", value: "-", change: 0 },
        ]);

        const targetsRes = await api.targets().catch(() => ({ targets: [] }));
        const targets = targetsRes.targets || [];
        if (targets.length > 0) {
          try {
            const cmpRes = await api.compareResult(targets[0].name);
            if (cmpRes.exists && cmpRes.metrics) {
              const m: CompareMetrics = cmpRes.metrics;
              setMetrics([
                { title: "知识库", value: `${kbs.length} 个`, change: 0 },
                { title: "Config 相似度", value: `${m.config_similarity.toFixed(1)}%`, change: 0 },
                { title: "Handler 相似度", value: `${m.handler_similarity.toFixed(1)}%`, change: 0 },
                { title: "整体相似度", value: `${m.overall_similarity.toFixed(1)}%`, change: 0 },
              ]);
              setConfigScores({
                topology: m.topology_similarity,
                cmd_list: m.cmd_list_similarity,
                tele_handler: m.tele_handler_similarity,
                internal_handler: m.internal_handler_similarity,
              });
            }
          } catch {
            // no compare result yet
          }
        }
      } catch (e) {
        console.error("Dashboard load error:", e);
      }
    }
    load();
  }, []);

  return (
    <div>
      <Metrics data={metrics} />

      {/* 生成流水线 */}
      <Container className="border-b border-border py-6">
        <header className="mb-4 flex items-baseline gap-3">
          <h2 className="text-[15px] font-semibold tracking-tight">生成流水线</h2>
          <span className="font-mono text-[10.5px] text-muted-foreground">
            协议文档 → 结构化中间表示（JSON）→ 目标代码 → 自动评测
          </span>
        </header>
        <PipelineFlow />
      </Container>

      {/* Config 得分 + 知识库构成 */}
      <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-5 laptop:divide-x laptop:divide-y-0">
        <Container className="py-6 laptop:col-span-3">
          <ConfigScores data={configScores} />
        </Container>
        <Container className="py-6 laptop:col-span-2">
          <header className="mb-3 flex items-baseline gap-2">
            <Database size={15} strokeWidth={1.8} className="translate-y-0.5 text-primary" />
            <h2 className="text-[15px] font-semibold tracking-tight">知识库构成</h2>
            {kbChunks !== null && (
              <span className="ml-auto font-mono text-[10.5px] text-muted-foreground">
                {kbSource} · {kbChunks} chunks
              </span>
            )}
          </header>
          <ChunkDistribution source={kbSource} />
        </Container>
      </div>

      {/* 评测体系 + 系统信息 */}
      <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-5 laptop:divide-x laptop:divide-y-0">
        <Container className="py-6 laptop:col-span-3">
          <header className="mb-4">
            <h2 className="text-[15px] font-semibold tracking-tight">评测体系</h2>
          </header>
          <EvalWeights />
        </Container>
        <Container className="py-6 laptop:col-span-2">
          <div className="flex h-full flex-col gap-3">
            <h2 className="text-[15px] font-semibold tracking-tight">系统信息</h2>
            <div className="overflow-hidden rounded-lg border border-border shadow-soft-sm">
              <table className="w-full text-sm">
                <tbody>
                  <tr className="border-b border-border">
                    <td className="bg-muted/50 px-4 py-2.5 font-medium">LLM 模型</td>
                    <td className="px-4 py-2.5 font-mono text-muted-foreground">
                      {settings?.chat_model || "-"}
                      {settings?.chat_provider && (
                        <span className="ml-1.5 rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground">
                          {settings.chat_provider === "openai" ? "OpenAI 兼容" : "智谱"}
                        </span>
                      )}
                    </td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="bg-muted/50 px-4 py-2.5 font-medium">Embedding 模型</td>
                    <td className="px-4 py-2.5 font-mono text-muted-foreground">
                      {settings?.embedding_model || "-"}
                      {settings?.embedding_provider && (
                        <span className="ml-1.5 rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground">
                          {settings.embedding_provider === "openai" ? "OpenAI 兼容" : "智谱"}
                        </span>
                      )}
                    </td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="bg-muted/50 px-4 py-2.5 font-medium">Embedding 维度</td>
                    <td className="px-4 py-2.5 font-mono text-muted-foreground">{settings?.embedding_dimensions || "-"}</td>
                  </tr>
                  <tr>
                    <td className="bg-muted/50 px-4 py-2.5 font-medium">API Key</td>
                    <td className="px-4 py-2.5 font-mono text-muted-foreground">
                      {settings?.api_key_set ? settings.api_key_preview : "未设置"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </Container>
      </div>
    </div>
  );
}
