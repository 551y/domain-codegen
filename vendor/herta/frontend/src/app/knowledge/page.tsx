"use client";

import { useEffect, useState, useCallback } from "react";
import { Boxes, Database, FileCode2, Plus } from "lucide-react";
import Container from "@/components/container";
import { EmbeddingScatter } from "@/components/chart-blocks";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { KBInfo, Projection2D } from "@/types/types";
import { cn } from "@/lib/utils";

const STRATEGY_OPTIONS = [
  { value: "direct", label: "Direct（直接注入）" },
  { value: "simple", label: "Simple（固定大小）" },
  { value: "semantic", label: "Semantic（语义分块）" },
  { value: "code_aware", label: "Code Aware（代码感知）" },
  { value: "proposition", label: "Proposition（原子命题）" },
];

const KIND_STYLE: Record<string, { label: string; cls: string }> = {
  doc: { label: "文档", cls: "bg-primary/10 text-primary" },
  code_json: { label: "JSON", cls: "bg-accent/10 text-accent-foreground" },
  code_c: { label: "C 源码", cls: "bg-indigo-500/10 text-indigo-600" },
  code_h: { label: "头文件", cls: "bg-sky-500/10 text-sky-600" },
};

interface ChunkItem {
  chunk_id: string;
  text_preview: string;
  kind: string;
  file: string;
  token_count: number;
}

export default function KnowledgePage() {
  const [kbs, setKbs] = useState<KBInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showBuild, setShowBuild] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildMsg, setBuildMsg] = useState("");
  const [selectedKb, setSelectedKb] = useState<string | null>(null);
  const [projection, setProjection] = useState<Projection2D[] | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);

  const [formSource, setFormSource] = useState("");
  const [formStrategy, setFormStrategy] = useState("direct");
  const [formChunkSize, setFormChunkSize] = useState(500);
  const [formOverlap, setFormOverlap] = useState(50);
  const [formForce, setFormForce] = useState(false);

  const loadKbs = useCallback(async () => {
    try {
      const res = await api.kbList();
      setKbs(res.knowledge_bases || []);
      // 默认展开第一个知识库的可视化
      if (res.knowledge_bases?.length && !selectedKb) {
        loadVisualization(res.knowledge_bases[0].source);
      }
    } catch {
      setKbs([]);
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { loadKbs(); }, [loadKbs]);

  const loadVisualization = useCallback(async (source: string) => {
    setSelectedKb(source);
    setProjection(null);
    setChunks([]);
    try {
      const [viz, chk] = await Promise.all([
        api.kbVisualization(source).catch(() => null),
        api.kbChunks(source).catch(() => null),
      ]);
      setProjection(viz?.points || null);
      setChunks(chk?.chunks || []);
      setChunkTotal(chk?.total || 0);
    } catch {
      setProjection(null);
    }
  }, []);

  const handleBuild = async () => {
    if (!formSource.trim()) return;
    setBuilding(true);
    setBuildMsg("构建中...");
    try {
      const res = await api.kbBuild(formSource, formStrategy, formChunkSize, formOverlap, formForce);
      const poll = async () => {
        try {
          const status = await api.taskStatus(res.task_id);
          if (status.status === "completed") {
            setBuildMsg("构建完成");
            setBuilding(false);
            setShowBuild(false);
            loadKbs();
          } else if (status.status === "failed") {
            setBuildMsg(`构建失败: ${status.error || "未知错误"}`);
            setBuilding(false);
          } else {
            setBuildMsg(status.message || "处理中...");
            setTimeout(poll, 2000);
          }
        } catch {
          setBuildMsg("状态查询失败");
          setBuilding(false);
        }
      };
      setTimeout(poll, 2000);
    } catch (e: unknown) {
      setBuildMsg(`构建失败: ${e instanceof Error ? e.message : "未知错误"}`);
      setBuilding(false);
    }
  };

  return (
    <div>
      {/* Toolbar */}
      <Container className="flex items-center justify-between border-b border-border py-3.5">
        <div className="flex items-center gap-2.5">
          <Database size={16} strokeWidth={1.8} className="text-primary" />
          <h2 className="text-sm font-medium">知识库</h2>
          <span className="rounded-full bg-primary/10 px-2 py-0.5 font-mono text-[11px] text-primary">
            {kbs.length} 个已构建
          </span>
        </div>
        <Button onClick={() => setShowBuild(true)} size="sm">
          <Plus size={14} className="mr-1" /> 构建知识库
        </Button>
      </Container>

      {/* KB 列表 */}
      <Container className="py-4">
        {loading ? (
          <div className="skeleton h-40 rounded-lg" />
        ) : kbs.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border py-10 text-center">
            <Boxes size={28} className="mx-auto text-muted-foreground/50" />
            <p className="mt-3 text-sm text-muted-foreground">暂无知识库，点击「构建知识库」开始</p>
          </div>
        ) : (
          <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft-sm">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">来源</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">分块策略</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Chunks</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">Embedding 模型</th>
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-muted-foreground">操作</th>
                </tr>
              </thead>
              <tbody>
                {kbs.map((kb) => (
                  <tr
                    key={kb.source}
                    className={cn(
                      "cursor-pointer border-t border-border transition-colors hover:bg-muted/30",
                      selectedKb === kb.source && "bg-primary/5",
                    )}
                    onClick={() => loadVisualization(kb.source)}
                  >
                    <td className="px-4 py-2.5 font-mono font-medium">{kb.source}</td>
                    <td className="px-4 py-2.5">
                      <span className="rounded bg-secondary px-2 py-0.5 font-mono text-[11px] text-secondary-foreground">
                        {kb.chunk_strategy}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono tabular-nums">{kb.chunk_count}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{kb.embedding_model}</td>
                    <td className="px-4 py-2.5">
                      <Button variant="ghost" size="sm">
                        可视化
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Container>

      {/* 可视化 + Chunks 样本 */}
      {selectedKb && (
        <Container className="border-t border-border py-5">
          <div className="mb-3 flex items-baseline gap-2.5">
            <h3 className="text-[15px] font-semibold tracking-tight">
              向量空间投影 · <span className="font-mono text-primary">{selectedKb}</span>
            </h3>
            <span className="font-mono text-[10.5px] text-muted-foreground">UMAP 2D · 按知识类型着色</span>
          </div>
          <div className="grid gap-4 xl:grid-cols-5">
            <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft-sm xl:col-span-3">
              {projection ? (
                <EmbeddingScatter data={projection} />
              ) : (
                <div className="skeleton h-[380px]" />
              )}
            </div>
            <div className="overflow-hidden rounded-lg border border-border bg-card shadow-soft-sm xl:col-span-2">
              <div className="flex items-center gap-2 border-b border-border bg-muted/50 px-4 py-2.5">
                <FileCode2 size={14} className="text-primary" />
                <span className="text-xs font-medium">Chunks 样本</span>
                <span className="ml-auto font-mono text-[10.5px] text-muted-foreground">
                  显示 {chunks.length}/{chunkTotal}
                </span>
              </div>
              <div className="max-h-[380px] divide-y divide-border overflow-y-auto">
                {chunks.length === 0 && (
                  <div className="p-4 text-xs text-muted-foreground">加载中...</div>
                )}
                {chunks.map((c) => {
                  const ks = KIND_STYLE[c.kind] || { label: c.kind, cls: "bg-muted text-muted-foreground" };
                  return (
                    <div key={c.chunk_id} className="px-4 py-2.5">
                      <div className="mb-1 flex items-center gap-2">
                        <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium", ks.cls)}>{ks.label}</span>
                        <span className="truncate font-mono text-[10.5px] text-muted-foreground">{c.file}</span>
                        <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground/60">
                          {c.token_count} tok
                        </span>
                      </div>
                      <p className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">{c.text_preview}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Container>
      )}

      {/* 构建弹窗 */}
      {showBuild && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => !building && setShowBuild(false)}>
          <div className="top-beam mx-4 w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-soft-lg" onClick={(e) => e.stopPropagation()}>
            <h3 className="mb-1 text-lg font-semibold tracking-tight">构建知识库</h3>
            <p className="mb-4 text-xs text-muted-foreground">解析 data/source_proto/&lt;来源&gt; 下的文档与代码，生成向量索引</p>
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-sm font-medium">来源标识</label>
                <input
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none transition-shadow focus:ring-2 focus:ring-ring/40"
                  placeholder="如: gdwx"
                  value={formSource}
                  onChange={(e) => setFormSource(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">分块策略</label>
                <select
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                  value={formStrategy}
                  onChange={(e) => setFormStrategy(e.target.value)}
                >
                  {STRATEGY_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
              {formStrategy !== "direct" && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="mb-1 block text-sm font-medium">Chunk Size</label>
                    <input
                      type="number"
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                      value={formChunkSize}
                      onChange={(e) => setFormChunkSize(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-sm font-medium">Overlap</label>
                    <input
                      type="number"
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                      value={formOverlap}
                      onChange={(e) => setFormOverlap(Number(e.target.value))}
                    />
                  </div>
                </div>
              )}
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={formForce}
                  onChange={(e) => setFormForce(e.target.checked)}
                  className="rounded accent-[hsl(var(--primary))]"
                />
                强制重建（删除已有索引）
              </label>
            </div>
            {buildMsg && <div className="mt-3 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">{buildMsg}</div>}
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowBuild(false)} disabled={building}>
                取消
              </Button>
              <Button onClick={handleBuild} disabled={building || !formSource.trim()}>
                {building ? "构建中..." : "开始构建"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
