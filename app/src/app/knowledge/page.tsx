"use client";

import { useEffect, useState } from "react";
import { ChevronDown, ChevronUp, Database, Plane, Satellite, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { OnnxRecord, ProtocolChunk } from "@/lib/api";
import { cn } from "@/lib/utils";

const KIND_STYLE: Record<string, { label: string; cls: string }> = {
  doc: { label: "文档", cls: "bg-primary/10 text-primary" },
  code_json: { label: "JSON 配置", cls: "bg-accent/10 text-accent" },
  code_c: { label: "C 源码", cls: "bg-indigo-500/10 text-indigo-500" },
  code_h: { label: "头文件", cls: "bg-purple-500/10 text-purple-500" },
};

const CATEGORY_STYLE: Record<string, string> = {
  算子映射: "bg-primary/10 text-primary",
  复合算子实现: "bg-indigo-500/10 text-indigo-500",
  算子测试用例: "bg-accent/10 text-accent",
  模型基准测试: "bg-amber-500/10 text-amber-600",
  技术文档: "bg-purple-500/10 text-purple-500",
  问题修复案例: "bg-emerald-500/10 text-emerald-600",
  版本兼容矩阵: "bg-muted text-muted-foreground",
  错误模式: "bg-destructive/10 text-destructive",
  转换示例: "bg-sky-500/10 text-sky-600",
};

export default function KnowledgePage() {
  const [tab, setTab] = useState<"protocol" | "operator">("protocol");
  const [query, setQuery] = useState("");
  const [chunks, setChunks] = useState<ProtocolChunk[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [kbSource, setKbSource] = useState("");
  const [records, setRecords] = useState<OnnxRecord[]>([]);
  const [recordTotal, setRecordTotal] = useState(0);
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const [openRecord, setOpenRecord] = useState<string | null>(null);

  // 初始加载
  useEffect(() => {
    api.protocolChunks("").then((r) => {
      setChunks(r.chunks);
      setChunkTotal(r.total);
      setKbSource(r.source);
    }).catch(() => {});
    api.onnxRecords("").then((r) => {
      setRecords(r.records);
      setRecordTotal(r.total);
    }).catch(() => {});
  }, []);

  // 搜索（防抖）
  useEffect(() => {
    const t = setTimeout(() => {
      if (tab === "protocol") {
        api.protocolChunks(query).then((r) => setChunks(r.chunks)).catch(() => {});
      } else {
        api.onnxRecords(query).then((r) => setRecords(r.records)).catch(() => {});
      }
    }, 250);
    return () => clearTimeout(t);
  }, [query, tab]);

  return (
    <div className="mx-auto w-full max-w-[1100px] px-6 py-6">
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-[18px] font-bold tracking-tight">两套领域知识库</h1>
        <span className="font-mono text-[10.5px] text-muted-foreground">
          同一构建规范：知识单元结构化 · 逐条验证 · 可机读
        </span>
      </div>

      {/* Tab 切换 */}
      <div className="mb-4 flex gap-1 rounded-lg border border-border bg-muted p-1">
        <button
          onClick={() => setTab("protocol")}
          className={cn(
            "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-[13px] transition-colors",
            tab === "protocol" ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Satellite size={14} className="text-primary" />
          星务协议知识库
          <span className="font-mono text-[10px] text-muted-foreground">{chunkTotal} 知识块</span>
        </button>
        <button
          onClick={() => setTab("operator")}
          className={cn(
            "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-[13px] transition-colors",
            tab === "operator" ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Plane size={14} className="text-primary" />
          PyTorch→ONNX 知识库
          <span className="font-mono text-[10px] text-muted-foreground">{recordTotal} 条知识</span>
        </button>
      </div>

      {/* 搜索 */}
      <div className="relative mb-4">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={tab === "protocol" ? "搜索知识块内容 / 编号…" : "搜索算子（如 conv2d / batchnorm）…"}
          className="w-full rounded-lg border border-input bg-card py-2.5 pl-9 pr-3 text-[13px] outline-none transition-colors focus:border-primary/50"
        />
      </div>

      {/* 协议知识块 */}
      {tab === "protocol" && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="border-b border-border bg-muted/50 px-4 py-2 font-mono text-[10.5px] text-muted-foreground">
            来源知识库：{kbSource} · 显示 {chunks.length}/{chunkTotal}
          </div>
          <div className="divide-y divide-border">
            {chunks.map((c) => {
              const ks = KIND_STYLE[c.kind] || { label: c.kind, cls: "bg-muted text-muted-foreground" };
              const open = openChunk === c.chunk_id;
              return (
                <div key={c.chunk_id}>
                  <button
                    onClick={() => setOpenChunk(open ? null : c.chunk_id)}
                    className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-muted/40"
                  >
                    <span className={cn("shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]", ks.cls)}>
                      {ks.label}
                    </span>
                    <span className="min-w-0 flex-1 truncate font-mono text-[11.5px]">{c.chunk_id}</span>
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground/70">
                      {c.file} · {c.token_count} tok
                    </span>
                    {open ? <ChevronUpIcon /> : <ChevronDownIcon />}
                  </button>
                  {open && (
                    <p className="whitespace-pre-wrap break-all border-t border-border bg-muted/30 px-4 py-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
                      {c.text_preview}
                    </p>
                  )}
                </div>
              );
            })}
            {chunks.length === 0 && (
              <div className="px-4 py-8 text-center text-[12px] text-muted-foreground">无匹配知识块</div>
            )}
          </div>
        </div>
      )}

      {/* ONNX 知识记录 */}
      {tab === "operator" && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="border-b border-border bg-muted/50 px-4 py-2 font-mono text-[10.5px] text-muted-foreground">
            算子映射 / 复合实现 / 测试用例 / 基准测试 / 技术文档等 9 类 · 显示 {records.length}/{recordTotal}
          </div>
          <div className="divide-y divide-border">
            {records.map((r) => {
              const open = openRecord === r.id;
              return (
                <div key={r.id}>
                  <button
                    onClick={() => setOpenRecord(open ? null : r.id)}
                    className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-muted/40"
                  >
                    <span className={cn("shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]", CATEGORY_STYLE[r.category_label] || "bg-muted text-muted-foreground")}>
                      {r.category_label}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[12.5px]">{r.title}</span>
                    <span className="max-w-[280px] truncate shrink-0 font-mono text-[10px] text-muted-foreground/70">
                      {r.source}
                    </span>
                    {open ? <ChevronUpIcon /> : <ChevronDownIcon />}
                  </button>
                  {open && (
                    <pre className="whitespace-pre-wrap break-all border-t border-border bg-muted/30 px-4 py-3 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                      {r.preview}
                    </pre>
                  )}
                </div>
              );
            })}
            {records.length === 0 && (
              <div className="px-4 py-8 text-center text-[12px] text-muted-foreground">无匹配记录</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ChevronDownIcon() {
  return <ChevronDown size={13} className="shrink-0 text-muted-foreground" />;
}
function ChevronUpIcon() {
  return <ChevronUp size={13} className="shrink-0 text-muted-foreground" />;
}
