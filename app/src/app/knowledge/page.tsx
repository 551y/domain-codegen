"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Database, Plane, Satellite, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { OnnxDataset, OnnxRecord, PaddleDataset, ProtocolChunk, ProtocolDataset } from "@/lib/api";
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

// ---------- 数据集可视化（两个知识库共用同一实现，风格与数据集独立页一致） ----------

interface VizRecord {
  id: string;
  title: string;
  category: string;
  category_label: string;
  text: string;
  isDoc: boolean;
  meta: [string, string][];
}

interface DatasetVizConfig {
  headerNote: string;
  primary: { total: number; label: string; sub: string };
  secondary?: { total: number; label: string; sub: string };
  kinds: { key: string; label: string; count: number }[];
  kindColors: Record<string, string>;
  records: VizRecord[];
  files: { path: string; lines: number; note: string }[];
}

const fmtNum = (v: number) => v.toLocaleString("en-US");

function DatasetViz({ cfg }: { cfg: DatasetVizConfig }) {
  const [sel, setSel] = useState(0);
  useEffect(() => { setSel(0); }, [cfg.records]);
  const rec = cfg.records[sel];

  // 同类连续排列（与数据集独立页点阵同口径：100 条一组、10×10，块过多时折行）
  const kindOrder = cfg.kinds.map((k) => k.key);
  const order = useMemo(() => {
    const idx: number[] = [];
    for (const k of kindOrder)
      cfg.records.forEach((r, i) => { if (r.category === k) idx.push(i); });
    return idx;
  }, [cfg]);
  const posOf = useMemo(() => {
    const m = new Map<number, number>();
    order.forEach((recIdx, offset) => m.set(recIdx, offset));
    return m;
  }, [order]);
  const posInOrder = posOf.get(sel) ?? 0;

  const pitch = 11, gap = 26, gapY = 10, labelH = 15;
  const groupCount = Math.ceil(order.length / 100);
  const cols = Math.max(1, Math.min(groupCount, 7));
  const rows = Math.max(1, Math.ceil(groupCount / cols));
  const vbW = cols * 10 * pitch + (cols - 1) * gap;
  const vbH = rows * (10 * pitch + labelH) + (rows - 1) * gapY + 6;
  const move = (d: number) => {
    const next = posInOrder + d;
    if (next >= 0 && next < order.length) setSel(order[next]);
  };

  return (
    <div className="mb-4 overflow-hidden rounded-xl border border-border bg-card">
      <div className="border-b border-border bg-muted/50 px-4 py-2 font-mono text-[10.5px] text-muted-foreground">
        {cfg.headerNote}
      </div>

      {/* 主数字 + 对照口径 */}
      <div className="flex flex-wrap items-end gap-x-8 gap-y-3 px-4 pt-4">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-[30px] font-bold leading-none tracking-tight">{fmtNum(cfg.primary.total)}</span>
            <span className="text-[12.5px] text-muted-foreground">{cfg.primary.label}</span>
          </div>
          <div className="mt-1 font-mono text-[9.5px] text-muted-foreground">{cfg.primary.sub}</div>
        </div>
        {cfg.secondary && (
          <div className="border-l border-border pl-4">
            <div className="flex items-baseline gap-2">
              <span className="text-[16px] font-semibold leading-none">{fmtNum(cfg.secondary.total)}</span>
              <span className="text-[11px] text-muted-foreground">{cfg.secondary.label}</span>
            </div>
            <div className="mt-1 font-mono text-[9.5px] text-muted-foreground">{cfg.secondary.sub}</div>
          </div>
        )}
        <div className="ml-auto flex max-w-[46%] flex-wrap items-center justify-end gap-x-3 gap-y-1 pb-0.5">
          {cfg.kinds.map((k) => (
            <span key={k.key} className="flex items-center gap-1.5 font-mono text-[10.5px] text-muted-foreground">
              <span className="h-2 w-2 rounded-full" style={{ background: cfg.kindColors[k.key] }} />
              {k.label} <b className="text-foreground">{fmtNum(k.count)}</b>
            </span>
          ))}
        </div>
      </div>

      {/* 点阵图：100 条一组、同类连续排列 */}
      <div className="px-4 pt-3">
        {cfg.records.length === 0 && <p className="py-4 text-center text-[12px] text-muted-foreground">无匹配记录</p>}
        <svg viewBox={`0 0 ${vbW} ${vbH}`} className="h-auto w-full max-w-[760px]" role="img"
          aria-label={`全部 ${cfg.records.length} 条记录，按类别着色`}>
          {order.map((recIdx, offset) => {
            const g = Math.floor(offset / 100), local = offset % 100;
            const cx = (g % cols) * (10 * pitch + gap) + (local % 10 + 0.5) * pitch;
            const cy = Math.floor(g / cols) * (10 * pitch + labelH + gapY) + (Math.floor(local / 10) + 0.5) * pitch;
            const r = cfg.records[recIdx];
            const active = sel === recIdx;
            return (
              <circle key={recIdx} cx={cx} cy={cy} r={active ? 4 : 3} fill={cfg.kindColors[r.category]}
                stroke={active ? "#1D1D1F" : "none"} strokeWidth={active ? 1.2 : 0}
                opacity={active ? 1 : 0.88} className="cursor-pointer" onClick={() => setSel(recIdx)}>
                <title>{r.title}</title>
              </circle>
            );
          })}
          {Array.from({ length: groupCount }, (_, g) => (
            <text key={g}
              x={(g % cols) * (10 * pitch + gap) + 5 * pitch}
              y={Math.floor(g / cols) * (10 * pitch + labelH + gapY) + 10 * pitch + labelH}
              textAnchor="middle" className="fill-muted-foreground" fontSize="10">
              {fmtNum(Math.min((g + 1) * 100, cfg.records.length))}
            </text>
          ))}
        </svg>
        <div className="mt-1 flex flex-wrap gap-3">
          {cfg.kinds.map((k) => (
            <button key={k.key}
              onClick={() => { const first = cfg.records.findIndex((r) => r.category === k.key); if (first >= 0) setSel(first); }}
              className="flex items-center gap-1.5 font-mono text-[10px] text-muted-foreground transition-colors hover:text-primary">
              <span className="h-2 w-2 rounded-full" style={{ background: cfg.kindColors[k.key] }} />
              {k.label} {fmtNum(k.count)}
            </button>
          ))}
        </div>
      </div>

      {/* 选中记录阅读卡 */}
      {rec && (
        <div className="mx-4 mt-3 overflow-hidden rounded-lg border border-border bg-muted/30">
          <div className="flex items-center gap-2 border-b border-border/60 px-3 py-1.5">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: cfg.kindColors[rec.category] }} />
            <span className="min-w-0 flex-1 truncate font-mono text-[11px] font-medium">{rec.title}</span>
            <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">{rec.category_label}</span>
            <button onClick={() => move(-1)} disabled={posInOrder === 0}
              className="shrink-0 px-1 text-[13px] text-primary disabled:text-muted-foreground/40">←</button>
            <span className="shrink-0 font-mono text-[9.5px] text-muted-foreground">{fmtNum(posInOrder + 1)} / {fmtNum(cfg.records.length)}</span>
            <button onClick={() => move(1)} disabled={posInOrder === cfg.records.length - 1}
              className="shrink-0 px-1 text-[13px] text-primary disabled:text-muted-foreground/40">→</button>
          </div>
          {rec.isDoc ? (
            <p className="code-scroll max-h-[200px] overflow-auto whitespace-pre-wrap break-all px-3 py-2.5 text-[12px] leading-relaxed text-foreground/85">{rec.text}</p>
          ) : (
            <pre className="code-scroll max-h-[200px] overflow-auto whitespace-pre-wrap break-all px-3 py-2.5 font-mono text-[10.5px] leading-relaxed text-foreground/90">{rec.text}</pre>
          )}
          <div className="flex flex-wrap gap-x-4 border-t border-border/60 px-3 py-1.5 font-mono text-[9.5px] text-muted-foreground">
            {rec.meta.map(([k, v]) => (
              <span key={k}>{k} <span className="text-foreground/80">{v}</span></span>
            ))}
          </div>
        </div>
      )}

      {/* 文件行数口径（数量形式：行数 ≠ 记录数，因文本字段内含换行符） */}
      <div className="overflow-x-auto px-4 py-3">
        <table className="w-full min-w-[460px] font-mono text-[10px]">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="pb-1.5 pr-4 font-medium">文件</th>
              <th className="pb-1.5 pr-4 font-medium">行数</th>
              <th className="pb-1.5 font-medium">说明</th>
            </tr>
          </thead>
          <tbody>
            {cfg.files.map((f) => (
              <tr key={f.path} className="border-b border-border/50 last:border-0">
                <td className="py-1.5 pr-4 text-foreground/85">{f.path}</td>
                <td className="py-1.5 pr-4 font-bold text-foreground">{fmtNum(f.lines)}</td>
                <td className="py-1.5 text-muted-foreground">{f.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 协议基本数据集：356 条原始基线为主，增强 3,090 对照（调色板同数据集独立页）
const DS_KIND_COLOR: Record<string, string> = { doc: "#5487C5", code_c: "#8E72BE", code_json: "#4F9D8A" };
const DS_KIND_LABEL: Record<string, string> = { doc: "协议文档", code_c: "C 代码", code_json: "配置 JSON" };
const DS_KIND_ORDER = ["doc", "code_c", "code_json"];

function ProtocolDatasetView({ data }: { data: ProtocolDataset }) {
  const cfg: DatasetVizConfig = {
    headerNote: "基本数据集可视化 · 星务协议知识库数据集",
    primary: {
      total: data.baseline.total, label: "条 · 基本数据集",
      sub: `${data.baseline.chunk} · ${data.baseline.vectorized} · gdwx ${fmtNum(data.baseline.protocols.gdwx)} + dllg ${fmtNum(data.baseline.protocols.dllg)}`,
    },
    secondary: {
      total: data.augmented.total, label: `条 · 数据增强后（×${data.augmented.ratio}）`,
      sub: `${data.augmented.chunk} · ${data.augmented.vectorized} · gdwx ${fmtNum(data.augmented.protocols.gdwx)} + dllg ${fmtNum(data.augmented.protocols.dllg)}`,
    },
    kinds: DS_KIND_ORDER.map((k) => ({ key: k, label: DS_KIND_LABEL[k], count: data.baseline.kinds[k] ?? 0 })),
    kindColors: DS_KIND_COLOR,
    records: data.records.map((r) => ({
      id: r.id, title: r.title, category: r.category, category_label: r.category_label,
      text: r.text, isDoc: r.category === "doc",
      meta: [["分块编号", r.id], ["来源协议", r.source], ["规模", `${fmtNum(r.char_count)} 字符 · ${fmtNum(r.token_count)} tok`]],
    })),
    files: data.files,
  };
  return <DatasetViz cfg={cfg} />;
}

// 模型部署适配数据集：当前交付 1,349 条为主，基线+增量与精选口径对照（调色板同数据集独立页）
const ONNX_PALETTE = ["#5487C5", "#8E72BE", "#4F9D8A", "#CD9251", "#C46D87", "#609BB2", "#8A9B50", "#AF7C5F", "#808590"];

function ModelDatasetView({ data }: { data: OnnxDataset }) {
  const cfg: DatasetVizConfig = {
    headerNote: "数据集可视化 · 模型部署适配知识库数据集",
    primary: {
      total: data.total, label: "条 · 当前交付数据集",
      sub: `基线 ${fmtNum(data.composition.baseline)} 条 + 实测验证增量 ${fmtNum(data.composition.additions)} 条（49 完整示例 + 32 修复对）`,
    },
    kinds: data.categories.map((c) => ({ key: c.key, label: c.label, count: c.count })),
    kindColors: Object.fromEntries(data.categories.map((c, i) => [c.key, ONNX_PALETTE[i % ONNX_PALETTE.length]])),
    records: data.records.map((r) => ({
      id: r.id, title: r.title, category: r.category, category_label: r.category_label,
      text: r.text, isDoc: r.category === "document",
      meta: [["编号", r.id], ["来源", r.source], ["框架版本", r.framework_version || "—"], ["验证标签", r.verification_status || "—"]],
    })),
    files: data.files,
  };
  return <DatasetViz cfg={cfg} />;
}

const PADDLE_PALETTE = ["#5487C5", "#CD9251", "#8E72BE", "#C46D87", "#808590"];

function PaddleDatasetView({ data }: { data: PaddleDataset }) {
  const cfg: DatasetVizConfig = {
    headerNote: "数据集可视化 · ModelX PyTorch→Paddle 映射规则",
    primary: {
      total: data.total, label: "条 · ModelX 映射规则",
      sub: `直接映射 ${fmtNum(data.composition.direct)} · 参数改写 ${fmtNum(data.composition.parameter_rewrites)} · 语义转换 ${fmtNum(data.composition.semantic_rewrites)}`,
    },
    secondary: {
      total: data.composition.unsupported, label: "条 · 不支持规则",
      sub: "来自 pytorch_to_paddlepaddle.json 的 Unsupported 条目",
    },
    kinds: data.categories.map((c) => ({ key: c.key, label: c.label, count: c.count })),
    kindColors: Object.fromEntries(data.categories.map((c, i) => [c.key, PADDLE_PALETTE[i % PADDLE_PALETTE.length]])),
    records: data.records.map((r) => ({
      id: r.id, title: r.title, category: r.category, category_label: r.category_label,
      text: r.text, isDoc: false,
      meta: [
        ["源 API", r.source_api], ["目标 API", r.target_api || "—"],
        ["Matcher", r.matcher || "—"], ["参数改写", Object.keys(r.params_change || {}).join(", ") || "—"],
      ],
    })),
    files: data.files,
  };
  return <DatasetViz cfg={cfg} />;
}

export default function KnowledgePage() {
  const [tab, setTab] = useState<"protocol" | "operator" | "paddle">("protocol");
  const [query, setQuery] = useState("");
  const [chunks, setChunks] = useState<ProtocolChunk[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [kbSource, setKbSource] = useState("");
  const [records, setRecords] = useState<OnnxRecord[]>([]);
  const [recordTotal, setRecordTotal] = useState(0);
  const [protoDs, setProtoDs] = useState<ProtocolDataset | null>(null);
  const [onnxDs, setOnnxDs] = useState<OnnxDataset | null>(null);
  const [paddleDs, setPaddleDs] = useState<PaddleDataset | null>(null);
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
    api.protocolDataset().then(setProtoDs).catch(() => {});
    api.onnxDataset().then(setOnnxDs).catch(() => {});
    api.paddleDataset().then(setPaddleDs).catch(() => {});
  }, []);

  // 搜索（防抖）
  useEffect(() => {
    const t = setTimeout(() => {
      if (tab === "protocol") {
        api.protocolChunks(query).then((r) => setChunks(r.chunks)).catch(() => {});
      } else if (tab === "operator") {
        api.onnxRecords(query).then((r) => setRecords(r.records)).catch(() => {});
      }
    }, 250);
    return () => clearTimeout(t);
  }, [query, tab]);

  const filteredPaddleDs = useMemo(() => {
    if (!paddleDs || tab !== "paddle" || !query.trim()) return paddleDs;
    const q = query.trim().toLowerCase();
    const records = paddleDs.records.filter((r) =>
      [r.title, r.source_api, r.target_api, r.matcher, r.category_label, r.text]
        .some((v) => String(v || "").toLowerCase().includes(q)),
    );
    const counts = new Map(records.map((r) => [r.category, 0]));
    records.forEach((r) => counts.set(r.category, (counts.get(r.category) || 0) + 1));
    return {
      ...paddleDs,
      total: records.length,
      categories: paddleDs.categories.map((c) => ({ ...c, count: counts.get(c.key) || 0 })),
      composition: {
        direct: counts.get("direct_mapping") || 0,
        parameter_rewrites: counts.get("parameter_rewrite") || 0,
        semantic_rewrites: counts.get("semantic_rewrite") || 0,
        unsupported: counts.get("unsupported") || 0,
      },
      records,
    };
  }, [paddleDs, query, tab]);

  return (
    <div className="mx-auto w-full max-w-[1100px] px-6 py-6">
      <div className="mb-4 flex items-center justify-center gap-3">
        <h1 className="text-[18px] font-bold tracking-tight">领域知识库与转换规则</h1>
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
          模型部署适配知识库
          <span className="font-mono text-[10px] text-muted-foreground">{recordTotal} 条知识</span>
        </button>
        <button
          onClick={() => setTab("paddle")}
          className={cn(
            "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-[13px] transition-colors",
            tab === "paddle" ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Database size={14} className="text-primary" />
          Paddle 转换规则
          <span className="font-mono text-[10px] text-muted-foreground">{paddleDs?.total ?? 0} 条规则</span>
        </button>
      </div>

      {/* 搜索 */}
      <div className="relative mb-4">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={tab === "protocol" ? "搜索知识块内容 / 编号…" : tab === "operator" ? "搜索 ONNX 算子（如 conv2d / batchnorm）…" : "搜索 Paddle 源 API / 目标 API / Matcher…"}
          className="w-full rounded-lg border border-input bg-card py-2.5 pl-9 pr-3 text-[13px] outline-none transition-colors focus:border-primary/50"
        />
      </div>

      {/* 协议知识块 */}
      {tab === "protocol" && protoDs && <ProtocolDatasetView data={protoDs} />}
      {tab === "protocol" && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="border-b border-border bg-muted/50 px-4 py-2 font-mono text-[10.5px] text-muted-foreground">
            在线知识库（增强版）· 来源：{kbSource} · 显示 {chunks.length}/{chunkTotal}
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
      {tab === "operator" && onnxDs && <ModelDatasetView data={onnxDs} />}
      {tab === "operator" && (
        <div className="overflow-hidden rounded-xl border border-border bg-card">
          <div className="border-b border-border bg-muted/50 px-4 py-2 font-mono text-[10.5px] text-muted-foreground">
            在线知识库 · 算子映射 / 复合实现 / 测试用例 / 基准测试 / 技术文档等 9 类 · 显示 {records.length}/{recordTotal}
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

      {tab === "paddle" && filteredPaddleDs && <PaddleDatasetView data={filteredPaddleDs} />}

    </div>
  );
}

function ChevronDownIcon() {
  return <ChevronDown size={13} className="shrink-0 text-muted-foreground" />;
}
function ChevronUpIcon() {
  return <ChevronUp size={13} className="shrink-0 text-muted-foreground" />;
}
