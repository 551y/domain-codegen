"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowLeftRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Download,
  FileCode2,
  FileText,
  Loader2,
  Package,
  Plane,
  Play,
  RotateCcw,
  Satellite,
  Upload,
  X,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import type { GenerateResult, KbCard, SampleItem, TaskIntermediate, TaskState } from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------- 常量 ----------

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

// 任务场景卡元信息（首页卡片 id：protocol / model）
const KB_META: Record<string, { icon: typeof Satellite; title: string; subtitle: string; desc: string }> = {
  protocol: {
    icon: Satellite,
    title: "卫星协议代码生成",
    subtitle: "Protocol Handler Generation",
    desc: "上传星务协议文档（.docx），检索增强生成遥测解析、遥控组包、内部总线交互等协议处理代码（config.json + handler.c），并对生成结果自动评测。",
  },
  model: {
    icon: Plane,
    title: "无人机模型部署适配",
    subtitle: "PyTorch → ONNX / PaddlePaddle",
    desc: "上传 PyTorch 模型源码（.py），注入模型部署适配知识库（两个目标共用同一数据集）：生成 ONNX 导出代码并真实执行导出、产出可下载的 .onnx 模型文件与配套机载 C++ 推理源码；或生成 PaddlePaddle 跨框架转换代码（含语义对齐处理），全流程自动评测。",
  },
};

// 任务条标签（按后端知识库 id：模型入口的两个目标各对应一条后端流水线）
const KB_CHIP: Record<string, string> = { protocol: "协议", operator: "ONNX", paddle: "Paddle" };

// 模型入口的目标框架选项（值即后端流水线 id；两个目标检索同一套模型部署适配知识库）
const TARGET_META: Record<"operator" | "paddle", { icon: typeof Satellite; title: string; sub: string }> = {
  operator: { icon: Plane, title: "PyTorch → ONNX", sub: "部署导出 · 产出 .onnx + C++ 推理源码" },
  paddle: { icon: ArrowLeftRight, title: "PyTorch → PaddlePaddle", sub: "跨框架转换 · 产出 Paddle 模型源码" },
};

const STAGES = ["解析输入", "检索知识库", "生成目标代码", "评测结果"];

function stageIndex(progress: number): number {
  if (progress < 15) return 0;
  if (progress < 40) return 1;
  if (progress < 82) return 2;
  return 3;
}

function isUnfinished(status: string): boolean {
  return status === "running" || status === "pending";
}

interface TrackedTask {
  id: string;
  kb: string;
  sourceId: string;
  label: string;
  startedAt: number;
  state: TaskState;
  result?: GenerateResult;
}

function Stepper({
  kb,
  source,
  active,
  hasResult,
  onReset,
  className,
}: {
  kb: string | null;
  source: boolean;
  active: boolean;
  hasResult: boolean;
  onReset?: () => void;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-2 font-mono text-[10.5px] tracking-wide text-muted-foreground", className)}>
      {["选择知识库", "上传代码", "生成", "结果与评测"].map((s, i) => (
        <div key={s} className="flex items-center gap-2">
          <span
            className={cn(
              "flex h-5 w-5 items-center justify-center rounded-full border text-[10px]",
              (i === 0 && kb) || (i === 1 && source) || (i === 2 && active) || (i === 3 && hasResult)
                ? "border-primary bg-primary/10 text-primary"
                : "border-border",
            )}
          >
            {i + 1}
          </span>
          <span>{s}</span>
          {i < 3 && <span className="text-border">——</span>}
        </div>
      ))}
      {kb && onReset && (
        <button onClick={onReset} className="ml-auto flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-[10.5px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary">
          <RotateCcw size={11} /> 重新选择
        </button>
      )}
    </div>
  );
}

// ---------- 中间过程输出（解析/检索等阶段产出，执行中实时展示） ----------

function LiveBox({ live }: { live: { stage: string; text: string; done?: boolean } }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [live.text]);
  return (
    <div>
      <div className="mb-1 flex items-center gap-1.5 text-[11.5px] text-muted-foreground">
        {live.done ? (
          <CheckCircle2 size={11} className="text-emerald-500" />
        ) : (
          <Loader2 size={11} className="animate-spin text-primary" />
        )}
        实时生成输出 · <span className="font-medium text-foreground">{live.stage}</span>
        <span className="font-mono text-[9.5px]">{live.done ? "已完成" : "LLM 流式"}</span>
      </div>
      <pre
        ref={ref}
        className="code-scroll max-h-[260px] overflow-auto whitespace-pre-wrap break-all rounded-lg bg-muted/70 p-2.5 font-mono text-[10.5px] leading-relaxed text-foreground/90"
      >
        {live.text}
        {!live.done && <span className="animate-pulse text-primary">▍</span>}
      </pre>
    </div>
  );
}

function IntermediateView({ data }: { data: TaskIntermediate }) {
  const hasAny =
    data.gen_mode || data.parsed || data.operators ||
    (data.retrieved && data.retrieved.length > 0) || data.retrieval || data.config_stats ||
    data.live || data.live_config || data.live_handler_plan || data.live_handler;
  if (!hasAny) return null;
  return (
    <div className="mt-3 space-y-2.5 rounded-lg border border-border/70 bg-card/70 p-3">
      <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        中间过程输出
      </div>

      {data.live && <LiveBox live={data.live} />}
      {data.live_config && <LiveBox live={data.live_config} />}
      {data.live_handler_plan && <LiveBox live={data.live_handler_plan} />}
      {data.live_handler && <LiveBox live={data.live_handler} />}

      {data.gen_mode && (
        <div className="flex items-center gap-2 text-[11.5px]">
          <span className="text-muted-foreground">生成方式</span>
          <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] text-primary">{data.gen_mode}</span>
        </div>
      )}

      {data.parsed && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
          {Object.entries(data.parsed).map(([k, v]) => (
            <span key={k}>{k}：<span className="text-foreground">{v}</span></span>
          ))}
        </div>
      )}

      {data.operators && (
        <div>
          <div className="mb-1 text-[11.5px] text-muted-foreground">
            解析输入：检测到 <span className="font-bold text-foreground">{data.operators.total_calls}</span> 个算子调用
          </div>
          <div className="flex flex-wrap gap-1">
            {data.operators.leaf_ops.slice(0, 16).map((op) => (
              <span key={op} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-foreground/80">{op}</span>
            ))}
            {data.operators.leaf_ops.length > 16 && (
              <span className="px-1 font-mono text-[10px] text-muted-foreground">…</span>
            )}
          </div>
        </div>
      )}

      {data.retrieval && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
          {Object.entries(data.retrieval).map(([k, v]) => (
            <span key={k}>{k}：<span className="font-bold text-foreground">{v}</span></span>
          ))}
        </div>
      )}

      {data.retrieved && data.retrieved.length > 0 && (
        <div>
          <div className="mb-1 text-[11.5px] text-muted-foreground">
            检索知识库：命中 <span className="font-bold text-foreground">{data.retrieved.length}</span> 条知识（离线检索，将注入生成上下文）
          </div>
          <div className="code-scroll max-h-[180px] space-y-1 overflow-auto">
            {data.retrieved.map((r, i) => (
              <div key={i} className="rounded bg-muted/40 px-2 py-1.5">
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate font-mono text-[10px]">{r.title}</span>
                  <span className={cn("shrink-0 rounded px-1 py-0.5 font-mono text-[9px]", CATEGORY_STYLE[r.category_label] || "bg-muted")}>{r.category_label}</span>
                </div>
                <div className="mt-0.5 truncate font-mono text-[9px] text-muted-foreground/60">
                  {r.source} · 相关度 {r.score}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.config_stats && Object.keys(data.config_stats).length > 0 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
          <span>Config 阶段产物：</span>
          {Object.entries(data.config_stats).map(([k, v]) => (
            <span key={k}>{k}：<span className="font-bold text-foreground">{v}</span></span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------- 主页面 ----------

export default function WorkbenchPage() {
  const [kbs, setKbs] = useState<KbCard[]>([]);
  const [kb, setKb] = useState<string | null>(null);
  const [target, setTarget] = useState<"operator" | "paddle">("operator");
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [source, setSource] = useState<{ mode: "sample" | "upload"; id: string; label: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState("");
  const [submitErr, setSubmitErr] = useState("");
  const [tasks, setTasks] = useState<Record<string, TrackedTask>>({});
  const [focusId, setFocusId] = useState<string | null>(null);
  const [inputOpen, setInputOpen] = useState(true);
  const [activeFile, setActiveFile] = useState(0);
  const [showHits, setShowHits] = useState(false);
  const [showFuncs, setShowFuncs] = useState(false);
  const tasksRef = useRef<Record<string, TrackedTask>>({});

  useEffect(() => {
    tasksRef.current = tasks;
  }, [tasks]);

  useEffect(() => {
    api.kbs().then((r) => setKbs(r.kbs)).catch(() => {});
  }, []);

  // 单轮询循环：并行刷新所有未完成任务；完成的任务自动拉取结果缓存
  useEffect(() => {
    const timer = setInterval(async () => {
      const pending = Object.values(tasksRef.current).filter((t) => isUnfinished(t.state.status));
      for (const t of pending) {
        try {
          const s = await api.task(t.id);
          if (s.status === "completed") {
            const r = await api.result(t.id).catch(() => null);
            setTasks((prev) =>
              prev[t.id] ? { ...prev, [t.id]: { ...prev[t.id], state: s, result: r ?? prev[t.id].result } } : prev,
            );
          } else {
            setTasks((prev) =>
              prev[t.id] ? { ...prev, [t.id]: { ...prev[t.id], state: s } } : prev,
            );
          }
        } catch { /* 单次轮询失败忽略 */ }
      }
    }, 1500);
    return () => clearInterval(timer);
  }, []);

  // 实际调用后端的流水线 id：模型入口按目标框架解析为 operator（ONNX）/ paddle（同一知识库）
  const activeKb = kb === "model" ? target : kb;
  const prevCardRef = useRef<string | null>(null);

  // 样例列表随有效流水线刷新；仅场景卡切换时复位输入——模型入口内切换目标框架保留
  // 已选样例/上传（两目标共用同一批样例 id，上传产物后端跨目录可解析）
  useEffect(() => {
    if (!activeKb) {
      prevCardRef.current = null;
      return;
    }
    if (prevCardRef.current !== kb) {
      setSource(null);
      setFocusId(null);
      prevCardRef.current = kb;
    }
    setSamples([]);
    api.samples(activeKb).then((r) => {
      setSamples(r.samples);
      setSource((prev) => {
        if (prev?.mode === "upload") return prev;
        if (prev?.mode === "sample" && r.samples.some((s) => (s.id || s.name) === prev.id)) return prev;
        // 默认选中第一个可完整演示的样例
        const first = r.samples.find((s) => s.pre_generated) || r.samples[0];
        return first ? { mode: "sample" as const, id: first.id || first.name, label: first.label } : prev;
      });
    }).catch(() => {});
  }, [kb, activeKb]);

  // 聚焦变化时复位代码页签
  useEffect(() => {
    setActiveFile(0);
    setShowHits(false);
    setShowFuncs(false);
  }, [focusId]);  const handleUpload = async (file: File) => {
    if (!activeKb) return;
    setUploading(true);
    setUploadErr("");
    try {
      const res = await api.upload(activeKb, file);
      setSource({ mode: "upload", id: res.upload_id, label: `${res.filename}（上传）` });
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  // 当前输入是否已有在跑任务（防止同输入重复提交；不同输入可并行）
  const dupRunning = !!source && Object.values(tasks).some(
    (t) => t.kb === activeKb && t.sourceId === source.id && isUnfinished(t.state.status),
  );

  const runGenerate = useCallback(async () => {
    if (!activeKb || !source) return;
    setSubmitErr("");
    try {
      const { task_id } = await api.generate(activeKb, source.id, source.mode);
      const tracked: TrackedTask = {
        id: task_id, kb: activeKb, sourceId: source.id, label: source.label,
        startedAt: Date.now(),
        state: { exists: true, status: "pending", progress: 0, message: "排队中" },
      };
      setTasks((prev) => ({ ...prev, [task_id]: tracked }));
      setFocusId(task_id);
    } catch (e) {
      setSubmitErr(e instanceof Error ? e.message : "提交失败");
    }
  }, [activeKb, source]);

  const focusTask = (id: string) => {
    const t = tasks[id];
    if (!t) return;
    setFocusId(id);
    if (t.kb === "operator" || t.kb === "paddle") {
      // 模型入口任务：聚焦场景卡并同步目标框架开关
      const tk = t.kb;
      if (kb !== "model") setKb("model");
      if (target !== tk) setTarget(tk);
    } else if (t.kb !== kb) {
      setKb(t.kb);
    }
  };

  const dismissTask = (id: string) => {
    setTasks((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    if (focusId === id) setFocusId(null);
  };

  const reset = () => {
    setKb(null);
    setSource(null);
    setFocusId(null);
  };

  const sortedTasks = Object.values(tasks).sort((a, b) => b.startedAt - a.startedAt);
  const focused = focusId ? tasks[focusId] : undefined;
  const focusedRunning = focused && isUnfinished(focused.state.status);
  const focusedResult = focused?.result;
  // 步骤条第 3/4 步语义：生成 = 有任务在跑或正在查看结果（步骤条保留已完成步骤点亮）；
  // 结果 = 正在查看结果。回到首页（无在跑任务且未聚焦结果）时 3、4 均不亮。
  const anyRunning = sortedTasks.some((t) => isUnfinished(t.state.status));
  const generated = anyRunning || !!focusedResult;
  // 阶段以后端显式上报的 stage 为准；stageIndex(progress) 仅作兜底，
  // 避免生成阶段（progress=30）被阈值误划入「检索知识库」
  const currentStage = focused
    ? (focused.state.stage ?? stageIndex(focused.state.progress))
    : 0;

  // 出结果时自动折叠输入区；聚焦运行中任务或无聚焦时自动展开
  useEffect(() => {
    if (focusedResult) setInputOpen(false);
    else setInputOpen(true);
  }, [focusedResult, focusId]);

  return (
    <div className="mx-auto w-full max-w-[1100px] space-y-5 px-6 py-6">
      {/* 步骤条（选择场景后固定于顶部） */}
      {kb && (
        <Stepper kb={kb} source={!!source} active={generated} hasResult={!!focusedResult} onReset={reset} />
      )}

      {/* 第一步：选择知识库 */}
      {!kb && (
        <section className="pt-8">
          <div className="mb-8">
            <h1 className="text-[26px] font-bold tracking-tight">选择任务场景 · 接入对应知识库</h1>
            <p className="mt-2 text-[14px] leading-relaxed text-muted-foreground">
              两类典型无人系统代码任务：星务协议处理注入星务协议知识库，
              无人机视觉模型部署/跨框架转换注入模型部署适配知识库（ONNX 与 Paddle 目标共用同一数据集）。
            </p>
            <Stepper kb={kb} source={!!source} active={generated} hasResult={!!focusedResult} className="mt-5" />
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            {(["protocol", "model"] as const).map((cid) => {
              const meta = KB_META[cid];
              const available = cid === "protocol"
                ? kbs.find((k) => k.id === "protocol")?.available ?? false
                : kbs.some((k) => k.id === "operator" && k.available);
              const Icon = meta.icon;
              return (
                <button
                  key={cid}
                  onClick={() => setKb(cid)}
                  disabled={!available}
                  className="top-beam glow-card group flex min-h-[240px] flex-col rounded-2xl border border-border bg-card p-8 text-left transition-all hover:border-primary/40 disabled:opacity-50"
                >
                  <div className="mb-5 flex items-center gap-3">
                    <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary transition-transform group-hover:scale-105">
                      <Icon size={22} strokeWidth={1.9} />
                    </span>
                    <div>
                      <div className="text-[17px] font-bold">{meta.title}</div>
                      <div className="mt-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                        {meta.subtitle}
                      </div>
                    </div>
                  </div>
                  <p className="flex-1 text-[13.5px] leading-relaxed text-muted-foreground">{meta.desc}</p>
                  {!available && (
                    <p className="mt-2 text-[11px] text-destructive">知识库数据缺失，请联系项目组</p>
                  )}
                </button>
              );
            })}
          </div>
        </section>
      )}

      {/* 任务条：所有任务（含后台运行中）的入口，随时点击切换查看 */}
      {sortedTasks.length > 0 && (
        <section className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">任务</span>
          {sortedTasks.map((t) => {
            const st = t.state.status;
            const activeChip = t.id === focusId;
            return (
              <button
                key={t.id}
                onClick={() => focusTask(t.id)}
                title={t.state.error || t.state.message}
                className={cn(
                  "group flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11.5px] transition-colors",
                  activeChip
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-card text-foreground/80 hover:border-primary/40",
                )}
              >
                {isUnfinished(st) ? (
                  <Loader2 size={12} className="animate-spin text-primary" />
                ) : st === "completed" ? (
                  <CheckCircle2 size={12} className="text-emerald-500" />
                ) : (
                  <XCircle size={12} className="text-destructive" />
                )}
                <span className="max-w-[180px] truncate font-mono">{t.label}</span>
                <span className="font-mono text-[9.5px] text-muted-foreground">
                  {KB_CHIP[t.kb] ?? t.kb}
                  {isUnfinished(st) ? ` · ${t.state.progress}%` : st === "completed" ? " · 完成" : " · 失败"}
                </span>
                {!isUnfinished(st) && (
                  <span
                    role="button"
                    tabIndex={-1}
                    onClick={(e) => { e.stopPropagation(); dismissTask(t.id); }}
                    className="ml-0.5 rounded p-0.5 text-muted-foreground/50 opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
                  >
                    <X size={11} />
                  </span>
                )}
              </button>
            );
          })}
        </section>
      )}

      {/* 第二步：输入（结果展示时折叠为摘要条，点击展开；在线生成期间仍可操作离线样例） */}
      {kb && !inputOpen && (
        <button
          onClick={() => setInputOpen(true)}
          className="flex w-full items-center gap-2.5 rounded-xl border border-border bg-card px-4 py-2.5 text-left transition-colors hover:border-primary/40"
        >
          <ChevronRight size={13} className="text-muted-foreground" />
          <span className="text-[12.5px] font-medium text-foreground/80">输入</span>
          <span className="font-mono text-[11.5px] text-muted-foreground">{source?.label || "未选择"}</span>
          <span className="ml-auto font-mono text-[10px] text-muted-foreground/70">点击展开修改</span>
        </button>
      )}
      {kb && inputOpen && (
        <section className="rounded-2xl border border-border bg-card p-5">
          <div className="mb-3 flex items-center gap-2">
            <FileText size={15} className="text-primary" />
            <h2 className="text-[15px] font-semibold">上传{kb === "protocol" ? "协议文档" : "模型源码"}</h2>
            <span className="font-mono text-[10.5px] text-muted-foreground">
              {kb === "protocol" ? "支持 .docx" : "支持 .py（PyTorch）"}
            </span>
          </div>

          {/* 目标框架（模型入口）：两个目标检索同一套模型部署适配知识库，仅生成目标不同 */}
          {kb === "model" && (
            <div className="mb-4">
              <div className="mb-1 text-[12px] font-medium text-muted-foreground">目标框架</div>
              <div className="flex max-w-[560px] gap-1 rounded-lg border border-border bg-muted p-1">
                {(Object.keys(TARGET_META) as (keyof typeof TARGET_META)[]).map((tg) => {
                  const tm = TARGET_META[tg];
                  const TIcon = tm.icon;
                  return (
                    <button
                      key={tg}
                      onClick={() => setTarget(tg)}
                      className={cn(
                        "flex flex-1 items-center gap-2 rounded-md px-3 py-2 text-left transition-colors",
                        target === tg ? "bg-card shadow-sm" : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      <TIcon size={14} className="shrink-0 text-primary" />
                      <div>
                        <div className="text-[12.5px] font-medium">{tm.title}</div>
                        <div className="mt-0.5 font-mono text-[9.5px] text-muted-foreground">{tm.sub}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* 预置样例 */}
          <div className="mb-1 text-[12px] font-medium text-muted-foreground">预置样例</div>
          <div className="mb-4 flex flex-wrap gap-2">
            {samples.map((s) => {
              const id = s.id || s.name;
              const active = source?.mode === "sample" && source.id === id;
              return (
                <button
                  key={id}
                  onClick={() => setSource({ mode: "sample", id, label: s.label })}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg border px-3 py-1.5 font-mono text-[12px] transition-colors",
                    active
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border bg-background text-foreground/80 hover:border-primary/40",
                  )}
                >
                  {s.name}
                  {s.pre_generated ? (
                    <span className="rounded bg-primary/15 px-1 py-0.5 text-[9px] text-primary">完整演示</span>
                  ) : null}
                </button>
              );
            })}
            {samples.length === 0 && <span className="text-[12px] text-muted-foreground">加载中…</span>}
          </div>

          {/* 上传 */}
          <div className="mb-1 text-[12px] font-medium text-muted-foreground">上传本地文件</div>
          <label
            className={cn(
              "flex cursor-pointer items-center justify-center gap-2 rounded-xl border-2 border-dashed border-border bg-background px-4 py-6 text-[12.5px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground",
              source?.mode === "upload" && "border-primary/50 bg-primary/5",
            )}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) handleUpload(f);
            }}
          >
            <Upload size={15} />
            {uploading
              ? "上传中…"
              : source?.mode === "upload"
                ? `已上传：${source.label}（点击可重新上传）`
                : `拖拽或点击上传 ${kb === "protocol" ? ".docx 协议文档" : ".py 模型源码"}`}
            <input
              type="file"
              className="hidden"
              accept={kb === "protocol" ? ".docx" : ".py"}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleUpload(f);
                e.target.value = "";
              }}
            />
          </label>
          {uploadErr && <p className="mt-2 text-[12px] text-destructive">{uploadErr}</p>}

          {/* 生成按钮 */}
          <div className="mt-5 flex items-center gap-3">
            <button
              onClick={runGenerate}
              disabled={!source || dupRunning}
              className="shadow-soft-md flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-[14px] font-medium text-primary-foreground transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {dupRunning ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
              {dupRunning ? "生成中…" : "开始生成"}
            </button>
            {source && (
              <span className="font-mono text-[11px] text-muted-foreground">
                当前输入：{source.label}
              </span>
            )}
          </div>
          {submitErr && <p className="mt-2 text-[12px] text-destructive">{submitErr}</p>}

          {/* 聚焦任务的进度面板（运行中 / 失败时展示；结果在下方单独区块） */}
          {focused && !focusedResult && (
            <div className="mt-5 rounded-xl border border-border bg-background p-4">
              <div className="mb-3 flex items-center gap-4">
                {STAGES.map((s, i) => {
                  const state = !focusedRunning
                    ? focused.state.status === "completed"
                      ? "done"
                      : i < currentStage
                        ? "done"
                        : i === currentStage
                          ? "active"
                          : "todo"
                    : i < currentStage
                      ? "done"
                      : i === currentStage
                        ? "active"
                        : "todo";
                  return (
                    <div key={s} className="flex items-center gap-1.5">
                      {state === "done" ? (
                        <CheckCircle2 size={14} className="text-emerald-500" />
                      ) : state === "active" ? (
                        <Loader2 size={14} className="animate-spin text-primary" />
                      ) : (
                        <span className="h-3.5 w-3.5 rounded-full border border-border" />
                      )}
                      <span
                        className={cn(
                          "text-[12px]",
                          state === "active" ? "font-medium text-primary" : state === "done" ? "text-emerald-600" : "text-muted-foreground/60",
                        )}
                      >
                        {s}
                      </span>
                      {i < STAGES.length - 1 && <span className="ml-1 text-border">—</span>}
                    </div>
                  );
                })}
                <span className="ml-auto max-w-[200px] truncate font-mono text-[10.5px] text-muted-foreground">
                  {focused.label}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-700"
                  style={{ width: `${focused.state.progress}%` }}
                />
              </div>
              <div className="mt-2 flex items-center justify-between font-mono text-[10.5px] text-muted-foreground">
                <span>{focused.state.message}</span>
                <span>{focused.state.progress}%</span>
              </div>
              {focused.state.status === "failed" && (
                <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-[12px] text-destructive">
                  {focused.state.error || focused.state.message}
                </div>
              )}
              {focused.state.status === "completed" && !focused.result && (
                <div className="mt-3 flex items-center gap-2 text-[12px] text-muted-foreground">
                  <Loader2 size={13} className="animate-spin" /> 结果载入中…
                </div>
              )}
              {focused.state.intermediate && <IntermediateView data={focused.state.intermediate} />}
            </div>
          )}
        </section>
      )}

      {/* 第三步：结果与评测（聚焦已完成任务时展示；不影响上方继续发起新任务） */}
      {focusedResult && focused && (
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-[17px] font-bold tracking-tight">
              生成结果 · <span className="font-mono text-primary">{focusedResult.target}</span>
            </h2>
            {focusedResult.pre_generated && (
              <span className="rounded-full border border-primary/30 bg-primary/5 px-2.5 py-0.5 font-mono text-[10px] text-primary">
                预生成结果 · 真实流水线历史产物（上传自有文件可现场生成）
              </span>
            )}
            <button onClick={() => setFocusId(null)} className="ml-auto flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary">
              <RotateCcw size={11} /> 收起结果，继续其他输入
            </button>
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            {/* 代码区 */}
            <div className="lg:col-span-3">
              <div className="overflow-hidden rounded-xl border border-border bg-card shadow-soft-sm">
                <div className="flex items-center gap-1 border-b border-border bg-muted/50 px-3 py-2">
                  {focusedResult.files.map((f, i) => (
                    <button
                      key={f.name}
                      onClick={() => setActiveFile(i)}
                      className={cn(
                        "flex items-center gap-1.5 rounded-md px-2.5 py-1 font-mono text-[11px] transition-colors",
                        activeFile === i ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      <FileCode2 size={12} />
                      {f.name}
                    </button>
                  ))}
                </div>
                <pre className="code-scroll max-h-[560px] overflow-auto p-4 font-mono text-[11.5px] leading-relaxed">
                  <code>{focusedResult.files[activeFile]?.content}</code>
                </pre>
              </div>
            </div>

            {/* 评测区 */}
            <div className="lg:col-span-2">
              {/* ONNX 模型文件产物卡（真实执行导出的 .onnx，可下载） */}
              {focusedResult.onnx_model && (
                <div className="top-beam mb-4 rounded-xl border border-border bg-card p-4">
                  <div className="mb-3 flex items-center gap-2">
                    <Package size={15} className="text-primary" />
                    <h3 className="text-[14px] font-semibold">ONNX 模型文件</h3>
                    <a
                      href={api.onnxDownloadUrl(focused.id)}
                      download={focusedResult.onnx_model.name}
                      className="shadow-soft-md ml-auto flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-[12px] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      <Download size={13} /> 下载 .onnx
                    </a>
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
                    <span>文件：<span className="text-foreground">{focusedResult.onnx_model.name}</span></span>
                    <span>大小：<span className="font-bold text-foreground">{focusedResult.onnx_model.size_mb} MB</span></span>
                    <span>算子节点：<span className="text-foreground">{focusedResult.onnx_model.nodes}</span></span>
                    {focusedResult.onnx_model.params_m != null && (
                      <span>参数量：<span className="text-foreground">{focusedResult.onnx_model.params_m} M</span></span>
                    )}
                    <span>
                      opset：<span className="text-foreground">
                        {focusedResult.onnx_model.opset.find((o) => o.domain === "ai.onnx")?.version ?? "—"}
                      </span>
                    </span>
                    <span>
                      校验：
                      <span className={focusedResult.onnx_model.checker ? "text-emerald-600" : "text-destructive"}>
                        {focusedResult.onnx_model.checker ? "onnx.checker 通过" : "未通过"}
                      </span>
                    </span>
                  </div>
                  <div className="mt-2 space-y-0.5 border-t border-border/60 pt-2 font-mono text-[10px] text-muted-foreground">
                    {focusedResult.onnx_model.inputs.map((io) => (
                      <div key={`in-${io.name}`}>输入 <span className="text-foreground">{io.name}</span> [{io.shape.join(", ")}]</div>
                    ))}
                    {focusedResult.onnx_model.outputs.map((io) => (
                      <div key={`out-${io.name}`}>输出 <span className="text-foreground">{io.name}</span> [{io.shape.join(", ")}]</div>
                    ))}
                    {(focusedResult.onnx_model.op_counts?.length ?? 0) > 0 && (
                      <div className="flex flex-wrap items-center gap-1 pt-1.5">
                        <span>算子分布</span>
                        {focusedResult.onnx_model.op_counts!.map(([op, c]) => (
                          <span key={op} className="rounded bg-muted px-1.5 py-0.5 text-[9.5px] text-foreground/75">
                            {op} ×{c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
              <div className="top-beam rounded-xl border border-border bg-card p-4">
                <h3 className="mb-3 text-[14px] font-semibold">自动评测</h3>
                {focusedResult.evaluation ? (
                  <div className="space-y-3">
                    {/* 通过/失败检查项（不打分） */}
                    {focusedResult.evaluation.checks && Object.keys(focusedResult.evaluation.checks).length > 0 && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2">
                        {Object.entries(focusedResult.evaluation.checks).map(([name, ok]) => (
                          <div key={name} className="flex items-center gap-1.5 py-0.5 text-[11.5px]">
                            {ok ? (
                              <CheckCircle2 size={12} className="shrink-0 text-emerald-500" />
                            ) : (
                              <XCircle size={12} className="shrink-0 text-destructive" />
                            )}
                            <span className={ok ? "text-foreground/85" : "text-destructive"}>{name}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* 连续程度分（相似度/覆盖率，进度条） */}
                    {Object.entries(focusedResult.evaluation.scores).map(([name, v]) => (
                      <div key={name}>
                        <div className="mb-1 flex items-baseline justify-between">
                          <span className="text-[12px] text-muted-foreground">{name}</span>
                          <span className="font-mono text-[12.5px] font-bold text-primary">{v}</span>
                        </div>
                        {v <= 100 && (
                          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                              style={{ width: `${Math.min(v, 100)}%` }}
                            />
                          </div>
                        )}
                      </div>
                    ))}

                    {focusedResult.evaluation.stats && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                        {Object.entries(focusedResult.evaluation.stats).map(([name, v]) => (
                          <div key={name} className="flex items-center justify-between py-0.5">
                            <span>{name}</span>
                            <span className="font-bold text-foreground">{v}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {focusedResult.evaluation.functions && focusedResult.evaluation.kind !== "full" && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                        函数匹配：{focusedResult.evaluation.functions["成功匹配"]} / {focusedResult.evaluation.functions["真实函数"]} 真实函数
                        （生成 {focusedResult.evaluation.functions["生成函数"]} 个）
                      </div>
                    )}

                    {/* 协议 GT 评测：函数匹配明细（AST 相似度 × 匈牙利算法一对一分配） */}
                    {focusedResult.evaluation.kind === "full" && focusedResult.evaluation.function_matches && (
                      <div>
                        <button
                          onClick={() => setShowFuncs((v) => !v)}
                          className="flex items-center gap-1 text-[11.5px] font-medium text-primary"
                        >
                          函数匹配明细（{focusedResult.evaluation.function_matches.length} 对）
                          {showFuncs ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                        {focusedResult.evaluation.match_note && (
                          <p className="mt-1 text-[10.5px] leading-relaxed text-muted-foreground">
                            {focusedResult.evaluation.match_note}
                          </p>
                        )}
                        {focusedResult.evaluation.functions && (
                          <div className="mt-1.5 flex flex-wrap gap-1.5 font-mono text-[10px]">
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">
                              成功匹配 {focusedResult.evaluation.functions["成功匹配"]}/{focusedResult.evaluation.functions["真实函数"]}
                            </span>
                            <span className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">
                              其中同名 {focusedResult.evaluation.functions["其中同名"]}
                            </span>
                            <span className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">
                              生成 {focusedResult.evaluation.functions["生成函数"]} 个
                            </span>
                            {(focusedResult.evaluation.functions["未覆盖真实函数"] ?? 0) > 0 && (
                              <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-destructive">
                                未覆盖 {focusedResult.evaluation.functions["未覆盖真实函数"]}
                              </span>
                            )}
                            {(focusedResult.evaluation.functions["多生成函数"] ?? 0) > 0 && (
                              <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-amber-600">
                                多生成 {focusedResult.evaluation.functions["多生成函数"]}
                              </span>
                            )}
                          </div>
                        )}
                        {showFuncs && (
                          <div className="code-scroll mt-2 max-h-[260px] overflow-auto rounded-lg bg-muted/50 p-2">
                            <table className="w-full font-mono text-[10px]">
                              <thead>
                                <tr className="text-left text-muted-foreground">
                                  <th className="pb-1 pr-2 font-medium">真实函数</th>
                                  <th className="pb-1 pr-2 font-medium">匹配到的生成函数</th>
                                  <th className="pb-1 pr-2 font-medium">AST 相似度</th>
                                  <th className="pb-1 pr-2 font-medium">同名</th>
                                  <th className="pb-1 font-medium">状态</th>
                                </tr>
                              </thead>
                              <tbody>
                                {focusedResult.evaluation.function_matches.map((m, i) => (
                                  <tr key={i} className="border-t border-border/50">
                                    <td className="py-1 pr-2 text-foreground/85">{m.gt}</td>
                                    <td className="py-1 pr-2 text-foreground/85">{m.gen ?? "—"}</td>
                                    <td className="py-1 pr-2 text-muted-foreground">
                                      <span className="font-bold text-foreground">{m.ast}%</span>（树 {m.tree_edit} / 语义 {m.semantic}）
                                    </td>
                                    <td className="py-1 pr-2">{m.name_match ? "✓" : "✗"}</td>
                                    <td className="py-1">
                                      {m.matched
                                        ? <span className="text-emerald-600">匹配</span>
                                        : <span className="text-destructive">未匹配</span>}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {(focusedResult.evaluation.unmatched_gt_functions?.length ?? 0) > 0 && (
                              <div className="mt-2 border-t border-border/50 pt-2">
                                <span className="text-[10px] text-muted-foreground">未覆盖真实函数：</span>
                                {focusedResult.evaluation.unmatched_gt_functions!.map((f) => (
                                  <span key={f} className="ml-1 rounded bg-destructive/10 px-1.5 py-0.5 font-mono text-[9.5px] text-destructive">{f}</span>
                                ))}
                              </div>
                            )}
                            {(focusedResult.evaluation.unmatched_gen_functions?.length ?? 0) > 0 && (
                              <div className="mt-1.5">
                                <span className="text-[10px] text-muted-foreground">多生成函数：</span>
                                {focusedResult.evaluation.unmatched_gen_functions!.map((f) => (
                                  <span key={f} className="ml-1 rounded bg-amber-500/10 px-1.5 py-0.5 font-mono text-[9.5px] text-amber-600">{f}</span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* 协议 GT 评测：Config 各维度问题明细 */}
                    {focusedResult.evaluation.kind === "full" && focusedResult.evaluation.issues &&
                      Object.values(focusedResult.evaluation.issues).some((l) => l.length > 0) && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2">
                        <div className="mb-1 text-[11px] text-muted-foreground">问题明细（Config 各维度，可直接指导修正）</div>
                        {Object.entries(focusedResult.evaluation.issues)
                          .filter(([, l]) => l.length > 0)
                          .map(([dim, list]) => (
                            <div key={dim} className="py-0.5 font-mono text-[10.5px] leading-relaxed">
                              <span className="text-muted-foreground">{dim}：</span>
                              <span className="text-foreground/85">{list.join("；")}</span>
                            </div>
                          ))}
                      </div>
                    )}

                    {focusedResult.evaluation.kind === "operator" && (
                      <div>
                        <div className="mb-1.5 text-[11px] text-muted-foreground">
                          输入算子 {focusedResult.evaluation.total_calls} 个 · 检索注入知识 {focusedResult.evaluation.retrieved_count} 条
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {focusedResult.evaluation.dynamic_axes && (
                            <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
                              动态 batch 声明 ✓
                            </span>
                          )}
                          {(focusedResult.evaluation.custom_symbolics ?? 0) > 0 && (
                            <span className="rounded bg-indigo-500/10 px-1.5 py-0.5 font-mono text-[10px] text-indigo-500">
                              自定义 symbolic ×{focusedResult.evaluation.custom_symbolics}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {focusedResult.evaluation.kind === "paddle" && (                      <div>
                        <div className="mb-1.5 text-[11px] text-muted-foreground">
                          输入算子 {focusedResult.evaluation.total_calls} 个 · 检索注入知识 {focusedResult.evaluation.retrieved_count} 条
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {(focusedResult.evaluation.custom_wrappers ?? 0) > 0 && (
                            <span className="rounded bg-indigo-500/10 px-1.5 py-0.5 font-mono text-[10px] text-indigo-500">
                              自定义对齐层 ×{focusedResult.evaluation.custom_wrappers}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {focusedResult.evaluation.line_similarity_vs_reference != null && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] text-muted-foreground">
                        与参考产物行级一致率：{focusedResult.evaluation.line_similarity_vs_reference}%
                      </div>
                    )}

                    {focusedResult.evaluation.note && (
                      <p className="text-[11px] leading-relaxed text-muted-foreground">{focusedResult.evaluation.note}</p>
                    )}

                    {focusedResult.evaluation.retrieved && focusedResult.evaluation.retrieved.length > 0 && (
                      <div>
                        <button
                          onClick={() => setShowHits((v) => !v)}
                          className="flex items-center gap-1 text-[11.5px] font-medium text-primary"
                        >
                          检索注入的知识（{focusedResult.evaluation.retrieved.length}）
                          {showHits ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                        {showHits && (
                          <div className="code-scroll mt-2 max-h-[220px] space-y-1 overflow-auto">
                            {focusedResult.evaluation.retrieved.map((r, i) => (
                              <div key={i} className="rounded bg-muted/40 px-2 py-1.5">
                                <div className="flex items-center gap-2">
                                  <span className="min-w-0 flex-1 truncate font-mono text-[10px]">{r.title}</span>
                                  <span className={cn("shrink-0 rounded px-1 py-0.5 font-mono text-[9px]", CATEGORY_STYLE[r.category_label] || "bg-muted")}>{r.category_label}</span>
                                </div>
                                <div className="mt-0.5 truncate font-mono text-[9px] text-muted-foreground/60">
                                  {r.source} · 相关度 {r.score}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-[12px] text-muted-foreground">无评测数据</p>
                )}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
