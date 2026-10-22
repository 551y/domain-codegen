"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  FileCode2,
  FileText,
  Loader2,
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
        className="code-scroll max-h-[180px] overflow-auto whitespace-pre-wrap break-all rounded-lg bg-muted/70 p-2.5 font-mono text-[10.5px] leading-relaxed text-foreground/90"
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
    data.live || data.live_config || data.live_handler;
  if (!hasAny) return null;
  return (
    <div className="mt-3 space-y-2.5 rounded-lg border border-border/70 bg-card/70 p-3">
      <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        中间过程输出
      </div>

      {data.live && <LiveBox live={data.live} />}
      {data.live_config && <LiveBox live={data.live_config} />}
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
            检索知识库：命中 <span className="font-bold text-foreground">{data.retrieved.length}</span> 条知识（BM25 离线检索，将注入生成上下文）
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

  useEffect(() => {
    if (!kb) return;
    setSamples([]);
    setSource(null);
    setFocusId(null);
    api.samples(kb).then((r) => {
      setSamples(r.samples);
      // 默认选中第一个可完整演示的样例
      const first = r.samples.find((s) => s.pre_generated) || r.samples[0];
      if (first) {
        setSource({ mode: "sample", id: first.id || first.name, label: first.label });
      }
    }).catch(() => {});
  }, [kb]);

  // 聚焦变化时复位代码页签
  useEffect(() => {
    setActiveFile(0);
    setShowHits(false);
  }, [focusId]);  const handleUpload = async (file: File) => {
    if (!kb) return;
    setUploading(true);
    setUploadErr("");
    try {
      const res = await api.upload(kb, file);
      setSource({ mode: "upload", id: res.upload_id, label: `${res.filename}（上传）` });
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  // 当前输入是否已有在跑任务（防止同输入重复提交；不同输入可并行）
  const dupRunning = !!source && Object.values(tasks).some(
    (t) => t.kb === kb && t.sourceId === source.id && isUnfinished(t.state.status),
  );

  const runGenerate = useCallback(async () => {
    if (!kb || !source) return;
    setSubmitErr("");
    try {
      const { task_id } = await api.generate(kb, source.id, source.mode);
      const tracked: TrackedTask = {
        id: task_id, kb, sourceId: source.id, label: source.label,
        startedAt: Date.now(),
        state: { exists: true, status: "pending", progress: 0, message: "排队中" },
      };
      setTasks((prev) => ({ ...prev, [task_id]: tracked }));
      setFocusId(task_id);
    } catch (e) {
      setSubmitErr(e instanceof Error ? e.message : "提交失败");
    }
  }, [kb, source]);

  const focusTask = (id: string) => {
    const t = tasks[id];
    if (!t) return;
    setFocusId(id);
    if (t.kb !== kb) setKb(t.kb);
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
  const anyResult = sortedTasks.some((t) => t.result);
  const currentStage = focused ? stageIndex(focused.state.progress) : 0;

  // 出结果时自动折叠输入区；聚焦运行中任务或无聚焦时自动展开
  useEffect(() => {
    if (focusedResult) setInputOpen(false);
    else setInputOpen(true);
  }, [focusedResult, focusId]);

  return (
    <div className="mx-auto w-full max-w-[1100px] space-y-5 px-6 py-6">
      {/* 步骤条（选择场景后固定于顶部） */}
      {kb && (
        <Stepper kb={kb} source={!!source} active={sortedTasks.length > 0} hasResult={anyResult} onReset={reset} />
      )}

      {/* 第一步：选择知识库 */}
      {!kb && (
        <section className="pt-8">
          <div className="mb-8">
            <h1 className="text-[26px] font-bold tracking-tight">选择任务场景 · 接入对应知识库</h1>
            <p className="mt-2 text-[14px] leading-relaxed text-muted-foreground">
              两类典型无人系统代码任务：星务协议处理注入协议知识库，
              模型ONNX部署注入 ONNX 算子知识库。
            </p>
            <Stepper kb={kb} source={!!source} active={sortedTasks.length > 0} hasResult={anyResult} className="mt-5" />
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            {kbs.map((k) => {
              const isProtocol = k.id === "protocol";
              const Icon = isProtocol ? Satellite : Plane;
              return (
                <button
                  key={k.id}
                  onClick={() => setKb(k.id)}
                  disabled={!k.available}
                  className="top-beam glow-card group flex min-h-[240px] flex-col rounded-2xl border border-border bg-card p-8 text-left transition-all hover:border-primary/40 disabled:opacity-50"
                >
                  <div className="mb-5 flex items-center gap-3">
                    <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary transition-transform group-hover:scale-105">
                      <Icon size={22} strokeWidth={1.9} />
                    </span>
                    <div>
                      <div className="text-[17px] font-bold">
                        {isProtocol ? "卫星协议代码生成" : "无人机模型部署适配"}
                      </div>
                      <div className="mt-0.5 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                        {isProtocol ? "Protocol Handler Generation" : "PyTorch → ONNX Export"}
                      </div>
                    </div>
                  </div>
                  <p className="flex-1 text-[13.5px] leading-relaxed text-muted-foreground">
                    {isProtocol
                      ? "上传星务协议文档（.docx），检索增强生成遥测解析、遥控组包、内部总线交互等协议处理代码（config.json + handler.c），并对生成结果自动评测。"
                      : "上传 PyTorch 模型源码（.py），检索 ONNX 算子知识库后由 LLM 生成 ONNX 导出代码（torch.onnx.export + 自定义 symbolic），并对生成结果自动评测。"}
                  </p>
                  {!k.available && (
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
                  {t.kb === "protocol" ? "协议" : "算子"}
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
          <div className="mb-1 text-[12px] font-medium text-muted-foreground">或上传自己的文件</div>
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
              <div className="top-beam rounded-xl border border-border bg-card p-4">
                <h3 className="mb-3 text-[14px] font-semibold">自动评测</h3>
                {focusedResult.evaluation ? (
                  <div className="space-y-3">
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

                    {focusedResult.evaluation.functions && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                        函数匹配：{focusedResult.evaluation.functions["成功匹配"]} / {focusedResult.evaluation.functions["真实函数"]} 真实函数
                        （生成 {focusedResult.evaluation.functions["生成函数"]} 个）
                      </div>
                    )}

                    {focusedResult.evaluation.kind === "operator" && (
                      <div>
                        <div className="mb-1.5 text-[11px] text-muted-foreground">
                          输入算子 {focusedResult.evaluation.total_calls} 个 · 检索注入知识 {focusedResult.evaluation.retrieved_count} 条
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {focusedResult.evaluation.has_export_call && (
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] text-primary">
                              torch.onnx.export ✓
                            </span>
                          )}
                          {focusedResult.evaluation.dynamic_axes && (
                            <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
                              动态 batch ✓
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
