"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileCode2,
  FileText,
  Loader2,
  Plane,
  Play,
  RotateCcw,
  Satellite,
  Upload,
} from "lucide-react";
import { api } from "@/lib/api";
import type { GenerateResult, KbCard, SampleItem, TaskState } from "@/lib/api";
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

// ---------- 主页面 ----------

export default function WorkbenchPage() {
  const [kbs, setKbs] = useState<KbCard[]>([]);
  const [kb, setKb] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [source, setSource] = useState<{ mode: "sample" | "upload"; id: string; label: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState("");
  const [task, setTask] = useState<TaskState | null>(null);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [activeFile, setActiveFile] = useState(0);
  const [showHits, setShowHits] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.kbs().then((r) => setKbs(r.kbs)).catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    if (!kb) return;
    setSamples([]);
    setSource(null);
    setResult(null);
    setTask(null);
    api.samples(kb).then((r) => {
      setSamples(r.samples);
      // 默认选中第一个可完整演示的样例
      const first = r.samples.find((s) => s.pre_generated) || r.samples[0];
      if (first) {
        setSource({ mode: "sample", id: first.id || first.name, label: first.label });
      }
    }).catch(() => {});
  }, [kb]);

  const handleUpload = async (file: File) => {
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

  const runGenerate = useCallback(async () => {
    if (!kb || !source) return;
    setResult(null);
    setActiveFile(0);
    try {
      const { task_id } = await api.generate(kb, source.id, source.mode);
      const poll = async () => {
        try {
          const t = await api.task(task_id);
          setTask(t);
          if (t.status === "completed") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            const r = await api.result(task_id);
            setResult(r);
          } else if (t.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } catch { /* 单次轮询失败忽略 */ }
      };
      poll();
      pollRef.current = setInterval(poll, 1500);
    } catch (e) {
      setTask({ exists: true, status: "failed", progress: 0, message: "提交失败", error: e instanceof Error ? e.message : "未知错误" });
    }
  }, [kb, source]);

  const reset = () => {
    setKb(null);
    setSource(null);
    setTask(null);
    setResult(null);
  };

  const running = task && (task.status === "running" || task.status === "pending");
  const currentStage = task ? stageIndex(task.progress) : 0;

  return (
    <div className="mx-auto w-full max-w-[1100px] space-y-5 px-6 py-6">
      {/* 步骤条（选择场景后固定于顶部） */}
      {kb && (
        <Stepper kb={kb} source={!!source} active={!!(running || result)} hasResult={!!result} onReset={reset} />
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
            <Stepper kb={kb} source={!!source} active={!!(running || result)} hasResult={!!result} className="mt-5" />
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

      {/* 第二步：输入 */}
      {kb && !result && (
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
              disabled={!source || !!running}
              className="shadow-soft-md flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-[14px] font-medium text-primary-foreground transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {running ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
              {running ? "生成中…" : "开始生成"}
            </button>
            {source && (
              <span className="font-mono text-[11px] text-muted-foreground">
                当前输入：{source.label}
              </span>
            )}
          </div>

          {/* 进度面板 */}
          {task && (
            <div className="mt-5 rounded-xl border border-border bg-background p-4">
              <div className="mb-3 flex items-center gap-4">
                {STAGES.map((s, i) => {
                  const state = !running
                    ? task.status === "completed"
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
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-700"
                  style={{ width: `${task.progress}%` }}
                />
              </div>
              <div className="mt-2 flex items-center justify-between font-mono text-[10.5px] text-muted-foreground">
                <span>{task.message}</span>
                <span>{task.progress}%</span>
              </div>
              {task.status === "failed" && (
                <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-[12px] text-destructive">
                  {task.error || task.message}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* 第三步：结果与评测 */}
      {result && (
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-[17px] font-bold tracking-tight">
              生成结果 · <span className="font-mono text-primary">{result.target}</span>
            </h2>
            {result.pre_generated && (
              <span className="rounded-full border border-primary/30 bg-primary/5 px-2.5 py-0.5 font-mono text-[10px] text-primary">
                预生成结果 · 真实流水线历史产物（上传自有文件可现场生成）
              </span>
            )}
            <button onClick={() => setResult(null)} className="ml-auto flex items-center gap-1 rounded-md border border-border px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary">
              <RotateCcw size={11} /> 继续生成其他输入
            </button>
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            {/* 代码区 */}
            <div className="lg:col-span-3">
              <div className="overflow-hidden rounded-xl border border-border bg-card shadow-soft-sm">
                <div className="flex items-center gap-1 border-b border-border bg-muted/50 px-3 py-2">
                  {result.files.map((f, i) => (
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
                  <code>{result.files[activeFile]?.content}</code>
                </pre>
              </div>
            </div>

            {/* 评测区 */}
            <div className="lg:col-span-2">
              <div className="top-beam rounded-xl border border-border bg-card p-4">
                <h3 className="mb-3 text-[14px] font-semibold">自动评测</h3>
                {result.evaluation ? (
                  <div className="space-y-3">
                    {Object.entries(result.evaluation.scores).map(([name, v]) => (
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

                    {result.evaluation.stats && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                        {Object.entries(result.evaluation.stats).map(([name, v]) => (
                          <div key={name} className="flex items-center justify-between py-0.5">
                            <span>{name}</span>
                            <span className="font-bold text-foreground">{v}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {result.evaluation.functions && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
                        函数匹配：{result.evaluation.functions["成功匹配"]} / {result.evaluation.functions["真实函数"]} 真实函数
                        （生成 {result.evaluation.functions["生成函数"]} 个）
                      </div>
                    )}

                    {result.evaluation.kind === "operator" && (
                      <div>
                        <div className="mb-1.5 text-[11px] text-muted-foreground">
                          输入算子 {result.evaluation.total_calls} 个 · 检索注入知识 {result.evaluation.retrieved_count} 条
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {result.evaluation.has_export_call && (
                            <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] text-primary">
                              torch.onnx.export ✓
                            </span>
                          )}
                          {result.evaluation.dynamic_axes && (
                            <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
                              动态 batch ✓
                            </span>
                          )}
                          {(result.evaluation.custom_symbolics ?? 0) > 0 && (
                            <span className="rounded bg-indigo-500/10 px-1.5 py-0.5 font-mono text-[10px] text-indigo-500">
                              自定义 symbolic ×{result.evaluation.custom_symbolics}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {result.evaluation.line_similarity_vs_reference != null && (
                      <div className="rounded-lg bg-muted/50 px-3 py-2 font-mono text-[10.5px] text-muted-foreground">
                        与参考产物行级一致率：{result.evaluation.line_similarity_vs_reference}%
                      </div>
                    )}

                    {result.evaluation.note && (
                      <p className="text-[11px] leading-relaxed text-muted-foreground">{result.evaluation.note}</p>
                    )}

                    {result.evaluation.retrieved && result.evaluation.retrieved.length > 0 && (
                      <div>
                        <button
                          onClick={() => setShowHits((v) => !v)}
                          className="flex items-center gap-1 text-[11.5px] font-medium text-primary"
                        >
                          检索注入的知识（{result.evaluation.retrieved.length}）
                          {showHits ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                        {showHits && (
                          <div className="code-scroll mt-2 max-h-[220px] space-y-1 overflow-auto">
                            {result.evaluation.retrieved.map((r, i) => (
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
