/**
 * domain-codegen 编排后端 API 客户端
 */

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      detail = JSON.parse(text).detail || text;
    } catch { /* 保留原文 */ }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

// ---------- 类型 ----------

export interface KbCard {
  id: string;
  available: boolean;
  // protocol
  bases?: { source: string; chunks: number }[];
  total_chunks?: number;
  form?: string;
  // operator
  total_rules?: number;
  category_counts?: Record<string, number>;
}

export interface SampleItem {
  name: string;
  label: string;
  id?: string;
  has_gt?: boolean;
  pre_generated?: boolean;
  domain?: string;
  modelx_variant?: string;
  demo_label?: string;
  demo_available?: boolean;
}

export interface RetrievedItem {
  title: string;
  category_label: string;
  score: number;
  source: string;
}

export interface LiveOutput {
  stage: string;
  text: string;
  done?: boolean;
}

export interface TaskIntermediate {
  parsed?: Record<string, string>;
  operators?: { total_calls: number; leaf_ops: string[] };
  retrieved?: RetrievedItem[];
  retrieval?: Record<string, number | string>;
  config_stats?: Record<string, number>;
  conversion_stats?: { success: number; failed: number; rate: number };
  evaluation_steps?: { name: string; status: "pending" | "running" | "passed" | "failed" | "skipped"; detail?: string }[];
  gen_mode?: string;
  live?: LiveOutput;
  live_config?: LiveOutput;
  live_handler_plan?: LiveOutput;
  live_handler?: LiveOutput;
}

export interface TaskState {
  exists: boolean;
  status: string;
  progress: number;
  /** 后端显式上报的阶段序号：0 解析输入 / 1 检索知识库 / 2 生成目标代码 / 3 评测结果 */
  stage?: number;
  message: string;
  error?: string | null;
  intermediate?: TaskIntermediate;
}

export interface ResultFile {
  name: string;
  language: string;
  content: string;
}

export interface Evaluation {
  kind: string;
  /** 连续程度分（相似度/覆盖率，进度条展示） */
  scores: Record<string, number>;
  /** 通过/失败型检查项（✓/✗ 清单展示，不打分） */
  checks?: Record<string, boolean>;
  stats?: Record<string, number | string>;
  functions?: Record<string, number>;
  syntax_valid?: boolean;
  has_export_call?: boolean;
  has_checker?: boolean;
  dynamic_axes?: boolean;
  custom_symbolics?: number;
  // ONNX 真实执行导出
  export_attempted?: boolean;
  export_executed?: boolean;
  onnx_checker?: boolean;
  /** onnxruntime 推理冒烟测试结果（装了 ORT 才有） */
  smoke_test?: { ok: boolean; error?: string; output_shape?: (number | string)[]; dynamic_batch_dim?: boolean; dynamic_batch_ok?: boolean } | null;
  /** 已生成配套的 ONNX Runtime C++ 推理调用源码 */
  cpp_inference?: boolean;
  // paddle 转换评测
  uses_paddle?: boolean;
  torch_residue?: boolean;
  custom_wrappers?: number;
  paddle_runtime?: {
    attempted: boolean;
    available?: boolean;
    import_ok?: boolean;
    construction_attempted?: boolean;
    construction_ok?: boolean;
    forward_attempted?: boolean;
    forward_ok?: boolean;
    class_name?: string;
    output_shape?: (number | string)[];
    input_shape?: number[];
    params?: number;
    profile_label?: string;
    finite?: boolean;
    output_shape_ok?: boolean;
    elapsed_s?: number;
    error?: string;
    skip_reason?: string;
  };
  unsupported_apis?: Record<string, number>;
  library_modifications?: { file: string; level: string; classes: string[] }[];
  compatibility_fixes?: string[];
  total_calls?: number;
  retrieved_count?: number;
  retrieved?: RetrievedItem[];
  // 协议 ground truth 评测明细
  issues?: Record<string, string[]>;
  function_matches?: { gt: string; gen: string | null; matched: boolean; name_match: boolean; ast: number; tree_edit: number; semantic: number }[];
  unmatched_gt_functions?: string[];
  unmatched_gen_functions?: string[];
  match_note?: string;
  line_similarity_vs_reference?: number | null;
  note?: string;
}

export interface OnnxArtifact {
  name: string;
  size_bytes: number;
  size_mb: number;
  nodes: number;
  opset: { domain: string; version: number }[];
  inputs: { name: string; shape: (number | string)[] }[];
  outputs: { name: string; shape: (number | string)[] }[];
  checker: boolean;
  params?: number;
  params_m?: number;
  /** 算子类型分布（按数量降序，前 8） */
  op_counts?: [string, number][];
}

export interface GenerateResult {
  kb: string;
  target: string;
  files: ResultFile[];
  /** 真实执行导出脚本产出的 ONNX 模型文件（无 torch 环境或导出失败时为 null） */
  onnx_model?: OnnxArtifact | null;
  evaluation: Evaluation | null;
  pre_generated: boolean;
}

export interface ProtocolChunk {
  chunk_id: string;
  kind: string;
  file: string;
  token_count: number;
  text_preview: string;
}

// 星务协议知识库 · 基本数据集可视化（356 条原始基线为主，3090 增强为对照）
export interface ProtocolDatasetRecord {
  id: string;
  title: string;
  category: string;
  category_label: string;
  source: string;
  file: string;
  text: string;
  char_count: number;
  token_count: number;
  chunk_index: number;
}

export interface ProtocolDataset {
  baseline: { total: number; protocols: Record<string, number>; kinds: Record<string, number>; mean_chars: number; chunk: string; vectorized: string };
  augmented: { total: number; protocols: Record<string, number>; kinds: Record<string, number>; mean_chars: number; chunk: string; vectorized: string; ratio: number };
  files: { path: string; lines: number; note: string }[];
  records: ProtocolDatasetRecord[];
}

// 模型部署适配知识库 · 数据集可视化（1,349 条交付记录为主，基线/增量/筛选为对照）
export interface OnnxDatasetRecord {
  id: string;
  title: string;
  category: string;
  category_label: string;
  source: string;
  text: string;
  framework_version: string;
  verification_status: string;
}

export interface OnnxDataset {
  total: number;
  categories: { key: string; label: string; count: number }[];
  composition: { baseline: number; additions: number };
  files: { path: string; lines: number; note: string }[];
  records: OnnxDatasetRecord[];
}

export interface PaddleDatasetRecord {
  id: string;
  title: string;
  category: string;
  category_label: string;
  source: string;
  text: string;
  source_api: string;
  target_api: string;
  matcher: string;
  classification?: number | null;
  params_change: Record<string, string>;
  framework_version: string;
  verification_status: string;
}

export interface PaddleDataset {
  total: number;
  categories: { key: string; label: string; count: number }[];
  composition: {
    direct: number;
    parameter_rewrites: number;
    semantic_rewrites: number;
    unsupported: number;
  };
  files: { path: string; lines: number; note: string }[];
  records: PaddleDatasetRecord[];
}

export interface OnnxRecord {
  id: string;
  title: string;
  category: string;
  category_label: string;
  source: string;
  preview: string;
}

export interface SettingsState {
  llm_available: boolean;
  chat_model: string;
}

// ---------- 接口 ----------

export const api = {
  health: () => request<{ status: string }>("/health"),

  kbs: () => request<{ kbs: KbCard[] }>("/kbs"),

  samples: (kb: string) => request<{ samples: SampleItem[] }>(`/samples?kb=${kb}`),

  upload: async (kb: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ upload_id: string; filename: string; size: number }>(
      `/upload?kb=${kb}`,
      { method: "POST", body: form },
    );
  },

  generate: (kb: string, source: string, mode: "sample" | "upload") =>
    request<{ task_id: string }>("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kb, source, mode }),
    }),

  task: (taskId: string) => request<TaskState>(`/task/${taskId}`),

  result: (taskId: string) => request<GenerateResult>(`/result/${taskId}`),

  /** .onnx 模型文件下载地址（配合 <a download> 使用） */
  onnxDownloadUrl: (taskId: string) => `${API_BASE}/result/${taskId}/onnx`,

  protocolChunks: (query = "", limit = 100) =>
    request<{ chunks: ProtocolChunk[]; total: number; source: string }>(
      `/kb/protocol/chunks?query=${encodeURIComponent(query)}&limit=${limit}`,
    ),

  /** 星务协议知识库 · 基本数据集可视化载荷（356 条原始基线 + 增强对照） */
  protocolDataset: () => request<ProtocolDataset>("/kb/protocol/dataset"),

  /** 模型部署适配知识库 · 数据集可视化载荷（1,349 条交付记录 + 对照口径） */
  onnxDataset: () => request<OnnxDataset>("/kb/onnx/dataset"),

  paddleDataset: () => request<PaddleDataset>("/kb/paddle/dataset"),

  onnxRecords: (query = "", category = "", limit = 100) =>
    request<{ records: OnnxRecord[]; total: number }>(
      `/kb/onnx/records?query=${encodeURIComponent(query)}&category=${encodeURIComponent(category)}&limit=${limit}`,
    ),

  settings: () => request<SettingsState>("/settings"),

  saveSettings: (data: { api_key?: string; chat_model?: string; chat_base_url?: string }) =>
    request<{ status: string; llm_available: boolean }>("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),
};
