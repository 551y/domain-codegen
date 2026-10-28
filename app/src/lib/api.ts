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
  semantics_demo?: boolean;
  support_note?: string;
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
  evaluation_steps?: { name: string; status: "pending" | "running" | "passed" | "failed" | "skipped" | "unverified"; detail?: string }[];
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
  semtrans?: {
    status?: "passed" | "failed" | "unverified";
    verified?: boolean;
    provider?: string;
    model?: string;
    generation_elapsed_s?: number;
    total_elapsed_s?: number;
    repair_rounds?: number;
    validation_rounds?: {
      phase?: string;
      status?: string;
      verified?: boolean;
      valid_paired_cases?: number;
      search_attempts?: number;
      elapsed_s?: number;
      atol?: number;
      rtol?: number;
      issues?: unknown;
      input_search?: { framework?: string; attempts?: number; valid_cases?: number; elapsed_s?: number; budget_ms?: number }[];
    }[];
    semantic_evidence?: {
      alignment_records?: Record<string, unknown>[];
      parameter_alignment_records?: Record<string, unknown>[];
      semantic_issue_count?: number;
      first_mismatch?: Record<string, unknown> | null;
    }[];
    numerical_checks?: {
      case_index?: number;
      passed?: boolean;
      outputs?: {
        output_index?: number;
        shape?: number[];
        max_absolute_error?: number;
        max_relative_error?: number;
        failed_elements?: number;
        element_count?: number;
      }[];
    }[][];
    numerical_summaries?: {
      checked_cases?: number;
      checked_outputs?: number;
      max_absolute_error?: number | null;
      max_relative_error?: number | null;
      failed_elements?: number;
      checked_elements?: number;
    }[];
    stages?: { name: string; status: string; detail?: string }[];
    diagnostic?: unknown;
    limitations?: string[];
    fault_injection?: { label?: string; mutation?: string; scope?: string };
    code_history?: { round: number; stage: string; content: string }[];
    atol?: number;
    rtol?: number;
    search_budget_ms?: number;
    input_shape?: number[];
    profile_label?: string;
  };
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
  source_api?: string;
  target_api?: string;
  source_definition: Record<string, unknown>;
  target_definition: Record<string, unknown>;
  framework_version: string;
  verification_status: string;
}

export interface PaddleDataset {
  total: number;
  source_revision: string;
  categories: { key: string; label: string; count: number }[];
  composition: {
    paired: number;
    source_only: number;
    target_only: number;
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

export type ModelProvider = "zcode" | "codex" | "deepseek";

export interface ProviderProfile {
  base_url: string;
  model: string;
  api_key_type: "standard" | "coding_plan" | "other" | "unclassified";
  api_key_configured: boolean;
  configured: boolean;
}

export interface SettingsState {
  llm_available: boolean;
  provider: ModelProvider;
  chat_model: string;
  chat_base_url: string;
  api_key_type: "standard" | "coding_plan" | "other" | "unclassified";
  api_key_configured: boolean;
  configured: boolean;
  providers: Record<ModelProvider, ProviderProfile>;
  catalog: Record<ModelProvider, { label: string; vendor: string; kind: string }>;
}

// ---------- 接口 ----------

export const api = {
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

  generate: (kb: string, source: string, mode: "sample" | "upload", profile?: Record<string, unknown>) =>
    request<{ task_id: string }>("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kb, source, mode, profile }),
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

  saveSettings: (data: { api_key?: string; chat_model?: string; chat_base_url?: string; chat_provider?: ModelProvider; api_key_type?: SettingsState["api_key_type"]; clear_api_key?: boolean }) =>
    request<{ status: string } & SettingsState>("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),

  testSettings: () => request<{ status: string; provider: string; model: string }>("/settings/test", {
    method: "POST",
  }),
};
