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
}

export interface TaskState {
  exists: boolean;
  status: string;
  progress: number;
  message: string;
  error?: string | null;
}

export interface ResultFile {
  name: string;
  language: string;
  content: string;
}

export interface Evaluation {
  kind: string;
  scores: Record<string, number>;
  stats?: Record<string, number>;
  functions?: Record<string, number>;
  syntax_valid?: boolean;
  has_export_call?: boolean;
  has_checker?: boolean;
  dynamic_axes?: boolean;
  custom_symbolics?: number;
  total_calls?: number;
  retrieved_count?: number;
  retrieved?: { title: string; category_label: string; score: number; source: string }[];
  line_similarity_vs_reference?: number | null;
  note?: string;
}

export interface GenerateResult {
  kb: string;
  target: string;
  files: ResultFile[];
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

  protocolChunks: (query = "", limit = 100) =>
    request<{ chunks: ProtocolChunk[]; total: number; source: string }>(
      `/kb/protocol/chunks?query=${encodeURIComponent(query)}&limit=${limit}`,
    ),

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
