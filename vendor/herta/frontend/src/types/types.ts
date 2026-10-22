export interface KBInfo {
  source: string;
  path: string;
  chunk_count: number;
  embedding_model: string;
  chunk_strategy: string;
  has_index: boolean;
  meta: Record<string, unknown>;
}

export interface TargetInfo {
  name: string;
  has_docx: boolean;
  has_gt_config: boolean;
  has_gt_handler: boolean;
}

export interface CompareMetrics {
  topology_similarity: number;
  cmd_list_similarity: number;
  tele_handler_similarity: number;
  internal_handler_similarity: number;
  config_similarity: number;
  handler_similarity: number;
  overall_similarity: number;
  details?: {
    topology?: { present_fields: string[]; missing_fields: string[] };
    cmd_list?: { gen_count: number; gt_count: number };
    tele_handler?: { gen_count: number; gt_count: number };
    internal_handler?: { gen_count: number; gt_count: number };
    function_matches?: FunctionMatch[];
    unmatched_gt_functions?: string[];
    unmatched_gen_functions?: string[];
  };
}

export interface FunctionMatch {
  gt_func_name: string;
  gen_func_name: string;
  is_name_match: boolean;
  is_matched: boolean;
  ast_similarity: number;
  tree_edit_similarity: number;
  semantic_feature_similarity: number;
}

export interface TaskStatus {
  exists: boolean;
  type: string;
  status: string;
  progress: number;
  message: string;
  result?: Record<string, unknown>;
  error?: string;
}

export interface SettingsData {
  api_key_set: boolean;
  api_key_preview: string;
  chat_provider: string;
  chat_api_key_preview: string;
  chat_base_url: string;
  chat_model: string;
  embedding_provider: string;
  embedding_api_key_preview: string;
  embedding_base_url: string;
  embedding_model: string;
  embedding_dimensions: number;
}

export interface Projection2D {
  x: number;
  y: number;
  chunk_id: string;
  kind: string;
  file: string;
  label: string;
}
