"use client";

import { useState } from "react";
import { KeyRound, X } from "lucide-react";
import { api } from "@/lib/api";

export default function SettingsModal({ onClose }: { onClose: () => void }) {
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  const handleSave = async () => {
    setSaving(true);
    setMsg("");
    try {
      const data: Record<string, string> = {};
      if (apiKey) data.api_key = apiKey;
      if (baseUrl) data.chat_base_url = baseUrl;
      if (model) data.chat_model = model;
      const res = await api.saveSettings(data);
      setMsg(res.llm_available ? "已保存，LLM 可用 ✓" : "已保存");
      setTimeout(onClose, 900);
    } catch (e) {
      setMsg(`保存失败: ${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setSaving(false);
    }
  };

  const inputCls =
    "w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none transition-colors focus:border-primary/50";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm" onClick={onClose}>
      <div
        className="shadow-soft-lg w-[440px] rounded-2xl border border-border bg-card p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-[15px] font-semibold">
            <KeyRound size={15} className="text-primary" />
            LLM 设置
          </h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X size={16} />
          </button>
        </div>
        <div className="space-y-4">
          <div>
            <label className="mb-1 block text-[13px] font-medium">API Key</label>
            <input
              type="password"
              className={inputCls}
              placeholder="智谱 BigModel Key（或内网 LLM 服务密钥）"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-[13px] font-medium">接口地址（可选，内网 LLM 时填）</label>
            <input
              className={inputCls}
              placeholder="https:// 内网 OpenAI 兼容服务地址"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-[13px] font-medium">模型（可选）</label>
            <input
              className={inputCls}
              placeholder="glm-5.1"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-lg bg-primary px-4 py-2 text-[13px] font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-60"
            >
              {saving ? "保存中…" : "保存"}
            </button>
            {msg && <span className="text-[12px] text-muted-foreground">{msg}</span>}
          </div>
          <p className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
            设置仅在当前后端进程内生效，重启后恢复环境变量配置。未配置 Key 时，预置样例可展示预生成结果。
          </p>
        </div>
      </div>
    </div>
  );
}
