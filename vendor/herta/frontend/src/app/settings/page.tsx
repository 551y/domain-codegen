"use client";

import { useEffect, useState } from "react";
import { KeyRound, MessageSquareText, Shapes } from "lucide-react";
import Container from "@/components/container";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { SettingsData } from "@/types/types";

const PROVIDERS = [
  { value: "zai", label: "智谱 AI（BigModel）" },
  { value: "openai", label: "OpenAI 兼容（DeepSeek / 硅基流动等）" },
];

const inputCls =
  "w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none transition-shadow focus:ring-2 focus:ring-ring/40";

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsData | null>(null);
  // Chat
  const [chatProvider, setChatProvider] = useState("zai");
  const [chatBaseUrl, setChatBaseUrl] = useState("");
  const [chatApiKey, setChatApiKey] = useState("");
  const [chatModel, setChatModel] = useState("");
  // Embedding
  const [embProvider, setEmbProvider] = useState("zai");
  const [embBaseUrl, setEmbBaseUrl] = useState("");
  const [embApiKey, setEmbApiKey] = useState("");
  const [embModel, setEmbModel] = useState("");

  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    api.getSettings()
      .then((data) => {
        setSettings(data);
        setChatProvider(data.chat_provider || "zai");
        setChatBaseUrl(data.chat_base_url || "");
        setChatModel(data.chat_model || "");
        setEmbProvider(data.embedding_provider || "zai");
        setEmbBaseUrl(data.embedding_base_url || "");
        setEmbModel(data.embedding_model || "");
      })
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMsg("");
    try {
      const data: Record<string, string> = {
        chat_provider: chatProvider,
        chat_model: chatModel,
        embedding_provider: embProvider,
        embedding_model: embModel,
      };
      if (chatBaseUrl) data.chat_base_url = chatBaseUrl;
      if (chatApiKey) data.chat_api_key = chatApiKey;
      if (embBaseUrl) data.embedding_base_url = embBaseUrl;
      if (embApiKey) data.embedding_api_key = embApiKey;
      await api.saveSettings(data);
      setMsg("保存成功");
      setChatApiKey("");
      setEmbApiKey("");
      setSettings(await api.getSettings());
    } catch (e: unknown) {
      setMsg(`保存失败: ${e instanceof Error ? e.message : "未知错误"}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Container className="py-6">
      <div className="mx-auto max-w-2xl space-y-5">
        {/* Chat 供应商 */}
        <div className="top-beam rounded-xl border border-border bg-card p-5 shadow-soft-sm">
          <h3 className="mb-4 flex items-center gap-2 text-[15px] font-semibold tracking-tight">
            <MessageSquareText size={16} strokeWidth={1.8} className="text-primary" />
            对话模型（Chat）
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-sm font-medium">供应商</label>
                <select className={inputCls} value={chatProvider} onChange={(e) => setChatProvider(e.target.value)}>
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">模型</label>
                <input
                  className={inputCls}
                  placeholder={chatProvider === "openai" ? "deepseek-chat" : "glm-5.1"}
                  value={chatModel}
                  onChange={(e) => setChatModel(e.target.value)}
                />
              </div>
            </div>
            {chatProvider === "openai" && (
              <div>
                <label className="mb-1 block text-sm font-medium">接口地址（Base URL）</label>
                <input
                  className={inputCls}
                  placeholder="https://api.deepseek.com"
                  value={chatBaseUrl}
                  onChange={(e) => setChatBaseUrl(e.target.value)}
                />
              </div>
            )}
            <div>
              <label className="mb-1 flex items-center gap-1.5 text-sm font-medium">
                <KeyRound size={13} className="text-muted-foreground" /> API Key
              </label>
              <input
                type="password"
                className={inputCls}
                placeholder={
                  settings?.chat_api_key_preview ? `当前: ${settings.chat_api_key_preview}` : "输入 API Key"
                }
                value={chatApiKey}
                onChange={(e) => setChatApiKey(e.target.value)}
              />
              <div className="mt-1 text-xs text-muted-foreground">
                {settings?.chat_api_key_preview ? "已设置，留空保持不变" : "未设置；智谱供应商时默认使用 ZAI_API_KEY"}
              </div>
            </div>
          </div>
        </div>

        {/* Embedding 供应商 */}
        <div className="top-beam rounded-xl border border-border bg-card p-5 shadow-soft-sm">
          <h3 className="mb-4 flex items-center gap-2 text-[15px] font-semibold tracking-tight">
            <Shapes size={16} strokeWidth={1.8} className="text-primary" />
            向量模型（Embedding）
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-sm font-medium">供应商</label>
                <select className={inputCls} value={embProvider} onChange={(e) => setEmbProvider(e.target.value)}>
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">模型</label>
                <input
                  className={inputCls}
                  placeholder={embProvider === "openai" ? "如 BAAI/bge-m3" : "embedding-3"}
                  value={embModel}
                  onChange={(e) => setEmbModel(e.target.value)}
                />
              </div>
            </div>
            {embProvider === "openai" && (
              <div>
                <label className="mb-1 block text-sm font-medium">接口地址（Base URL）</label>
                <input
                  className={inputCls}
                  placeholder="https://api.siliconflow.cn"
                  value={embBaseUrl}
                  onChange={(e) => setEmbBaseUrl(e.target.value)}
                />
              </div>
            )}
            <div>
              <label className="mb-1 flex items-center gap-1.5 text-sm font-medium">
                <KeyRound size={13} className="text-muted-foreground" /> API Key
              </label>
              <input
                type="password"
                className={inputCls}
                placeholder={
                  settings?.embedding_api_key_preview ? `当前: ${settings.embedding_api_key_preview}` : "输入 API Key"
                }
                value={embApiKey}
                onChange={(e) => setEmbApiKey(e.target.value)}
              />
              <div className="mt-1 text-xs text-muted-foreground">
                {settings?.embedding_api_key_preview
                  ? "已设置，留空保持不变"
                  : "未设置；智谱供应商时默认使用 ZAI_API_KEY"}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button onClick={handleSave} disabled={saving}>
            {saving ? "保存中..." : "保存设置"}
          </Button>
          {msg && <span className="text-sm text-muted-foreground">{msg}</span>}
        </div>

        <div className="rounded-lg border border-border bg-muted/40 px-4 py-3 text-xs leading-relaxed text-muted-foreground">
          <p>提示：此处的设置仅在当前服务器会话内有效，重启后恢复环境变量配置。</p>
          <p className="mt-1">
            DeepSeek 可使用 OpenAI 兼容协议接入对话（Base URL 填 <code className="rounded bg-muted px-1 font-mono">https://api.deepseek.com</code>、
            模型 <code className="rounded bg-muted px-1 font-mono">deepseek-chat</code>）；但 DeepSeek 不提供 embedding 接口，
            向量模型需保持智谱或改用硅基流动等兼容服务（如 <code className="rounded bg-muted px-1 font-mono">BAAI/bge-m3</code>）。
          </p>
        </div>
      </div>
    </Container>
  );
}
