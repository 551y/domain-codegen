"use client";

import { useEffect, useState } from "react";
import { Check, CheckCircle2, KeyRound, X, XCircle } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelProvider, SettingsState } from "@/lib/api";

const PROVIDERS: Array<{
  id: ModelProvider;
  label: string;
  vendor: string;
  description: string;
  logo: string;
  tone: string;
}> = [
  { id: "zcode", label: "ZCode", vendor: "智谱 AI", description: "GLM / OpenAI 兼容接口", logo: "/brands/zhipu.svg", tone: "text-violet-500 bg-violet-500/10" },
  { id: "codex", label: "Codex", vendor: "OpenAI", description: "OpenAI API · Codex 模型", logo: "/brands/openai.svg", tone: "text-emerald-600 bg-emerald-500/10" },
  { id: "deepseek", label: "DeepSeek", vendor: "DeepSeek", description: "DeepSeek 官方 API", logo: "/brands/deepseek.svg", tone: "text-sky-500 bg-sky-500/10" },
];

type KeyType = SettingsState["api_key_type"];

export default function SettingsModal({ onClose }: { onClose: () => void }) {
  const [provider, setProvider] = useState<ModelProvider>("zcode");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKeyType, setApiKeyType] = useState<KeyType>("standard");
  const [apiKeyConfigured, setApiKeyConfigured] = useState(false);
  const [profiles, setProfiles] = useState<SettingsState["providers"] | null>(null);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [msgTone, setMsgTone] = useState<"neutral" | "success" | "error">("neutral");

  const showMessage = (text: string, tone: "neutral" | "success" | "error") => {
    setMsg(text);
    setMsgTone(tone);
  };

  const applyProfile = (settings: SettingsState, selected: ModelProvider) => {
    const profile = settings.providers?.[selected];
    if (!profile) return;
    setProvider(selected);
    setBaseUrl(profile.base_url);
    setModel(profile.model);
    setApiKeyType(profile.api_key_type || "standard");
    setApiKeyConfigured(profile.api_key_configured);
    setApiKey("");
  };

  useEffect(() => {
    api.settings().then((settings) => {
      setProfiles(settings.providers);
      applyProfile(settings, settings.provider);
    }).catch(() => showMessage("无法读取模型服务设置", "error"));
  }, []);

  const selectProvider = (selected: ModelProvider) => {
    if (!profiles?.[selected]) return;
    setProvider(selected);
    setBaseUrl(profiles[selected].base_url);
    setModel(profiles[selected].model);
    setApiKeyType(profiles[selected].api_key_type || "standard");
    setApiKeyConfigured(profiles[selected].api_key_configured);
    setApiKey("");
    showMessage("", "neutral");
  };

  const handleSave = async (testAfterSave = false, clearApiKey = false) => {
    setSaving(true);
    showMessage("", "neutral");
    try {
      const res = await api.saveSettings({
        chat_provider: provider,
        chat_base_url: baseUrl,
        chat_model: model,
        api_key_type: apiKeyType,
        ...(apiKey ? { api_key: apiKey } : {}),
        ...(clearApiKey ? { clear_api_key: true } : {}),
      });
      setProfiles(res.providers);
      setApiKey("");
      setApiKeyConfigured(res.api_key_configured);
      setApiKeyType(res.api_key_type);
      showMessage(clearApiKey ? "已清除当前服务 API Key" : res.api_key_configured ? "已加密保存 · 当前服务已配置" : "设置已保存", "success");
      if (testAfterSave && !clearApiKey) {
        if (res.api_key_type === "coding_plan") {
          showMessage("Coding Plan 仅可用于官方支持工具，本平台不会发送该 Key。", "error");
          return;
        }
        const tested = await api.testSettings();
        showMessage(`连接成功 · ${tested.provider} / ${tested.model}`, "success");
      }
    } catch (e) {
      showMessage(`${testAfterSave ? "连接测试失败" : "保存失败"}: ${e instanceof Error ? e.message : "未知错误"}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const inputCls = "w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none transition-colors focus:border-primary/50";
  const selectedMeta = PROVIDERS.find((item) => item.id === provider) || PROVIDERS[0];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/20 backdrop-blur-sm" onClick={onClose}>
      <div className="shadow-soft-lg max-h-[92vh] w-[560px] overflow-y-auto rounded-2xl border border-border bg-card p-6" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-[15px] font-semibold"><KeyRound size={15} className="text-primary" />LLM 设置</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground"><X size={16} /></button>
        </div>

        <div className="space-y-4">
          <div>
            <div className="mb-2 text-[13px] font-medium">选择模型服务</div>
            <div className="grid grid-cols-3 gap-2">
              {PROVIDERS.map((item) => {
                const configured = profiles?.[item.id]?.api_key_configured;
                return (
                  <button key={item.id} type="button" onClick={() => selectProvider(item.id)}
                    className={`relative rounded-xl border p-3 text-left transition-all ${provider === item.id ? "border-primary bg-primary/5 ring-1 ring-primary/30" : "border-border hover:border-primary/40"}`}>
                    {provider === item.id && <Check size={13} className="absolute right-2 top-2 text-primary" />}
                    <span className={`mb-2 flex h-8 w-8 items-center justify-center rounded-lg ${item.tone}`}><img src={item.logo} alt={`${item.label} 官方图标`} className="h-5 w-5 object-contain" /></span>
                    <span className="block text-[12px] font-semibold">{item.label}</span>
                    <span className="mt-0.5 block text-[9.5px] text-muted-foreground">{item.vendor}</span>
                    <span className="mt-1 block text-[9px] leading-tight text-muted-foreground/80">{item.description}</span>
                    {configured && <span className="mt-2 inline-block rounded bg-emerald-500/10 px-1.5 py-0.5 text-[9px] text-emerald-600">已配置</span>}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
            当前服务：<span className="font-medium text-foreground">{selectedMeta.label} · {selectedMeta.vendor}</span>。三套服务的 Base URL、模型和 API Key 独立保存；协议、PyTorch→ONNX 与 PyTorch→Paddle（SemTrans）使用当前选中的服务。
          </div>

          <div>
            <label className="mb-1 block text-[13px] font-medium">模型服务 API Key</label>
            <input type="password" autoComplete="new-password" className={inputCls}
              placeholder={apiKeyConfigured ? "已配置；留空保留当前 Key，输入新 Key 可替换" : "粘贴 API Key（保存后不再显示）"}
              value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <p className="mt-1 text-[10px] text-muted-foreground">{apiKeyConfigured ? "当前服务 API Key 已加密保存" : "当前服务尚未配置 API Key"}</p>
          </div>
          <div>
            <label className="mb-1 block text-[13px] font-medium">官方 OpenAI-compatible Base URL</label>
            <input className={inputCls} value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
          </div>
          <div>
            <label className="mb-1 block text-[13px] font-medium">模型 ID</label>
            <input className={inputCls} value={model} onChange={(e) => setModel(e.target.value)} />
          </div>
          <div>
            <label className="mb-1 block text-[13px] font-medium">API Key 类别</label>
            <select className={inputCls} value={apiKeyType} onChange={(e) => setApiKeyType(e.target.value as KeyType)}>
              <option value="standard">标准 API Key（可由本平台后端调用）</option>
              <option value="coding_plan">GLM Coding Plan（仅官方支持工具）</option>
              <option value="other">其他 OpenAI-compatible 服务 Key</option>
              <option value="unclassified" disabled>类别未确认</option>
            </select>
            {apiKeyType === "coding_plan" && <p className="mt-1 rounded bg-amber-500/10 px-2 py-1.5 text-[10px] leading-relaxed text-amber-800">可加密保存，但本平台不会发送 Coding Plan Key；请使用智谱标准 API Key。</p>}
          </div>
          <p className="rounded-lg border border-border bg-muted/40 px-3 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
            ZCode：<code>https://open.bigmodel.cn/api/paas/v4</code>；Codex：<code>https://api.openai.com/v1</code>（Responses API）；DeepSeek：<code>https://api.deepseek.com</code>（Chat Completions）。模型 ID 可按账号实际可用模型调整。<br />
            API Key 只在后端调用期间解密，页面不会回显已保存密钥。参考 <a className="text-primary underline" href="https://docs.bigmodel.cn/cn/coding-plan/tool/zcode" target="_blank" rel="noreferrer">ZCode 文档</a>、<a className="text-primary underline" href="https://platform.openai.com/docs/api-reference/chat" target="_blank" rel="noreferrer">OpenAI API</a> 和 <a className="text-primary underline" href="https://api-docs.deepseek.com/" target="_blank" rel="noreferrer">DeepSeek API</a>。
          </p>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <button onClick={() => handleSave()} disabled={saving} className="rounded-lg bg-primary px-4 py-2 text-[13px] font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-60">{saving ? "保存中…" : "加密保存"}</button>
            <button onClick={() => handleSave(true)} disabled={saving || apiKeyType === "coding_plan" || apiKeyType === "unclassified" || (!apiKeyConfigured && !apiKey)} className="rounded-lg border border-border px-3 py-2 text-[12px] disabled:opacity-50">连接当前服务</button>
            <button onClick={() => handleSave(false, true)} disabled={saving || !apiKeyConfigured} className="rounded-lg border border-border px-3 py-2 text-[12px] text-destructive disabled:opacity-50">清除当前 Key</button>
            {msg && (
              <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[12px] font-semibold ${msgTone === "success" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700" : msgTone === "error" ? "border-red-500/30 bg-red-500/10 text-red-700" : "border-border bg-muted/50 text-muted-foreground"}`} role="status" aria-live="polite">
                {msgTone === "success" ? <CheckCircle2 size={14} aria-hidden="true" /> : msgTone === "error" ? <XCircle size={14} aria-hidden="true" /> : null}
                {msg}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
