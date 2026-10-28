"""Per-provider model settings with Windows DPAPI-protected API-key storage.

The UI has one settings entry, but each supported service has an independent
profile. The active profile is mirrored to Herta's legacy environment
variables for backwards compatibility; the encrypted store remains the source
of truth.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Optional

_LOCK = threading.RLock()
_APP_NAME = "H.DomainCodegen.ModelProvider.v2"
VALID_KEY_TYPES = {"standard", "coding_plan", "other", "unclassified"}
SUPPORTED_PROVIDERS = {"zcode", "codex", "deepseek"}
PROVIDER_ALIASES = {"openai": "codex", "zai": "zcode"}
PROVIDER_DEFAULTS = {
    "zcode": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-5.3", "api_key_type": "standard"},
    "codex": {"base_url": "https://api.openai.com/v1", "model": "gpt-5-codex", "api_key_type": "standard"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key_type": "standard"},
}
DEFAULT_PROVIDER = "zcode"
DEFAULT_BASE_URL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]["base_url"]
DEFAULT_MODEL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]["model"]


def normalize_provider(value: Optional[str], *, base_url: Optional[str] = None) -> str:
    """Normalize old names without guessing a new credential."""
    provider = (value or DEFAULT_PROVIDER).strip().lower()
    if provider in PROVIDER_ALIASES:
        if provider == "openai" and base_url:
            lowered = base_url.lower()
            if "bigmodel.cn" in lowered:
                return "zcode"
            if "deepseek.com" in lowered:
                return "deepseek"
        return PROVIDER_ALIASES[provider]
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("不支持的模型服务")
    return provider


def provider_catalog() -> dict[str, dict[str, str]]:
    return {
        "zcode": {"label": "ZCode", "vendor": "智谱 AI", "kind": "OpenAI-compatible"},
        "codex": {"label": "Codex", "vendor": "OpenAI", "kind": "OpenAI API"},
        "deepseek": {"label": "DeepSeek", "vendor": "DeepSeek", "kind": "OpenAI-compatible"},
    }


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise OSError("持久化模型服务密钥需要 Windows DPAPI")
    crypt32 = ctypes.WinDLL("Crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("Kernel32", use_last_error=True)
    source_buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = _DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _DataBlob()
    description = wintypes.LPWSTR()
    if decrypt:
        ok = crypt32.CryptUnprotectData(ctypes.byref(source), ctypes.byref(description), None, None, None, 0x1, ctypes.byref(result))
    else:
        description_text = ctypes.create_unicode_buffer(_APP_NAME)
        ok = crypt32.CryptProtectData(ctypes.byref(source), description_text, None, None, None, 0x1, ctypes.byref(result))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)


def _settings_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    if not root:
        raise OSError("找不到当前 Windows 用户的 LOCALAPPDATA")
    return Path(root) / "DomainCodegen" / "model-provider-settings.json"


def _empty_profile(provider: str) -> dict:
    defaults = PROVIDER_DEFAULTS[provider]
    return {"base_url": defaults["base_url"], "model": defaults["model"], "api_key_type": defaults["api_key_type"], "api_key": ""}


def _read_encrypted_profile(raw: dict, provider: str) -> dict:
    profile = _empty_profile(provider)
    encrypted = raw.get("api_key_dpapi", "")
    profile.update({"base_url": str(raw.get("base_url", profile["base_url"])), "model": str(raw.get("model", profile["model"])), "api_key_type": str(raw.get("api_key_type", "unclassified" if encrypted else profile["api_key_type"]))})
    if encrypted:
        profile["api_key"] = _dpapi(base64.b64decode(encrypted), decrypt=True).decode("utf-8")
    return profile


def _read_unlocked() -> dict:
    profiles = {provider: _empty_profile(provider) for provider in SUPPORTED_PROVIDERS}
    active = DEFAULT_PROVIDER
    if os.name != "nt":
        return {"provider": active, **profiles[active], "profiles": profiles}
    path = _settings_path()
    if not path.exists():
        return {"provider": active, **profiles[active], "profiles": profiles}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw.get("providers"), dict):
        active = normalize_provider(raw.get("active_provider", DEFAULT_PROVIDER))
        for name in SUPPORTED_PROVIDERS:
            entry = raw["providers"].get(name)
            if isinstance(entry, dict):
                profiles[name] = _read_encrypted_profile(entry, name)
    else:
        old_base = str(raw.get("base_url", DEFAULT_BASE_URL))
        active = normalize_provider(raw.get("provider", "openai"), base_url=old_base)
        profiles[active] = _read_encrypted_profile(raw, active)
    return {"provider": active, **profiles[active], "profiles": profiles}


def get_provider_settings() -> dict:
    with _LOCK:
        if os.name != "nt":
            return {"provider": _MEMORY_SETTINGS["provider"], **_MEMORY_SETTINGS["profiles"][_MEMORY_SETTINGS["provider"]], "profiles": _MEMORY_SETTINGS["profiles"]}
        return _read_unlocked()


def save_provider_settings(*, provider: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None,
                           api_key: Optional[str] = None, api_key_type: Optional[str] = None, clear_api_key: bool = False) -> dict:
    with _LOCK:
        current = _read_unlocked() if os.name == "nt" else get_provider_settings()
        selected = normalize_provider(provider or current["provider"], base_url=base_url)
        profiles = {name: dict(profile) for name, profile in current["profiles"].items()}
        updated = dict(profiles[selected])
        if base_url is not None:
            updated["base_url"] = base_url.strip().rstrip("/")
        if model is not None:
            updated["model"] = model.strip()
        if api_key_type is not None:
            if api_key_type not in VALID_KEY_TYPES:
                raise ValueError("不支持的 API Key 类别")
            updated["api_key_type"] = api_key_type
        if clear_api_key:
            updated["api_key"] = ""
        elif api_key:
            updated["api_key"] = api_key
        profiles[selected] = updated
        if os.name == "nt":
            path = _settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 2, "active_provider": selected, "providers": {}}
            for name, profile in profiles.items():
                payload["providers"][name] = {"base_url": profile["base_url"], "model": profile["model"], "api_key_type": profile["api_key_type"], "api_key_dpapi": (base64.b64encode(_dpapi(profile["api_key"].encode("utf-8"), decrypt=False)).decode("ascii") if profile.get("api_key") else "")}
            fd, temporary_name = tempfile.mkstemp(prefix="provider-settings-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        else:
            _MEMORY_SETTINGS["provider"] = selected
            _MEMORY_SETTINGS["profiles"] = profiles
        return {"provider": selected, **updated, "profiles": profiles}


_MEMORY_SETTINGS = {"provider": DEFAULT_PROVIDER, "profiles": {name: _empty_profile(name) for name in SUPPORTED_PROVIDERS}}


def public_provider_settings(settings: Optional[dict] = None) -> dict:
    value = settings if settings is not None else get_provider_settings()
    profiles = value.get("profiles", {})
    public_profiles = {}
    for name in SUPPORTED_PROVIDERS:
        profile = profiles.get(name, _empty_profile(name))
        public_profiles[name] = {"base_url": profile.get("base_url", PROVIDER_DEFAULTS[name]["base_url"]), "model": profile.get("model", PROVIDER_DEFAULTS[name]["model"]), "api_key_type": profile.get("api_key_type", "standard"), "api_key_configured": bool(profile.get("api_key")), "configured": bool(profile.get("api_key") and profile.get("base_url") and profile.get("model"))}
    active = normalize_provider(value.get("provider", DEFAULT_PROVIDER))
    selected = public_profiles[active]
    return {"provider": active, "base_url": selected["base_url"], "model": selected["model"], "api_key_type": selected["api_key_type"], "api_key_configured": selected["api_key_configured"], "configured": selected["configured"], "providers": public_profiles, "catalog": provider_catalog()}


def restore_provider_settings_to_env() -> None:
    """Mirror the active profile to Herta's legacy OpenAI-compatible settings."""
    try:
        if os.name == "nt" and not _settings_path().exists():
            legacy_key = os.environ.get("HERTA_CHAT_API_KEY") or os.environ.get("ZAI_API_KEY", "")
            if legacy_key:
                save_provider_settings(provider=os.environ.get("HERTA_CHAT_PROVIDER", "openai"), base_url=os.environ.get("HERTA_CHAT_BASE_URL", DEFAULT_BASE_URL), model=os.environ.get("HERTA_CHAT_MODEL", DEFAULT_MODEL), api_key=legacy_key, api_key_type="unclassified")
        settings = get_provider_settings()
    except Exception:
        return
    os.environ["HERTA_CHAT_API_KEY"] = settings.get("api_key", "")
    os.environ["ZAI_API_KEY"] = settings.get("api_key", "")
    os.environ["HERTA_CHAT_PROVIDER"] = "openai"
    os.environ["HERTA_CHAT_BASE_URL"] = settings.get("base_url", DEFAULT_BASE_URL)
    os.environ["HERTA_CHAT_MODEL"] = settings.get("model", DEFAULT_MODEL)
    os.environ["DOMAINCODEGEN_PROVIDER"] = settings["provider"]
    os.environ["DOMAINCODEGEN_API_KEY_TYPE"] = settings.get("api_key_type", "standard")
