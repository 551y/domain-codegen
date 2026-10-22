import {
  Gauge,
  Database,
  Code2,
  GitCompareArrows,
  Settings,
  type LucideIcon,
} from "lucide-react";

export type Navigation = {
  icon: LucideIcon;
  name: string;
  href: string;
};

export type NavSection = {
  label: string;
  items: Navigation[];
};

export const siteConfig = {
  title: "Herta — 工业协议代码生成系统",
  description: "基于 RAG 的工业协议处理器代码自动生成平台：知识库构建、Config/Handler 生成、多维度评测",
};

export const navSections: NavSection[] = [
  {
    label: "流水线",
    items: [
      { icon: Gauge, name: "仪表盘", href: "/" },
      { icon: Database, name: "知识库", href: "/knowledge" },
      { icon: Code2, name: "代码生成", href: "/generate" },
      { icon: GitCompareArrows, name: "评测对比", href: "/compare" },
    ],
  },
  {
    label: "系统",
    items: [{ icon: Settings, name: "设置", href: "/settings" }],
  },
];

// 兼容旧引用
export const navigations: Navigation[] = navSections.flatMap((s) => s.items);
