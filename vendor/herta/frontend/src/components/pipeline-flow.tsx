import { ArrowRight, FileText, GitCompareArrows, Scissors, Search, Sparkles, Boxes } from "lucide-react";

/**
 * Herta RAG 流水线：协议文档 → 结构化中间表示 → 生成 → 评测
 * 纯 CSS 横向流程条（响应式换行）
 */
const STAGES = [
  { icon: FileText, title: "协议文档解析", desc: "docx / pdf / C / JSON", tint: "text-primary bg-primary/10" },
  { icon: Scissors, title: "结构感知分块", desc: "函数边界 · JSON 完整", tint: "text-accent bg-accent/10" },
  { icon: Boxes, title: "向量索引", desc: "FAISS + BM25 混合", tint: "text-primary bg-primary/10" },
  { icon: Search, title: "检索重排", desc: "多 query · bigram 精排", tint: "text-accent bg-accent/10" },
  { icon: Sparkles, title: "三阶段生成", desc: "Plan → Code → 验证", tint: "text-primary bg-primary/10" },
  { icon: GitCompareArrows, title: "多维度评测", desc: "JSON 四维 + AST 匹配", tint: "text-accent bg-accent/10" },
];

export default function PipelineFlow() {
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
      {STAGES.map((s, i) => (
        <div key={s.title} className="group relative">
          <div className="glow-card h-full rounded-lg border border-border bg-card px-3.5 py-3 shadow-soft-sm">
            <div className={`flex h-8 w-8 items-center justify-center rounded-md ${s.tint}`}>
              <s.icon size={16} strokeWidth={1.8} />
            </div>
            <div className="mt-2.5 text-[12.5px] font-semibold leading-tight">{s.title}</div>
            <div className="mt-1 font-mono text-[10px] leading-snug text-muted-foreground">{s.desc}</div>
          </div>
          {i < STAGES.length - 1 && (
            <ArrowRight
              size={14}
              className="absolute -right-2.5 top-1/2 z-10 hidden -translate-y-1/2 text-accent xl:block"
            />
          )}
        </div>
      ))}
    </div>
  );
}
