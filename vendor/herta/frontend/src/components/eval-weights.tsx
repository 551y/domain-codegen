"use client";

/**
 * 评测权重可视化：整体 = 0.35×Config + 0.65×Handler；Config 四维权重
 * 纯 CSS 堆叠条，随主题变量着色
 */
export default function EvalWeights() {
  return (
    <div className="space-y-5">
      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <span className="text-xs font-medium">整体相似度构成</span>
          <span className="font-mono text-[10.5px] text-muted-foreground">Overall = 0.35 × Config + 0.65 × Handler</span>
        </div>
        <div className="flex h-7 overflow-hidden rounded-lg border border-border">
          <div
            className="flex items-center justify-center bg-primary/80 font-mono text-[10.5px] font-medium text-primary-foreground"
            style={{ width: "35%" }}
          >
            Config 35%
          </div>
          <div
            className="flex items-center justify-center bg-accent/80 font-mono text-[10.5px] font-medium text-accent-foreground"
            style={{ width: "65%" }}
          >
            Handler 65%
          </div>
        </div>
      </div>
      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <span className="text-xs font-medium">Config 四维权重</span>
          <span className="font-mono text-[10.5px] text-muted-foreground">拓扑 / 命令 / 遥测 / 内部</span>
        </div>
        <div className="flex h-7 overflow-hidden rounded-lg border border-border font-mono text-[10px] text-primary-foreground">
          {[
            { w: 20, label: "20%", c: "bg-[hsl(221,83%,63%)]" },
            { w: 30, label: "30%", c: "bg-[hsl(199,95%,50%)]" },
            { w: 30, label: "30%", c: "bg-[hsl(232,74%,68%)]" },
            { w: 20, label: "20%", c: "bg-[hsl(210,100%,72%)]" },
          ].map((s, i) => (
            <div key={i} className={`flex items-center justify-center ${s.c}`} style={{ width: `${s.w}%` }}>
              {s.label}
            </div>
          ))}
        </div>
        <div className="mt-2 grid grid-cols-4 gap-1 text-center text-[10.5px] text-muted-foreground">
          <span>拓扑结构</span>
          <span>命令表</span>
          <span>遥测处理</span>
          <span>内部处理</span>
        </div>
      </div>
      <p className="border-t border-border pt-3 text-[11.5px] leading-relaxed text-muted-foreground">
        Handler 占比更大，体现代码实现质量在协议迁移中的核心地位；AST 相似度由树编辑距离（70%）与语义特征（30%）合成。
      </p>
    </div>
  );
}
