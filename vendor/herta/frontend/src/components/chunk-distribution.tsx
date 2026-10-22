"use client";

import { useEffect, useState } from "react";
import { VChart } from "@visactor/react-vchart";
import { api } from "@/lib/api";
import type { Projection2D } from "@/types/types";

const KIND_LABELS: Record<string, string> = {
  doc: "文档",
  code_json: "JSON 配置",
  code_c: "C 源码",
  code_h: "头文件",
};

/** 知识库 chunk 类型分布（环形图），数据复用可视化投影端点 */
export default function ChunkDistribution({ source }: { source: string | null }) {
  const [dist, setDist] = useState<{ kind: string; count: number }[]>([]);

  useEffect(() => {
    if (!source) return;
    api
      .kbVisualization(source)
      .then((res) => {
        const counts: Record<string, number> = {};
        for (const p of res.points || []) {
          const label = KIND_LABELS[p.kind] || p.kind;
          counts[label] = (counts[label] || 0) + 1;
        }
        setDist(Object.entries(counts).map(([kind, count]) => ({ kind, count })));
      })
      .catch(() => {});
  }, [source]);

  if (!source || dist.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-xs text-muted-foreground">
        暂无知识库数据
      </div>
    );
  }

  const total = dist.reduce((s, d) => s + d.count, 0);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const spec: any = {
    type: "pie",
    data: [{ id: "dist", values: dist }],
    categoryField: "kind",
    valueField: "count",
    innerRadius: 0.62,
    outerRadius: 0.88,
    padAngle: 1.5,
    cornerRadius: 5,
    legends: { visible: true, orient: "bottom", item: { label: { style: { fontSize: 11 } } } },
    label: {
      visible: true,
      style: { fontSize: 11, fontWeight: 600 },
      formatMethod: (_t: string, d: Record<string, unknown>) =>
        `${Math.round(((d?.count as number) / total) * 100)}%`,
    },
    tooltip: {
      mark: {
        content: [
          { key: (d: Record<string, unknown>) => d?.kind, value: (d: Record<string, unknown>) => `${d?.count} chunks` },
        ],
      },
    },
  };

  return (
    <div>
      <VChart spec={spec} style={{ height: 210 }} />
      <div className="mt-1 text-center font-mono text-[10.5px] text-muted-foreground">
        共 {total} 个 chunks · UMAP 投影同源数据
      </div>
    </div>
  );
}
