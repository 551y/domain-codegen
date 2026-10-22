"use client";

import { VChart } from "@visactor/react-vchart";
import type { Projection2D } from "@/types/types";

const KIND_COLORS: Record<string, string> = {
  doc: "#2563eb",
  code_json: "#0891b2",
  code_c: "#6366f1",
  code_h: "#38bdf8",
};

const KIND_LABELS: Record<string, string> = {
  doc: "文档",
  code_json: "JSON 配置",
  code_c: "C 源码",
  code_h: "头文件",
};

export default function EmbeddingScatterChart({ data }: { data: Projection2D[] }) {
  const values = data.map((p) => ({
    x: p.x,
    y: p.y,
    kind: KIND_LABELS[p.kind] || p.kind,
    label: p.label || p.chunk_id,
    file: p.file,
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const spec: any = {
    type: "scatter",
    data: [{ id: "scatter", values }],
    xField: "x",
    yField: "y",
    seriesField: "kind",
    size: 10,
    padding: { left: 10, right: 20, top: 10, bottom: 10 },
    color: {
      type: "ordinal",
      domain: Object.values(KIND_LABELS),
      range: Object.values(KIND_COLORS),
    },
    legends: [
      {
        visible: true,
        orient: "top",
        position: "start",
        item: { shape: { style: { size: 8 } } },
      },
    ],
    point: {
      style: {
        fillOpacity: 0.7,
        stroke: false,
        size: 10,
      },
      state: {
        hover: { fillOpacity: 1, stroke: "#1e40af", lineWidth: 1 },
      },
    },
    tooltip: {
      trigger: ["click", "hover"],
      mark: {
        content: [
          { key: "类型", value: (d: Record<string, unknown>) => d?.kind },
          { key: "文件", value: (d: Record<string, unknown>) => d?.file },
          { key: "标签", value: (d: Record<string, unknown>) => d?.label },
        ],
      },
    },
    axes: [
      { orient: "bottom", type: "linear", visible: false },
      { orient: "left", type: "linear", visible: false },
    ],
  };

  return <VChart spec={spec} style={{ height: "100%", minHeight: 380 }} />;
}
