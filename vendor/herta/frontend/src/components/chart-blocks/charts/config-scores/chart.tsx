"use client";

import { VChart } from "@visactor/react-vchart";
import type { IBarChartSpec } from "@visactor/vchart";
import type { ConfigScoreData } from "./index";

const DIMENSION_LABELS: Record<string, string> = {
  topology: "拓扑结构",
  cmd_list: "命令表",
  tele_handler: "遥测处理",
  internal_handler: "内部处理",
};

const DIMENSION_COLORS: Record<string, string> = {
  topology: "#2563eb",
  cmd_list: "#0891b2",
  tele_handler: "#6366f1",
  internal_handler: "#38bdf8",
};

export default function ConfigScoresChart({ data }: { data: ConfigScoreData }) {
  const values = Object.entries(data).map(([key, value]) => ({
    dimension: DIMENSION_LABELS[key] || key,
    score: Math.round(value * 100) / 100,
  }));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const spec: any = {
    type: "bar",
    data: [{ id: "scores", values }],
    xField: "score",
    yField: "dimension",
    seriesField: "dimension",
    direction: "horizontal",
    padding: { left: 20, right: 40, top: 10, bottom: 10 },
    color: {
      type: "ordinal",
      domain: Object.values(DIMENSION_LABELS),
      range: Object.values(DIMENSION_COLORS),
    },
    legends: { visible: false },
    axes: [
      {
        orient: "bottom",
        type: "linear",
        min: 0,
        max: 100,
        tick: { visible: true },
        label: { formatMethod: (v: string | string[]) => `${v}%` },
        grid: { visible: true, style: { stroke: "#dbe4f5", lineWidth: 0.5 } },
      },
      {
        orient: "left",
        type: "band",
        domainLine: { visible: false },
        tick: { visible: false },
        label: { style: { fontSize: 13 } },
      },
    ],
    bar: {
      style: {
        cornerRadius: [0, 6, 6, 0],
      },
      state: {
        hover: { outerBorder: { distance: 2, lineWidth: 2 } },
      },
    },
    tooltip: {
      trigger: ["click", "hover"],
      mark: {
        content: [
          { key: (d: Record<string, unknown>) => d?.dimension, value: (d: Record<string, unknown>) => `${d?.score}%` },
        ],
      },
    },
    label: {
      visible: true,
      position: "right",
      style: { fontSize: 12, fill: "#64748b" },
      formatMethod: (text: string | string[]) => `${text}%`,
    },
  };

  return <VChart spec={spec} style={{ height: "100%", minHeight: 260 }} />;
}
