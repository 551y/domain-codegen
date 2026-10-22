"use client";

import { ScatterChart } from "lucide-react";
import ChartTitle from "../../components/chart-title";
import EmbeddingScatterChart from "./chart";
import type { Projection2D } from "@/types/types";

export default function EmbeddingScatter({ data }: { data: Projection2D[] | null }) {
  return (
    <section className="flex flex-col gap-2">
      <ChartTitle title="知识库 Embedding 空间分布" icon={ScatterChart} />
      <div className="relative h-[380px]">
        {data && data.length > 0 ? (
          <EmbeddingScatterChart data={data} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            暂无可视化数据，请先生成知识库
          </div>
        )}
      </div>
    </section>
  );
}
