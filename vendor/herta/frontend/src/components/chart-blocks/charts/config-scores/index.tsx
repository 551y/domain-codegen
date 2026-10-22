"use client";

import { BarChart3 } from "lucide-react";
import ChartTitle from "../../components/chart-title";
import ConfigScoresChart from "./chart";

export interface ConfigScoreData {
  topology: number;
  cmd_list: number;
  tele_handler: number;
  internal_handler: number;
}

export default function ConfigScores({ data }: { data: ConfigScoreData | null }) {
  return (
    <section className="flex flex-col gap-2">
      <ChartTitle title="Config 相似度分维得分" icon={BarChart3} />
      <div className="relative h-[280px]">
        {data ? (
          <ConfigScoresChart data={data} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            暂无评测数据
          </div>
        )}
      </div>
    </section>
  );
}
