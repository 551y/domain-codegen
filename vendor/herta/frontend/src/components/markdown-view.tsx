"use client";

import { Fragment } from "react";

/**
 * 轻量 Markdown 渲染器（评测报告专用）
 * 支持：标题、表格、粗体、行内代码、列表、分隔线、段落
 */

function inline(text: string, keyPrefix: string): React.ReactNode[] {
  // 粗体 + 行内代码
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-foreground">
          {p.slice(2, -2)}
        </strong>
      );
    if (p.startsWith("`") && p.endsWith("`"))
      return (
        <code key={`${keyPrefix}-${i}`} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">
          {p.slice(1, -1)}
        </code>
      );
    return <Fragment key={`${keyPrefix}-${i}`}>{p}</Fragment>;
  });
}

export default function MarkdownView({ content }: { content: string }) {
  const lines = content.split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // 表格
    if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|[\s:-]+\|/.test(lines[i + 1])) {
      const header = line.split("|").map((s) => s.trim()).filter(Boolean);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].split("|").map((s) => s.trim()).filter(Boolean));
        i++;
      }
      blocks.push(
        <div key={key++} className="my-3 overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                {header.map((h, j) => (
                  <th key={j} className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                    {inline(h, `h${j}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className="border-t border-border">
                  {r.map((c, ci) => (
                    <td key={ci} className="px-3 py-1.5 text-[13px]">
                      {inline(c, `r${ri}c${ci}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // 标题
    const hMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const cls =
        level === 1
          ? "mt-5 mb-3 text-xl font-bold tracking-tight"
          : level === 2
            ? "mt-5 mb-2.5 text-base font-semibold tracking-tight"
            : "mt-4 mb-2 text-sm font-semibold";
      blocks.push(
        <div key={key++} className={cls}>
          {inline(hMatch[2], `h${key}`)}
        </div>,
      );
      i++;
      continue;
    }

    // 分隔线
    if (/^\s*---+\s*$/.test(line)) {
      blocks.push(<hr key={key++} className="my-4 border-border" />);
      i++;
      continue;
    }

    // 列表
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={key++} className="my-2 space-y-1.5 pl-1">
          {items.map((it, j) => (
            <li key={j} className="flex gap-2 text-sm leading-relaxed text-muted-foreground">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-primary/60" />
              <span>{inline(it, `li${j}`)}</span>
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // 空行
    if (!line.trim()) {
      i++;
      continue;
    }

    // 段落
    blocks.push(
      <p key={key++} className="my-2 text-sm leading-relaxed text-muted-foreground">
        {inline(line, `p${key}`)}
      </p>,
    );
    i++;
  }

  return <div className="max-w-none">{blocks}</div>;
}
