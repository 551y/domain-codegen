"""
报告生成模块

生成对比评测报告：
- comparison_report.md：可读性强的 Markdown 报告
- metrics.json：结构化指标数据
"""

import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

from herta.compare.json_compare import ConfigComparisonResult
from herta.compare.function_matcher import FunctionMatchingResult
from herta.logging import get_logger

logger = get_logger(__name__)


class ReportBuilder:
    """报告生成器"""

    def __init__(
        self,
        config_result: ConfigComparisonResult,
        handler_result: FunctionMatchingResult,
        target: str,
    ):
        """
        初始化报告生成器

        Args:
            config_result: 配置比较结果
            handler_result: Handler 比较结果
            target: 目标协议名称
        """
        self.config_result = config_result
        self.handler_result = handler_result
        self.target = target

    def generate_metrics(self) -> Dict[str, Any]:
        """
        生成 metrics.json 数据结构

        Returns:
            包含所有指标的字典
        """
        metrics = {
            # 基本信息
            "target": self.target,
            "generated_at": datetime.now().isoformat(),

            # Config 指标
            "topology_similarity": round(self.config_result.topology_similarity.score, 2),
            "cmd_list_similarity": round(self.config_result.cmd_list_similarity.score, 2),
            "tele_handler_similarity": round(self.config_result.tele_handler_similarity.score, 2),
            "internal_handler_similarity": round(self.config_result.internal_handler_similarity.score, 2),
            "config_similarity": round(self.config_result.config_similarity, 2),

            # Handler 指标
            "handler_similarity": round(self.handler_result.handler_similarity, 2),

            # 整体指标
            "overall_similarity": self._calculate_overall_similarity(),

            # 详细结果
            "config_details": self.config_result.to_dict(),
            "function_match_results": [
                m.to_dict() for m in self.handler_result.matches
            ],

            # 未匹配项
            "unmatched_gt_functions": self.handler_result.unmatched_gt_functions,
            "unmatched_gen_functions": self.handler_result.unmatched_gen_functions,
        }

        return metrics

    def generate_markdown_report(self) -> str:
        """
        生成 Markdown 报告

        Returns:
            Markdown 格式报告文本
        """
        lines = []

        # 标题
        lines.append(f"# Herta 对比评测报告 - {self.target}")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 总体评分
        overall = self._calculate_overall_similarity()
        lines.append("## 总体评分")
        lines.append("")
        lines.append(f"**整体相似度: {overall:.2f}%**")
        lines.append("")
        lines.append("### 各维度得分")
        lines.append("")
        lines.append("| 维度 | 得分 | 权重 |")
        lines.append("|------|------|------|")
        lines.append(f"| Config 拓扑结构 | {self.config_result.topology_similarity.score:.2f}% | 20% |")
        lines.append(f"| Config 命令表 | {self.config_result.cmd_list_similarity.score:.2f}% | 30% |")
        lines.append(f"| Config 遥测表 | {self.config_result.tele_handler_similarity.score:.2f}% | 30% |")
        lines.append(f"| Config 内部处理 | {self.config_result.internal_handler_similarity.score:.2f}% | 20% |")
        lines.append(f"| **Config 总分** | **{self.config_result.config_similarity:.2f}%** | - |")
        lines.append(f"| Handler 代码 | {self.handler_result.handler_similarity:.2f}% | 65% |")
        lines.append(f"| **整体加权** | **{overall:.2f}%** | - |")
        lines.append("")

        # Config 详细分析
        lines.append("---")
        lines.append("")
        lines.append("## Config 详细分析")
        lines.append("")

        # 拓扑结构分析
        lines.append("### 1. 拓扑结构相似度")
        lines.append("")
        topo = self.config_result.topology_similarity
        lines.append(f"**得分: {topo.score:.2f}%**")
        lines.append("")
        if topo.issues:
            lines.append("**发现的问题:**")
            for issue in topo.issues:
                lines.append(f"- {issue}")
        else:
            lines.append("✓ 拓扑结构检查通过")
        lines.append("")

        # 命令表分析
        lines.append("### 2. 命令表相似度")
        lines.append("")
        cmd = self.config_result.cmd_list_similarity
        lines.append(f"**得分: {cmd.score:.2f}%**")
        lines.append("")
        lines.append(f"- 生成命令数: {cmd.details.get('gen_count', 0)}")
        lines.append(f"- 真实命令数: {cmd.details.get('gt_count', 0)}")
        lines.append("")

        # 命令对比表格
        cmd_comparisons = cmd.details.get('cmd_comparisons', [])
        if cmd_comparisons:
            lines.append("**命令对比详情:**")
            lines.append("")
            lines.append("| 命令 | Handler 匹配 | Desc 相似度 | 得分 |")
            lines.append("|------|--------------|-------------|------|")
            for comp in cmd_comparisons:
                cmd_name = comp.get('cmd', 'unknown')
                handler_check = comp.get('checks', {}).get('handler', {})
                handler_match = "✓" if handler_check.get('match') else "✗"
                # desc 相似度在 checks['desc'] 下，handler 分支只有 match 标志
                desc_sim = comp.get('checks', {}).get('desc', {}).get('similarity', 0)
                score = comp.get('score', 0)
                lines.append(f"| {cmd_name} | {handler_match} | {desc_sim:.0f}% | {score:.1f} |")
            lines.append("")

        if cmd.issues:
            lines.append("**问题:**")
            for issue in cmd.issues:
                lines.append(f"- {issue}")
            lines.append("")

        # 遥测表分析
        lines.append("### 3. 遥测表相似度")
        lines.append("")
        tele = self.config_result.tele_handler_similarity
        lines.append(f"**得分: {tele.score:.2f}%**")
        lines.append("")
        lines.append(f"- 生成遥测项: {tele.details.get('gen_count', 0)}")
        lines.append(f"- 真实遥测项: {tele.details.get('gt_count', 0)}")
        lines.append("")
        if tele.issues:
            for issue in tele.issues:
                lines.append(f"- {issue}")
            lines.append("")

        # 内部处理表分析
        lines.append("### 4. 内部处理表相似度")
        lines.append("")
        internal = self.config_result.internal_handler_similarity
        lines.append(f"**得分: {internal.score:.2f}%**")
        lines.append("")
        lines.append(f"- 生成内部处理项: {internal.details.get('gen_count', 0)}")
        lines.append(f"- 真实内部处理项: {internal.details.get('gt_count', 0)}")
        lines.append("")
        if internal.issues:
            for issue in internal.issues:
                lines.append(f"- {issue}")
            lines.append("")

        # Handler 详细分析
        lines.append("---")
        lines.append("")
        lines.append("## Handler 代码详细分析")
        lines.append("")
        lines.append(f"**Handler 相似度: {self.handler_result.handler_similarity:.2f}%**")
        lines.append("")

        # 函数匹配表格
        lines.append("### 函数匹配结果")
        lines.append("")

        matches = self.handler_result.matches
        if matches:
            lines.append("| 真实函数 | 生成函数 | 同名 | AST 相似度 | TED | 语义 | 状态 |")
            lines.append("|----------|----------|------|------------|-----|------|------|")
            for match in matches:
                gt_name = match.gt_func_name
                gen_name = match.gen_func_name or "-"
                same_name = "✓" if match.is_name_match else "✗"
                ast_sim = f"{match.ast_similarity * 100:.1f}%" if match.is_matched else "-"
                ted = f"{match.tree_edit_similarity * 100:.0f}%" if match.is_matched else "-"
                sem = f"{match.semantic_feature_similarity * 100:.0f}%" if match.is_matched else "-"
                status = "✓ 已匹配" if match.is_matched else "✗ 未匹配"
                lines.append(f"| {gt_name} | {gen_name} | {same_name} | {ast_sim} | {ted} | {sem} | {status} |")
            lines.append("")

        # 未匹配函数
        if self.handler_result.unmatched_gt_functions:
            lines.append("### 未匹配的真实函数")
            lines.append("")
            lines.append("以下真实函数在生成代码中未找到匹配:")
            lines.append("")
            for func_name in self.handler_result.unmatched_gt_functions:
                lines.append(f"- `{func_name}`")
            lines.append("")

        if self.handler_result.unmatched_gen_functions:
            lines.append("### 未匹配的生成函数")
            lines.append("")
            lines.append("以下生成函数在真实代码中没有对应:")
            lines.append("")
            for func_name in self.handler_result.unmatched_gen_functions:
                lines.append(f"- `{func_name}`")
            lines.append("")

        return "\n".join(lines)

    def _calculate_overall_similarity(self) -> float:
        """
        计算整体相似度

        权重:
        - config_similarity: 35%
        - handler_similarity: 65%
        """
        return (
            self.config_result.config_similarity * 0.35 +
            self.handler_result.handler_similarity * 0.65
        )

    def save_reports(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        保存报告到文件

        Args:
            output_dir: 输出目录

        Returns:
            (metrics_path, report_path) 元组
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 metrics.json
        metrics = self.generate_metrics()
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics 已保存: {metrics_path}")

        # 保存 comparison_report.md
        report = self.generate_markdown_report()
        report_path = output_dir / "comparison_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report 已保存: {report_path}")

        return metrics_path, report_path


def generate_comparison_report(
    config_result: ConfigComparisonResult,
    handler_result: FunctionMatchingResult,
    target: str,
    output_dir: str,
) -> Tuple[Path, Path]:
    """
    便捷函数：生成并保存对比报告

    Args:
        config_result: 配置比较结果
        handler_result: Handler 比较结果
        target: 目标协议名称
        output_dir: 输出目录

    Returns:
        (metrics_path, report_path) 元组
    """
    builder = ReportBuilder(config_result, handler_result, target)
    return builder.save_reports(Path(output_dir))
