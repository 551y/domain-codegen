"""
Compare 模块 - 结果对比评测

比较 LLM 生成的 config.json 与 handler.c 和预期真实文件之间的相似度，
输出 metrics.json 与 comparison_report.md。

子模块：
- json_compare: JSON 配置结构比较
- c_ast_parser: C 代码 AST 解析
- ast_similarity: AST 相似度计算
- function_matcher: 函数匹配算法
- report_builder: 报告生成
"""

from herta.compare.json_compare import ConfigComparator, compare_configs
from herta.compare.c_ast_parser import CASTParser, parse_c_functions
from herta.compare.ast_similarity import ASTSimilarityCalculator, calculate_ast_similarity
from herta.compare.function_matcher import FunctionMatcher, match_functions
from herta.compare.report_builder import ReportBuilder, generate_comparison_report

__all__ = [
    # JSON 比较
    "ConfigComparator",
    "compare_configs",
    # C 代码解析
    "CASTParser",
    "parse_c_functions",
    # AST 相似度
    "ASTSimilarityCalculator",
    "calculate_ast_similarity",
    # 函数匹配
    "FunctionMatcher",
    "match_functions",
    # 报告生成
    "ReportBuilder",
    "generate_comparison_report",
]
