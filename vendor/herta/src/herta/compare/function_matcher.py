"""
函数匹配模块

使用最大权匹配算法将真实函数与生成的函数进行一对一匹配。
基于 AST 相似度构建相似度矩阵，使用匈牙利算法求解最优匹配。
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from herta.compare.c_ast_parser import FunctionInfo
from herta.compare.ast_similarity import ASTSimilarityCalculator, SimilarityResult
from herta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FunctionMatch:
    """函数匹配结果"""

    gt_func_name: str  # 真实函数名
    gen_func_name: Optional[str]  # 匹配的生成函数名
    is_name_match: bool  # 是否同名匹配
    tree_edit_similarity: float  # 0-1
    semantic_feature_similarity: float  # 0-1
    ast_similarity: float  # 0-1 加权总分
    is_matched: bool  # 是否成功匹配

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gt_func_name": self.gt_func_name,
            "gen_func_name": self.gen_func_name,
            "is_name_match": self.is_name_match,
            "tree_edit_similarity": round(self.tree_edit_similarity, 4),
            "semantic_feature_similarity": round(self.semantic_feature_similarity, 4),
            "ast_similarity": round(self.ast_similarity, 4),
            "is_matched": self.is_matched,
        }


@dataclass
class FunctionMatchingResult:
    """函数匹配完整结果"""

    matches: List[FunctionMatch]  # 所有匹配结果
    unmatched_gt_functions: List[str]  # 未匹配的真实函数
    unmatched_gen_functions: List[str]  # 未匹配的生成函数
    handler_similarity: float  # 整体相似度 (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_gt_functions": self.unmatched_gt_functions,
            "unmatched_gen_functions": self.unmatched_gen_functions,
            "handler_similarity": round(self.handler_similarity, 2),
        }


class FunctionMatcher:
    """
    函数匹配器

    使用最大权匹配算法（匈牙利算法）将真实函数与生成的函数进行一对一匹配。
    """

    # 相似度阈值，低于此值视为未匹配
    SIMILARITY_THRESHOLD = 0.3

    def __init__(self):
        self.similarity_calculator = ASTSimilarityCalculator()

    def match(
        self,
        gen_functions: List[FunctionInfo],
        gt_functions: List[FunctionInfo],
    ) -> FunctionMatchingResult:
        """
        执行函数匹配

        以真实函数列表为基准，与生成函数进行一对一匹配。

        Args:
            gen_functions: 生成的函数列表
            gt_functions: 真实函数列表（基准）

        Returns:
            FunctionMatchingResult 匹配结果
        """
        logger.info(f"开始函数匹配: 生成 {len(gen_functions)} 个 vs 真实 {len(gt_functions)} 个")

        if not gt_functions:
            logger.warning("真实函数列表为空")
            return FunctionMatchingResult(
                matches=[],
                unmatched_gt_functions=[],
                unmatched_gen_functions=[f.name for f in gen_functions],
                handler_similarity=0.0,
            )

        if not gen_functions:
            logger.warning("生成函数列表为空")
            return FunctionMatchingResult(
                matches=[],
                unmatched_gt_functions=[f.name for f in gt_functions],
                unmatched_gen_functions=[],
                handler_similarity=0.0,
            )

        # 构建相似度矩阵
        similarity_matrix = self._build_similarity_matrix(gen_functions, gt_functions)

        # 执行最优匹配
        matches = self._optimal_matching(
            gen_functions, gt_functions, similarity_matrix
        )

        # 识别未匹配函数：凡进入 matches 列表的（含低于阈值的配对行）都视为已配对，
        # 否则同一函数会同时出现在配对表与未匹配列表中
        paired_gt_names = {m.gt_func_name for m in matches}
        paired_gen_names = {m.gen_func_name for m in matches if m.gen_func_name}

        unmatched_gt = [f.name for f in gt_functions if f.name not in paired_gt_names]
        unmatched_gen = [f.name for f in gen_functions if f.name not in paired_gen_names]

        # 计算整体 handler 相似度
        handler_sim = self._calculate_handler_similarity(matches, len(gt_functions))

        logger.info(f"匹配完成: 成功 {len([m for m in matches if m.is_matched])} 对, "
                   f"未匹配真实 {len(unmatched_gt)} 个, 生成 {len(unmatched_gen)} 个")

        return FunctionMatchingResult(
            matches=matches,
            unmatched_gt_functions=unmatched_gt,
            unmatched_gen_functions=unmatched_gen,
            handler_similarity=handler_sim,
        )

    def _build_similarity_matrix(
        self,
        gen_functions: List[FunctionInfo],
        gt_functions: List[FunctionInfo],
    ) -> np.ndarray:
        """
        构建相似度矩阵

        矩阵 shape: (len(gen_functions), len(gt_functions))
        matrix[i][j] = gen_i 与 gt_j 的相似度
        """
        matrix = np.zeros((len(gen_functions), len(gt_functions)))

        for i, gen_func in enumerate(gen_functions):
            for j, gt_func in enumerate(gt_functions):
                # 计算 AST 相似度
                result = self.similarity_calculator.calculate(
                    gen_func.ast,
                    gt_func.ast,
                    gen_func.body_source,
                    gt_func.body_source,
                )
                matrix[i][j] = result.ast_similarity

        return matrix

    def _optimal_matching(
        self,
        gen_functions: List[FunctionInfo],
        gt_functions: List[FunctionInfo],
        similarity_matrix: np.ndarray,
    ) -> List[FunctionMatch]:
        """
        执行最优匹配

        使用匈牙利算法求解最大权匹配。
        """
        n_gen = len(gen_functions)
        n_gt = len(gt_functions)

        matches = []

        if n_gen == 0 or n_gt == 0:
            # 空列表处理
            for gt_func in gt_functions:
                matches.append(FunctionMatch(
                    gt_func_name=gt_func.name,
                    gen_func_name=None,
                    is_name_match=False,
                    tree_edit_similarity=0.0,
                    semantic_feature_similarity=0.0,
                    ast_similarity=0.0,
                    is_matched=False,
                ))
            return matches

        if SCIPY_AVAILABLE and n_gen > 0 and n_gt > 0:
            try:
                # 使用匈牙利算法
                # linear_sum_assignment 最小化代价，所以我们用负相似度作为代价
                cost_matrix = 1.0 - similarity_matrix  # 转换为代价

                # 处理非方阵情况：匈牙利算法要求方阵
                max_size = max(n_gen, n_gt)
                if max_size > min(n_gen, n_gt):
                    # 扩展到方阵，用高代价填充
                    if n_gen < max_size:
                        padding = np.ones((max_size - n_gen, n_gt)) * 2.0
                        cost_matrix = np.vstack([cost_matrix, padding])
                    if n_gt < max_size:
                        padding = np.ones((max_size, max_size - n_gt)) * 2.0
                        cost_matrix = np.hstack([cost_matrix, padding])

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # 只保留原始范围内的匹配
                matched_pairs = set()
                for r, c in zip(row_ind, col_ind):
                    if r < n_gen and c < n_gt:
                        matched_pairs.add((r, c))

                # 为所有真实函数创建匹配结果
                matched_gen_indices = set()
                for j, gt_func in enumerate(gt_functions):
                    best_match = None
                    best_sim = 0.0

                    # 找到最佳匹配
                    for i in range(n_gen):
                        if (i, j) in matched_pairs:
                            sim = similarity_matrix[i][j]
                            if sim > best_sim:
                                best_sim = sim
                                best_match = i

                    if best_match is not None and best_sim >= self.SIMILARITY_THRESHOLD:
                        gen_func = gen_functions[best_match]
                        result = self.similarity_calculator.calculate(
                            gen_func.ast, gt_func.ast,
                            gen_func.body_source, gt_func.body_source
                        )

                        matches.append(FunctionMatch(
                            gt_func_name=gt_func.name,
                            gen_func_name=gen_func.name,
                            is_name_match=(gen_func.name == gt_func.name),
                            tree_edit_similarity=result.tree_edit_similarity,
                            semantic_feature_similarity=result.semantic_feature_similarity,
                            ast_similarity=result.ast_similarity,
                            is_matched=True,
                        ))
                        matched_gen_indices.add(best_match)
                    else:
                        # 未匹配
                        matches.append(FunctionMatch(
                            gt_func_name=gt_func.name,
                            gen_func_name=gen_functions[best_match].name if best_match is not None else None,
                            is_name_match=False,
                            tree_edit_similarity=0.0,
                            semantic_feature_similarity=0.0,
                            ast_similarity=0.0,
                            is_matched=False,
                        ))

            except Exception as e:
                logger.warning(f"匈牙利算法匹配失败: {e}，使用贪心匹配")
                matches = self._greedy_matching(gen_functions, gt_functions, similarity_matrix)
        else:
            # 使用贪心匹配
            matches = self._greedy_matching(gen_functions, gt_functions, similarity_matrix)

        return matches

    def _greedy_matching(
        self,
        gen_functions: List[FunctionInfo],
        gt_functions: List[FunctionInfo],
        similarity_matrix: np.ndarray,
    ) -> List[FunctionMatch]:
        """
        贪心匹配算法

        作为匈牙利算法的备选。
        """
        n_gen = len(gen_functions)
        n_gt = len(gt_functions)

        matches = []
        used_gen = set()

        # 对每个真实函数，找最佳匹配的生成函数
        for j, gt_func in enumerate(gt_functions):
            best_gen_idx = -1
            best_sim = self.SIMILARITY_THRESHOLD

            for i in range(n_gen):
                if i in used_gen:
                    continue
                sim = similarity_matrix[i][j]
                if sim > best_sim:
                    best_sim = sim
                    best_gen_idx = i

            if best_gen_idx >= 0:
                gen_func = gen_functions[best_gen_idx]
                result = self.similarity_calculator.calculate(
                    gen_func.ast, gt_func.ast,
                    gen_func.body_source, gt_func.body_source
                )

                matches.append(FunctionMatch(
                    gt_func_name=gt_func.name,
                    gen_func_name=gen_func.name,
                    is_name_match=(gen_func.name == gt_func.name),
                    tree_edit_similarity=result.tree_edit_similarity,
                    semantic_feature_similarity=result.semantic_feature_similarity,
                    ast_similarity=result.ast_similarity,
                    is_matched=True,
                ))
                used_gen.add(best_gen_idx)
            else:
                # 未找到匹配
                matches.append(FunctionMatch(
                    gt_func_name=gt_func.name,
                    gen_func_name=None,
                    is_name_match=False,
                    tree_edit_similarity=0.0,
                    semantic_feature_similarity=0.0,
                    ast_similarity=0.0,
                    is_matched=False,
                ))

        return matches

    def _calculate_handler_similarity(
        self,
        matches: List[FunctionMatch],
        total_gt_functions: int,
    ) -> float:
        """
        计算整体 handler 相似度

        基于：
        - 所有真实函数的平均 AST 相似度
        - 函数覆盖率
        - 同名匹配率
        """
        if not matches or total_gt_functions == 0:
            return 0.0

        # 计算匹配的函数的平均相似度
        matched_sims = [m.ast_similarity for m in matches if m.is_matched]
        avg_similarity = sum(matched_sims) / len(matched_sims) if matched_sims else 0.0

        # 覆盖率
        coverage = len(matched_sims) / total_gt_functions if total_gt_functions > 0 else 0.0

        # 同名匹配率
        name_matches = sum(1 for m in matches if m.is_matched and m.is_name_match)
        name_match_rate = name_matches / len(matched_sims) if matched_sims else 0.0

        # 综合评分
        # 覆盖率占大头（函数缺失应严重扣分），平均相似度和同名率为辅
        handler_sim = (
            avg_similarity * 0.50 +
            coverage * 0.30 +
            name_match_rate * 0.20
        ) * 100  # 转换为百分制

        return min(handler_sim, 100.0)


def match_functions(
    gen_functions: List[FunctionInfo],
    gt_functions: List[FunctionInfo],
) -> FunctionMatchingResult:
    """
    便捷函数：匹配函数列表

    Args:
        gen_functions: 生成的函数列表
        gt_functions: 真实函数列表

    Returns:
        FunctionMatchingResult
    """
    matcher = FunctionMatcher()
    return matcher.match(gen_functions, gt_functions)
