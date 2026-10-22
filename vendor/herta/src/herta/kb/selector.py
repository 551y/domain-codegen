"""
自动大小选择分块策略 (Auto-Size Chunker)

从候选的分块大小和重叠大小中，选择最优组合
当前实现简化版启发式策略，基于分块质量指标进行评估
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from herta.kb.schemas import SourceDocument, Chunk
from herta.kb.chunker import BaseChunker, ChunkConfig, ChunkReport
from herta.kb.simple_chunker import SimpleChunker
from herta.logging import get_logger
from herta.utils.paths import ensure_dir

logger = get_logger(__name__)


@dataclass
class SizeCandidate:
    """候选分块参数"""

    chunk_size: int
    chunk_overlap: int
    score: float = 0.0
    report: Optional[ChunkReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "score": self.score,
            "report": self.report.to_dict() if self.report else None,
        }


class AutoSizeChunker(BaseChunker):
    """
    自动大小选择分块策略

    评估多组候选参数，选择最优组合

    当前使用简化启发式策略（非完整RAG基准评测）：
    1. 对每组参数执行分块
    2. 计算分块质量指标
    3. 选择综合评分最高的参数
    """

    def __init__(self, config: ChunkConfig):
        super().__init__(config)
        if config.strategy != "autosize":
            raise ValueError(f"AutoSizeChunker 需要 strategy='autosize', 得到 '{config.strategy}'")

        self.candidate_sizes = config.candidate_sizes or [300, 500, 800, 1000]
        self.candidate_overlaps = config.candidate_overlaps or [50, 100]
        self.eval_questions = config.eval_questions
        self.selection_output = config.selection_output

        self._selection_report: Optional[Dict[str, Any]] = None
        self._best_params: Optional[SizeCandidate] = None

    def chunk(self, doc: SourceDocument) -> List[Chunk]:
        """
        对单个文档分块

        对于单个文档，直接使用默认参数分块
        完整的参数选择应在 chunk_batch 中进行
        """
        # 使用默认参数
        config = ChunkConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy="simple"
        )
        simple_chunker = SimpleChunker(config)
        return simple_chunker.chunk(doc)

    def chunk_batch(
        self,
        docs: List[SourceDocument],
        progress_callback: Optional[Any] = None
    ) -> List[Chunk]:
        """
        批量分块并自动选择最优参数

        流程：
        1. 对每组候选参数执行分块
        2. 评估分块质量
        3. 选择最优参数
        4. 使用最优参数重新分块（或使用缓存结果）
        5. 生成选择报告
        """
        logger.info("=" * 60)
        logger.info("开始 Auto-Size 分块参数选择")
        logger.info("=" * 60)
        logger.info(f"候选大小: {self.candidate_sizes}")
        logger.info(f"候选重叠: {self.candidate_overlaps}")

        # 生成所有候选组合
        candidates: List[SizeCandidate] = []
        for size in self.candidate_sizes:
            for overlap in self.candidate_overlaps:
                if overlap < size:  # 重叠必须小于大小
                    candidates.append(SizeCandidate(size, overlap))

        logger.info(f"共 {len(candidates)} 组候选参数")

        # 评估每组参数
        evaluated = []
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"评估候选 {i}/{len(candidates)}: size={candidate.chunk_size}, overlap={candidate.chunk_overlap}")

            score, report = self._evaluate_candidate(docs, candidate)
            candidate.score = score
            candidate.report = report
            evaluated.append(candidate)

            logger.info(f"  得分: {score:.4f}, chunks: {report.total_chunks if report else 0}")

        # 选择最优参数
        best_candidate = max(evaluated, key=lambda c: c.score)
        self._best_params = best_candidate

        logger.info("-" * 60)
        logger.info(f"最优参数: size={best_candidate.chunk_size}, overlap={best_candidate.chunk_overlap}")
        logger.info(f"最优得分: {best_candidate.score:.4f}")
        logger.info("-" * 60)

        # 使用最优参数重新分块
        logger.info("使用最优参数执行最终分块...")
        final_config = ChunkConfig(
            chunk_size=best_candidate.chunk_size,
            chunk_overlap=best_candidate.chunk_overlap,
            strategy="simple"
        )
        final_chunker = SimpleChunker(final_config)

        all_chunks = final_chunker.chunk_batch(docs)

        # 生成报告
        self._report = final_chunker.get_report()
        self._selection_report = self._generate_selection_report(evaluated, best_candidate)

        # 保存选择报告
        if self.selection_output:
            self._save_selection_report(self.selection_output)

        logger.info(f"Auto-Size 分块完成: {len(docs)} 个文档 -> {len(all_chunks)} 个 chunks")
        return all_chunks

    def _evaluate_candidate(
        self,
        docs: List[SourceDocument],
        candidate: SizeCandidate
    ) -> tuple[float, Optional[ChunkReport]]:
        """
        评估候选参数

        返回：
            (score, report) 元组
        """
        # 创建临时 chunker
        config = ChunkConfig(
            chunk_size=candidate.chunk_size,
            chunk_overlap=candidate.chunk_overlap,
            strategy="simple"
        )
        chunker = SimpleChunker(config)

        # 执行分块（chunk_batch 才会生成统计报告，供评分使用）
        all_chunks = chunker.chunk_batch(docs)

        # 获取报告
        report = chunker.get_report()

        # 计算评分（启发式）
        score = self._calculate_score(report, candidate)

        return score, report

    def _calculate_score(self, report: Optional[ChunkReport], candidate: SizeCandidate) -> float:
        """
        计算分块质量评分（启发式）

        考虑因素：
        1. 平均 chunk 大小（适中为佳）
        2. chunk 数量（不过多也不过少）
        3. 大小差异（越均匀越好）
        """
        if not report or report.total_chunks == 0:
            return 0.0

        scores = []

        # 1. 平均大小评分（目标 500-800 字符）
        avg_size = report.avg_chunk_size
        if 400 <= avg_size <= 1000:
            size_score = 1.0
        elif 200 <= avg_size < 400:
            size_score = 0.7
        elif 1000 < avg_size <= 1500:
            size_score = 0.8
        else:
            size_score = 0.4
        scores.append(size_score)

        # 2. 大小均匀性评分（标准差/均值，越低越好）
        if report.avg_chunk_size > 0:
            sizes = [report.min_chunk_size, report.max_chunk_size]
            if len(sizes) > 1 and report.total_chunks > 1:
                # 用 (max-min)/avg 作为变异指标
                variation = (report.max_chunk_size - report.min_chunk_size) / report.avg_chunk_size
                uniformity_score = max(0, 1 - variation / 2)  # 归一化
            else:
                uniformity_score = 1.0
        else:
            uniformity_score = 0.0
        scores.append(uniformity_score)

        # 3. chunk 数量评分（目标 50-500 个，根据文档大小调整）
        total_chars = report.total_chars
        if total_chars > 0:
            expected_chunks = total_chars / 600  # 假设理想平均 600 字符
            actual_chunks = report.total_chunks

            # 数量偏差评分
            if 0.5 <= actual_chunks / expected_chunks <= 2.0:
                count_score = 1.0
            elif 0.3 <= actual_chunks / expected_chunks < 0.5:
                count_score = 0.7
            elif 2.0 < actual_chunks / expected_chunks <= 3.0:
                count_score = 0.8
            else:
                count_score = 0.5
        else:
            count_score = 0.0
        scores.append(count_score)

        # 4. 重叠比例评分（适中为佳，避免信息丢失或冗余）
        overlap_ratio = candidate.chunk_overlap / candidate.chunk_size
        if 0.05 <= overlap_ratio <= 0.2:  # 5%-20%
            overlap_score = 1.0
        elif 0.2 < overlap_ratio <= 0.3:
            overlap_score = 0.8
        else:
            overlap_score = 0.6
        scores.append(overlap_score)

        # 综合评分（加权平均）
        weights = [0.35, 0.25, 0.25, 0.15]
        final_score = sum(s * w for s, w in zip(scores, weights))

        return final_score

    def _generate_selection_report(
        self,
        candidates: List[SizeCandidate],
        best: SizeCandidate
    ) -> Dict[str, Any]:
        """生成参数选择报告"""
        return {
            "strategy": "autosize",
            "evaluation_method": "heuristic_scoring",
            "note": "当前使用启发式评分，非完整RAG基准评测",
            "candidate_sizes": self.candidate_sizes,
            "candidate_overlaps": self.candidate_overlaps,
            "all_candidates": [c.to_dict() for c in candidates],
            "selected": best.to_dict(),
            "selection_criteria": {
                "avg_size_target": "400-1000 chars",
                "uniformity": "prefer low variation",
                "count_balance": "prefer expected count",
                "overlap_ratio": "prefer 5%-20%",
            }
        }

    def _save_selection_report(self, output_path: Path) -> None:
        """保存选择报告到文件"""
        if not self._selection_report:
            return

        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self._selection_report, f, indent=2, ensure_ascii=False)

        logger.info(f"选择报告已保存: {output_path}")

    def get_selection_report(self) -> Optional[Dict[str, Any]]:
        """获取参数选择报告"""
        return self._selection_report

    def get_best_params(self) -> Optional[SizeCandidate]:
        """获取最优参数"""
        return self._best_params

