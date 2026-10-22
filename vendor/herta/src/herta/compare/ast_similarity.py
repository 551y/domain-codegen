from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter
import re

try:
    from apted import APTED, Config
    APTED_AVAILABLE = True
except ImportError:
    APTED_AVAILABLE = False

from herta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimilarityResult:
    """相似度结果"""

    tree_edit_similarity: float  # 0-1
    semantic_feature_similarity: float  # 0-1
    ast_similarity: float  # 0-1 加权总分
    details: Dict[str, Any]


def node_to_dict(node) -> Dict[str, Any]:
    """将 tree-sitter 节点转换为字典表示"""
    if node is None:
        return {"type": "null", "children": []}

    children = []
    if hasattr(node, "children") and node.children:
        for child in node.children:
            children.append(node_to_dict(child))

    return {
        "type": node.type if hasattr(node, "type") else "unknown",
        "children": children,
    }


class ASTNode:
    """用于 apted 的 AST 节点包装类"""

    def __init__(self, name: str, children: List["ASTNode"] = None):
        self.name = name
        self.children = children or []

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> "ASTNode":
        """从字典创建 ASTNode"""
        node_type = node_dict.get("type", "unknown")
        children = [
            cls.from_dict(child) for child in node_dict.get("children", [])
        ]
        return cls(node_type, children)

    def __repr__(self):
        return f"ASTNode({self.name})"


class CustomConfig(Config):
    """自定义 apted 配置"""

    def rename(self, node1, node2):
        """计算重命名代价"""
        if node1.name == node2.name:
            return 0
        # 某些节点类型可以视为相似
        similar_types = {
            ("binary_expression", "assignment_expression"),
            ("call_expression", "identifier"),
        }
        type_pair = (node1.name, node2.name)
        if type_pair in similar_types or type_pair[::-1] in similar_types:
            return 0.5
        return 1.0

    def delete(self, node):
        """删除代价"""
        return 1.0

    def insert(self, node):
        """插入代价"""
        return 1.0


class SemanticFeatureExtractor:
    """语义特征提取器"""

    def __init__(self, source: str):
        self.source = source

    def extract(self) -> Dict[str, Any]:
        """提取所有语义特征"""
        return {
            "call_set": self._extract_calls(),
            "literal_set": self._extract_literals(),
            "assignment_set": self._extract_assignments(),
            "identifier_set": self._extract_identifiers(),
            "return_patterns": self._extract_returns(),
            "control_flow": self._extract_control_flow(),
        }

    def _extract_calls(self) -> Set[str]:
        """提取函数调用集合"""
        calls = set()
        # 匹配函数调用: func_name(
        pattern = r'\b([a-zA-Z_]\w*)\s*\('
        for match in re.finditer(pattern, self.source):
            func_name = match.group(1)
            # 排除关键字
            if func_name not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                calls.add(func_name)
        return calls

    def _extract_literals(self) -> Set[str]:
        """提取字面量常量集合"""
        literals = set()
        # 十六进制（优先匹配，避免被十进制正则截断）
        hex_numbers = re.findall(r'\b0x[0-9a-fA-F]+\b', self.source)
        literals.update(hex_numbers)
        # 独立数字常量（排除标识符内部的数字，如 uint8_t 中的 8）
        numbers = re.findall(r'(?<![a-zA-Z_])\d+(?![a-zA-Z_])', self.source)
        literals.update(numbers)
        # 字符串
        strings = re.findall(r'"([^"]*)"', self.source)
        literals.update(strings)
        return literals

    def _extract_assignments(self) -> Set[str]:
        """提取赋值左值模式"""
        assignments = set()
        # 匹配赋值: x = 或 x->y = 或 x.y =
        pattern = r'([a-zA-Z_]\w*(?:\s*[\.\-]\s*[a-zA-Z_]\w*)?)\s*=(?!=)'
        for match in re.finditer(pattern, self.source):
            assignments.add(match.group(1).strip())
        return assignments

    def _extract_identifiers(self) -> Set[str]:
        """提取标识符集合"""
        identifiers = set()
        # 匹配标识符
        pattern = r'\b[a-zA-Z_]\w*\b'
        keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'sizeof', 'typeof',
            'int', 'char', 'void', 'float', 'double', 'long', 'short',
            'unsigned', 'signed', 'const', 'volatile', 'static', 'extern',
            'struct', 'union', 'enum', 'typedef',
        }
        for match in re.finditer(pattern, self.source):
            ident = match.group()
            if ident not in keywords:
                identifiers.add(ident)
        return identifiers

    def _extract_returns(self) -> List[str]:
        """提取 return 语句模式"""
        returns = []
        # 匹配 return 语句
        pattern = r'return\s*([^;]*)?;'
        for match in re.finditer(pattern, self.source):
            ret_value = match.group(1).strip() if match.group(1) else ""
            # 简化分类
            if not ret_value:
                returns.append("void")
            elif ret_value.isdigit() or ret_value in ['0', '1', '-1']:
                returns.append("constant")
            elif ret_value == "OK" or "SUCCESS" in ret_value:
                returns.append("success")
            elif "ERROR" in ret_value or "FAIL" in ret_value:
                returns.append("error")
            else:
                returns.append("variable")
        return returns

    def _extract_control_flow(self) -> Dict[str, int]:
        """提取控制流模式"""
        return {
            "if_count": len(re.findall(r'\bif\b', self.source)),
            "else_count": len(re.findall(r'\belse\b', self.source)),
            "for_count": len(re.findall(r'\bfor\b', self.source)),
            "while_count": len(re.findall(r'\bwhile\b', self.source)),
            "switch_count": len(re.findall(r'\bswitch\b', self.source)),
            "case_count": len(re.findall(r'\bcase\b', self.source)),
            "break_count": len(re.findall(r'\bbreak\b', self.source)),
            "continue_count": len(re.findall(r'\bcontinue\b', self.source)),
        }


class ASTSimilarityCalculator:
    """AST 相似度计算器"""

    def __init__(self):
        if not APTED_AVAILABLE:
            logger.warning("apted 未安装，树编辑相似度将使用简化计算")

    def calculate(
        self,
        gen_ast,
        gt_ast,
        gen_source: str,
        gt_source: str,
    ) -> SimilarityResult:
        """
        计算两个函数 AST 的相似度

        Args:
            gen_ast: 生成函数的 AST 节点
            gt_ast: 真实函数的 AST 节点
            gen_source: 生成函数源码
            gt_source: 真实函数源码

        Returns:
            SimilarityResult 相似度结果
        """
        # 1. 计算树编辑距离相似度
        tree_sim = self._calculate_tree_edit_similarity(gen_ast, gt_ast)

        # 2. 计算语义特征相似度
        semantic_sim = self._calculate_semantic_similarity(gen_source, gt_source)

        # 3. 加权计算总分
        ast_sim = 0.70 * tree_sim + 0.30 * semantic_sim

        return SimilarityResult(
            tree_edit_similarity=tree_sim,
            semantic_feature_similarity=semantic_sim,
            ast_similarity=ast_sim,
            details={
                "tree_edit": tree_sim,
                "semantic": semantic_sim,
                "weighted": ast_sim,
            },
        )

    def _calculate_tree_edit_similarity(self, gen_ast, gt_ast) -> float:
        """
        计算树编辑距离相似度

        使用 apted 算法计算 TED，然后归一化
        """
        if not APTED_AVAILABLE:
            # 简化计算：使用节点数量差异
            return self._simple_tree_similarity(gen_ast, gt_ast)

        try:
            # 转换为 apted 格式
            gen_dict = node_to_dict(gen_ast)
            gt_dict = node_to_dict(gt_ast)

            gen_tree = ASTNode.from_dict(gen_dict)
            gt_tree = ASTNode.from_dict(gt_dict)

            # 使用 apted 计算编辑距离
            apted = APTED(gen_tree, gt_tree, CustomConfig())
            distance = apted.compute_edit_distance()

            # 归一化：编辑距离最多等于较大树的节点数（全删+全插）
            max_nodes = max(self._count_nodes(gen_tree), self._count_nodes(gt_tree))
            if max_nodes == 0:
                return 1.0

            # 距离越小越相似，完全相同 → 1.0，完全不同 → 0.0
            similarity = max(0.0, 1.0 - (distance / max_nodes))
            return similarity

        except Exception as e:
            logger.warning(f"树编辑距离计算失败: {e}，使用简化计算")
            return self._simple_tree_similarity(gen_ast, gt_ast)

    def _simple_tree_similarity(self, gen_ast, gt_ast) -> float:
        """简化树相似度计算（节点数量和类型统计）"""
        gen_stats = self._collect_node_stats(gen_ast)
        gt_stats = self._collect_node_stats(gt_ast)

        if not gen_stats or not gt_stats:
            return 0.0

        # 计算节点数量相似度
        gen_total = sum(gen_stats.values())
        gt_total = sum(gt_stats.values())
        if gen_total == 0 or gt_total == 0:
            return 0.0

        count_sim = 1.0 - abs(gen_total - gt_total) / max(gen_total, gt_total)

        # 计算节点类型相似度
        all_types = set(gen_stats.keys()) | set(gt_stats.keys())
        if not all_types:
            return count_sim

        type_matches = 0
        for node_type in all_types:
            gen_count = gen_stats.get(node_type, 0)
            gt_count = gt_stats.get(node_type, 0)
            if gen_count > 0 and gt_count > 0:
                # 都有这种类型
                type_matches += min(gen_count, gt_count) / max(gen_count, gt_count)

        type_sim = type_matches / len(all_types) if all_types else 0.0

        # 综合
        return 0.5 * count_sim + 0.5 * type_sim

    def _collect_node_stats(self, node) -> Dict[str, int]:
        """收集节点类型统计"""
        stats = Counter()
        if node is None:
            return dict(stats)

        if hasattr(node, "type") and node.type:
            stats[node.type] += 1

        if hasattr(node, "children") and node.children:
            for child in node.children:
                child_stats = self._collect_node_stats(child)
                stats.update(child_stats)

        return dict(stats)

    def _count_nodes(self, tree: ASTNode) -> int:
        """计算树中节点数量"""
        count = 1  # 当前节点
        for child in tree.children:
            count += self._count_nodes(child)
        return count

    def _calculate_semantic_similarity(self, gen_source: str, gt_source: str) -> float:
        """
        计算语义特征相似度

        提取并比较：
        - 调用函数集合
        - 字面量常量集合
        - 赋值左值模式
        - 标识符集合
        - return 语句模式
        - 控制流模式
        """
        gen_extractor = SemanticFeatureExtractor(gen_source)
        gt_extractor = SemanticFeatureExtractor(gt_source)

        gen_features = gen_extractor.extract()
        gt_features = gt_extractor.extract()

        similarities = []

        # 1. 调用函数集合相似度 (Jaccard)
        call_sim = self._jaccard_similarity(
            gen_features["call_set"],
            gt_features["call_set"],
        )
        similarities.append(("calls", call_sim, 0.25))

        # 2. 字面量相似度
        literal_sim = self._jaccard_similarity(
            gen_features["literal_set"],
            gt_features["literal_set"],
        )
        similarities.append(("literals", literal_sim, 0.15))

        # 3. 赋值模式相似度
        assign_sim = self._jaccard_similarity(
            gen_features["assignment_set"],
            gt_features["assignment_set"],
        )
        similarities.append(("assignments", assign_sim, 0.20))

        # 4. 标识符相似度
        ident_sim = self._jaccard_similarity(
            gen_features["identifier_set"],
            gt_features["identifier_set"],
        )
        similarities.append(("identifiers", ident_sim, 0.15))

        # 5. return 模式相似度
        return_sim = self._list_similarity(
            gen_features["return_patterns"],
            gt_features["return_patterns"],
        )
        similarities.append(("returns", return_sim, 0.10))

        # 6. 控制流相似度
        control_sim = self._dict_similarity(
            gen_features["control_flow"],
            gt_features["control_flow"],
        )
        similarities.append(("control_flow", control_sim, 0.15))

        # 加权平均
        total_weight = sum(weight for _, _, weight in similarities)
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算 Jaccard 相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """计算列表相似度"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0

        # 使用 Counter 比较频率
        counter1 = Counter(list1)
        counter2 = Counter(list2)

        # 计算交集
        intersection = sum((counter1 & counter2).values())
        total = sum((counter1 | counter2).values())

        return intersection / total if total > 0 else 0.0

    def _dict_similarity(self, dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
        """计算字典数值相似度"""
        if not dict1 and not dict2:
            return 1.0

        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0

        total_diff = 0
        total_max = 0

        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            total_diff += abs(val1 - val2)
            total_max += max(val1, val2, 1)  # 避免除以0

        return 1.0 - (total_diff / total_max) if total_max > 0 else 1.0


def calculate_ast_similarity(
    gen_ast,
    gt_ast,
    gen_source: str,
    gt_source: str,
) -> SimilarityResult:
    """
    便捷函数：计算 AST 相似度

    Args:
        gen_ast: 生成函数的 AST
        gt_ast: 真实函数的 AST
        gen_source: 生成函数源码
        gt_source: 真实函数源码

    Returns:
        SimilarityResult
    """
    calculator = ASTSimilarityCalculator()
    return calculator.calculate(gen_ast, gt_ast, gen_source, gt_source)
