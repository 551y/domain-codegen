"""
JSON 配置比较模块

比较 LLM 生成的 config.json 与真实 config.json，
输出 topology_similarity、cmd_list_similarity、tele_handler_similarity、
internal_handler_similarity 和 config_similarity。
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from rapidfuzz import fuzz
from pathlib import Path

from herta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompareResult:
    """比较结果数据类"""

    score: float  # 0-100 百分制
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class ConfigComparisonResult:
    """配置比较完整结果"""

    topology_similarity: CompareResult
    cmd_list_similarity: CompareResult
    tele_handler_similarity: CompareResult
    internal_handler_similarity: CompareResult
    config_similarity: float  # 加权总分

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_similarity": {
                "score": self.topology_similarity.score,
                "details": self.topology_similarity.details,
                "issues": self.topology_similarity.issues,
            },
            "cmd_list_similarity": {
                "score": self.cmd_list_similarity.score,
                "details": self.cmd_list_similarity.details,
                "issues": self.cmd_list_similarity.issues,
            },
            "tele_handler_similarity": {
                "score": self.tele_handler_similarity.score,
                "details": self.tele_handler_similarity.details,
                "issues": self.tele_handler_similarity.issues,
            },
            "internal_handler_similarity": {
                "score": self.internal_handler_similarity.score,
                "details": self.internal_handler_similarity.details,
                "issues": self.internal_handler_similarity.issues,
            },
            "config_similarity": self.config_similarity,
        }


class ConfigComparator:
    """配置比较器"""

    # 权重配置
    WEIGHTS = {
        "topology": 0.20,
        "cmd_list": 0.30,
        "tele_handler": 0.30,
        "internal_handler": 0.20,
    }

    # handler 名前缀在 __init__ 中从配置内容动态推断（_infer_handler_prefixes）

    def _normalize_handler_name(self, name: str) -> str:
        """去掉协议前缀，统一比较"""
        name = name.strip().lower()
        for prefix in self._handler_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.strip("_")

    # 期望的第二层字段
    EXPECTED_TOP_FIELDS = [
        "version",
        "author",
        "start_index",
        "cmd_list",
        "tele_handler",
        "internal_handler",
    ]

    # 期望的列表元素字段
    EXPECTED_CMD_FIELDS = ["desc", "cmd", "handler"]
    EXPECTED_TELE_FIELDS = ["desc", "len", "handler"]
    EXPECTED_INTERNAL_FIELDS = ["desc", "len", "handler"]

    def __init__(self, gen_config: Dict[str, Any], gt_config: Dict[str, Any]):
        """
        初始化比较器

        Args:
            gen_config: 生成的配置
            gt_config: 真实配置（ground truth）
        """
        self.gen = gen_config
        self.gt = gt_config
        # 协议前缀从两份配置的 handler 名动态推断（而非硬编码 dllg_/gdwx_），
        # 以支持任意目标协议名称
        self._handler_prefixes = self._infer_handler_prefixes()

    def _infer_handler_prefixes(self) -> List[str]:
        """从 handler 名推断协议前缀（取首个下划线前的段，如 dllg_xxx → dllg_）"""
        prefixes = set()
        for cfg in (self.gen, self.gt):
            for section in ("cmd_list", "tele_handler", "internal_handler"):
                for item in cfg.get(section, []) or []:
                    handler = str(item.get("handler", ""))
                    if "_" in handler:
                        prefixes.add(handler.split("_", 1)[0] + "_")
        # 长的前缀优先匹配，避免 dllg_ 被 dllg 吃掉
        return sorted(prefixes, key=len, reverse=True)

    def compare(self) -> ConfigComparisonResult:
        """
        执行完整比较

        Returns:
            ConfigComparisonResult 包含所有维度的相似度
        """
        logger.info("开始配置比较...")

        # 1. 顶层树结构相似度
        topology_result = self._compare_topology()
        logger.info(f"拓扑相似度: {topology_result.score:.2f}")

        # 2. 命令表相似度
        cmd_result = self._compare_cmd_list()
        logger.info(f"命令表相似度: {cmd_result.score:.2f}")

        # 3. 遥测表相似度
        tele_result = self._compare_tele_handler()
        logger.info(f"遥测表相似度: {tele_result.score:.2f}")

        # 4. 内部处理表相似度
        internal_result = self._compare_internal_handler()
        logger.info(f"内部处理相似度: {internal_result.score:.2f}")

        # 5. 计算总分
        config_similarity = (
            topology_result.score * self.WEIGHTS["topology"]
            + cmd_result.score * self.WEIGHTS["cmd_list"]
            + tele_result.score * self.WEIGHTS["tele_handler"]
            + internal_result.score * self.WEIGHTS["internal_handler"]
        )

        logger.info(f"配置总相似度: {config_similarity:.2f}")

        return ConfigComparisonResult(
            topology_similarity=topology_result,
            cmd_list_similarity=cmd_result,
            tele_handler_similarity=tele_result,
            internal_handler_similarity=internal_result,
            config_similarity=config_similarity,
        )

    def _compare_topology(self) -> CompareResult:
        """
        比较顶层树结构相似度

        检查四层树结构：
        - 第一层：根节点（应为 object）
        - 第二层：6个期望字段
        - 第三层：列表及其元素类型
        - 第四层：列表元素内部字段
        """
        score = 0.0
        max_score = 100.0
        details = {}
        issues = []

        # 第一层：根节点检查
        if not isinstance(self.gen, dict):
            issues.append("根节点不是对象/dict")
            return CompareResult(score=0.0, details={"layer": 1}, issues=issues)

        score += 10.0  # 根节点是 object

        # 第二层：检查6个期望字段
        present_fields = []
        missing_fields = []
        for field in self.EXPECTED_TOP_FIELDS:
            if field in self.gen:
                present_fields.append(field)
            else:
                missing_fields.append(field)

        field_score = (len(present_fields) / len(self.EXPECTED_TOP_FIELDS)) * 30.0
        score += field_score

        details["present_fields"] = present_fields
        details["missing_fields"] = missing_fields

        if missing_fields:
            issues.append(f"缺失第二层字段: {missing_fields}")

        # 第三层：检查列表类型和非空
        list_checks = {}
        valid_lists = 0

        for name in ["cmd_list", "tele_handler", "internal_handler"]:
            val = self.gen.get(name)
            if isinstance(val, list) and len(val) > 0:
                list_checks[name] = True
                valid_lists += 1
            elif isinstance(val, list):
                list_checks[name] = False
                issues.append(f"{name} 为空列表")
            else:
                list_checks[name] = False
                if name not in self.gen:
                    issues.append(f"{name} 不存在")
                else:
                    issues.append(f"{name} 不是列表类型")

        score += (valid_lists / 3) * 30.0

        details["list_type_check"] = list_checks

        # 第四层：检查列表元素字段
        element_field_score = 0.0
        element_checks = {}

        for list_name, expected_fields in [
            ("cmd_list", self.EXPECTED_CMD_FIELDS),
            ("tele_handler", self.EXPECTED_TELE_FIELDS),
            ("internal_handler", self.EXPECTED_INTERNAL_FIELDS),
        ]:
            lst = self.gen.get(list_name, [])
            if not lst or not isinstance(lst, list):
                element_checks[list_name] = {"error": "列表不存在或为空"}
                continue

            # 检查第一个元素的结构
            first_elem = lst[0] if lst else {}
            if isinstance(first_elem, dict):
                elem_fields = list(first_elem.keys())
                present = [f for f in expected_fields if f in elem_fields]
                element_checks[list_name] = {
                    "expected": expected_fields,
                    "present": present,
                    "missing": [f for f in expected_fields if f not in elem_fields],
                }
                element_field_score += (len(present) / len(expected_fields)) * 10.0 / 3
            else:
                element_checks[list_name] = {"error": "元素不是对象"}
                issues.append(f"{list_name} 的元素不是对象/dict")

        score += element_field_score
        details["element_field_check"] = element_checks

        return CompareResult(
            score=min(score, max_score),
            details=details,
            issues=issues,
        )

    def _compare_cmd_list(self) -> CompareResult:
        """比较命令表相似度"""
        gen_list = self.gen.get("cmd_list", [])
        gt_list = self.gt.get("cmd_list", [])

        if not isinstance(gen_list, list) or not isinstance(gt_list, list):
            return CompareResult(
                score=0.0,
                details={"error": "cmd_list 不是列表"},
                issues=["cmd_list 类型错误"],
            )

        # 转换为以 cmd 为 key 的字典（统一转为 int 避免类型不匹配）
        def _normalize_cmd_key(item):
            v = item.get("cmd")
            try:
                return int(v)
            except (ValueError, TypeError):
                return v

        gen_cmds = {_normalize_cmd_key(item): item for item in gen_list if isinstance(item, dict)}
        gt_cmds = {_normalize_cmd_key(item): item for item in gt_list if isinstance(item, dict)}

        details = {
            "gen_count": len(gen_list),
            "gt_count": len(gt_list),
            "cmd_comparisons": [],
        }
        issues = []

        # 数量检查
        if len(gen_list) != len(gt_list):
            issues.append(f"命令数量不一致: 生成 {len(gen_list)} vs 真实 {len(gt_list)}")

        total_score = 0.0
        matched_count = 0

        for cmd, gt_item in gt_cmds.items():
            if cmd not in gen_cmds:
                issues.append(f"缺失命令: {cmd}")
                continue

            gen_item = gen_cmds[cmd]
            comparison = self._compare_cmd_item(gen_item, gt_item, cmd)
            details["cmd_comparisons"].append(comparison)
            total_score += comparison["score"]
            matched_count += 1

        # 计算平均分
        if matched_count > 0:
            avg_score = total_score / len(gt_cmds)  # 以真实命令数为分母
        else:
            avg_score = 0.0

        # 数量覆盖率惩罚：生成不足按比例扣分
        if len(gt_list) > 0:
            count_ratio = min(len(gen_list), len(gt_list)) / len(gt_list)
            avg_score *= count_ratio

        return CompareResult(
            score=max(0.0, avg_score),
            details=details,
            issues=issues,
        )

    def _compare_cmd_item(
        self, gen_item: Dict, gt_item: Dict, cmd: str
    ) -> Dict[str, Any]:
        """比较单个命令项"""
        result = {"cmd": cmd, "score": 0.0, "checks": {}}

        # handler 匹配（支持归一化：去掉协议前缀后比较）
        gen_handler = str(gen_item.get("handler", "")).strip()
        gt_handler = str(gt_item.get("handler", "")).strip()
        handler_match = (
            gen_handler == gt_handler
            or self._normalize_handler_name(gen_handler) == self._normalize_handler_name(gt_handler)
        )
        result["checks"]["handler"] = {
            "gen": gen_handler,
            "gt": gt_handler,
            "match": handler_match,
        }

        # desc 近似匹配
        gen_desc = gen_item.get("desc", "")
        gt_desc = gt_item.get("desc", "")
        desc_similarity = fuzz.ratio(str(gen_desc), str(gt_desc))
        result["checks"]["desc"] = {
            "gen": gen_desc,
            "gt": gt_desc,
            "similarity": desc_similarity,
        }

        # 计算单项得分
        # handler 完全匹配: 50分
        # desc 相似度: 50分 (按相似度比例)
        score = 0.0
        if handler_match:
            score += 50.0
        score += (desc_similarity / 100.0) * 50.0

        result["score"] = score
        return result

    def _compare_tele_handler(self) -> CompareResult:
        """比较遥测表相似度"""
        gen_list = self.gen.get("tele_handler", [])
        gt_list = self.gt.get("tele_handler", [])

        return self._compare_handler_list(
            gen_list, gt_list, "tele_handler", ["handler", "len", "desc"]
        )

    def _compare_internal_handler(self) -> CompareResult:
        """比较内部处理表相似度"""
        gen_list = self.gen.get("internal_handler", [])
        gt_list = self.gt.get("internal_handler", [])

        return self._compare_handler_list(
            gen_list, gt_list, "internal_handler", ["handler", "len", "desc"]
        )

    def _compare_handler_list(
        self,
        gen_list: List,
        gt_list: List,
        list_name: str,
        fields: List[str],
    ) -> CompareResult:
        """通用列表比较逻辑"""
        if not isinstance(gen_list, list) or not isinstance(gt_list, list):
            return CompareResult(
                score=0.0,
                details={"error": f"{list_name} 不是列表"},
                issues=[f"{list_name} 类型错误"],
            )

        # 以 handler 为 key 构建字典（支持归一化匹配，去掉协议前缀）
        gen_items: Dict[str, Dict] = {}
        for item in gen_list:
            if isinstance(item, dict):
                raw_handler = (item.get("handler") or "").strip()
                gen_items[raw_handler] = item
                norm = self._normalize_handler_name(raw_handler)
                if norm != raw_handler:
                    gen_items[norm] = item
        gt_items = {(item.get("handler") or "").strip(): item for item in gt_list if isinstance(item, dict)}

        details = {
            "gen_count": len(gen_list),
            "gt_count": len(gt_list),
            "item_comparisons": [],
        }
        issues = []

        if len(gen_list) != len(gt_list):
            issues.append(f"{list_name} 数量不一致: 生成 {len(gen_list)} vs 真实 {len(gt_list)}")

        total_score = 0.0
        matched_count = 0

        for handler, gt_item in gt_items.items():
            if handler not in gen_items:
                issues.append(f"缺失 {list_name}: {handler}")
                continue

            gen_item = gen_items[handler]
            comparison = self._compare_handler_item(gen_item, gt_item, handler, fields)
            details["item_comparisons"].append(comparison)
            total_score += comparison["score"]
            matched_count += 1

        if matched_count > 0:
            avg_score = (total_score / len(gt_items)) if gt_items else 0.0
        else:
            avg_score = 0.0

        # 数量覆盖率惩罚：生成不足按比例扣分
        if len(gt_list) > 0:
            count_ratio = min(len(gen_list), len(gt_list)) / len(gt_list)
            avg_score *= count_ratio

        return CompareResult(
            score=max(0.0, avg_score),
            details=details,
            issues=issues,
        )

    def _compare_handler_item(
        self,
        gen_item: Dict,
        gt_item: Dict,
        handler: str,
        fields: List[str],
    ) -> Dict[str, Any]:
        """比较单个 handler 项"""
        result = {"handler": handler, "score": 0.0, "checks": {}}

        max_field_score = 100.0 / len(fields)
        total_score = 0.0

        for field in fields:
            gen_val = gen_item.get(field, "")
            gt_val = gt_item.get(field, "")

            if field == "desc":
                # desc 使用模糊匹配
                similarity = fuzz.ratio(str(gen_val), str(gt_val))
                field_score = (similarity / 100.0) * max_field_score
                result["checks"][field] = {
                    "gen": gen_val,
                    "gt": gt_val,
                    "similarity": similarity,
                    "score": field_score,
                }
            elif field == "len":
                # len 字段统一转为 int 比较
                try:
                    match = int(gen_val) == int(gt_val)
                except (ValueError, TypeError):
                    match = str(gen_val).strip() == str(gt_val).strip()
                field_score = max_field_score if match else 0.0
                result["checks"][field] = {
                    "gen": gen_val,
                    "gt": gt_val,
                    "match": match,
                    "score": field_score,
                }
            else:
                # handler 字段：支持归一化匹配
                gen_h = str(gen_val).strip()
                gt_h = str(gt_val).strip()
                match = (
                    gen_h == gt_h
                    or self._normalize_handler_name(gen_h) == self._normalize_handler_name(gt_h)
                )
                field_score = max_field_score if match else 0.0
                result["checks"][field] = {
                    "gen": gen_val,
                    "gt": gt_val,
                    "match": match,
                    "score": field_score,
                }

            total_score += field_score

        result["score"] = total_score
        return result


def compare_configs(
    gen_config_path: str,
    gt_config_path: str,
) -> ConfigComparisonResult:
    """
    便捷函数：比较两个配置文件

    Args:
        gen_config_path: 生成的配置文件路径
        gt_config_path: 真实配置文件路径

    Returns:
        ConfigComparisonResult 比较结果
    """
    with open(gen_config_path, "r", encoding="utf-8") as f:
        gen_config = json.load(f)

    with open(gt_config_path, "r", encoding="utf-8") as f:
        gt_config = json.load(f)

    comparator = ConfigComparator(gen_config, gt_config)
    return comparator.compare()
