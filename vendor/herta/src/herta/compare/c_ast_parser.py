"""
C 代码 AST 解析模块

使用 tree-sitter 解析 C 文件，提取函数定义信息：
- 函数名
- 返回类型
- 参数列表
- 函数体源码
- 函数体 AST
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tree_sitter import Language, Parser, Tree, Node
    from tree_sitter_c import language as c_language
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # 创建占位类以避免导入错误
    class Node:
        pass
    class Tree:
        pass

from herta.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FunctionInfo:
    """函数信息数据类"""

    name: str
    return_type: str
    parameters: List[Dict[str, str]]  # [{"type": "int", "name": "x"}, ...]
    body_source: str  # 函数体源码
    ast: Any  # 函数体 AST 节点
    full_source: str  # 完整函数定义源码

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "body_source": self.body_source[:200] + "..." if len(self.body_source) > 200 else self.body_source,
            "full_source": self.full_source[:200] + "..." if len(self.full_source) > 200 else self.full_source,
        }


class CASTParser:
    """
    C 代码 AST 解析器

    使用 tree-sitter 解析 C 文件，提取函数定义。
    """

    def __init__(self):
        """初始化解析器"""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter 未安装。请执行: pip install tree-sitter tree-sitter-c"
            )

        self.language = Language(c_language())
        self.parser = Parser(self.language)

    def parse_file(self, file_path: str) -> List[FunctionInfo]:
        """
        解析 C 文件，提取所有函数

        Args:
            file_path: C 文件路径

        Returns:
            FunctionInfo 列表
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        source = path.read_text(encoding="utf-8")
        return self.parse_source(source)

    def parse_source(self, source: str) -> List[FunctionInfo]:
        """
        解析 C 源码字符串

        Args:
            source: C 源码

        Returns:
            FunctionInfo 列表
        """
        tree = self.parser.parse(source.encode("utf-8"))
        root_node = tree.root_node

        functions = []
        for node in root_node.children:
            if node.type == "function_definition":
                func_info = self._extract_function(node, source)
                if func_info:
                    functions.append(func_info)

        logger.info(f"解析到 {len(functions)} 个函数")
        return functions

    def _extract_function(self, node: Node, source: str) -> Optional[FunctionInfo]:
        """
        从 AST 节点提取函数信息

        Args:
            node: function_definition 节点
            source: 完整源码

        Returns:
            FunctionInfo 或 None
        """
        try:
            # 提取函数名
            name = self._extract_function_name(node)
            if not name:
                return None

            # 提取返回类型
            return_type = self._extract_return_type(node, source)

            # 提取参数列表
            parameters = self._extract_parameters(node, source)

            # 提取函数体
            body_node = self._find_child_by_type(node, "compound_statement")
            body_source = self._node_to_source(body_node, source) if body_node else ""

            # 完整函数源码
            full_source = self._node_to_source(node, source)

            return FunctionInfo(
                name=name,
                return_type=return_type,
                parameters=parameters,
                body_source=body_source,
                ast=body_node,
                full_source=full_source,
            )

        except Exception as e:
            logger.warning(f"提取函数信息失败: {e}")
            return None

    def _extract_function_name(self, node: Node) -> Optional[str]:
        """提取函数名"""
        # 函数定义结构: (function_definition
        #   (type) 返回类型
        #   (function_declarator
        #     (identifier) 函数名
        #     (parameter_list) 参数
        #   )
        #   (compound_statement) 函数体
        # )

        declarator = self._find_child_by_type(node, "function_declarator")
        if not declarator:
            return None

        # 在 function_declarator 中查找 identifier
        for child in declarator.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            # 处理指针函数的情况
            if child.type == "pointer_declarator":
                # 递归查找 identifier
                name = self._find_identifier_in_node(child)
                if name:
                    return name

        return None

    def _find_identifier_in_node(self, node: Node) -> Optional[str]:
        """在节点中递归查找 identifier"""
        if node.type == "identifier":
            text = node.text
            return text.decode("utf-8") if isinstance(text, bytes) else text

        for child in getattr(node, "children", []):
            name = self._find_identifier_in_node(child)
            if name:
                return name
        return None

    def _extract_return_type(self, node: Node, source: str) -> str:
        """提取返回类型"""
        # 查找类型说明符
        type_node = self._find_child_by_type(node, "primitive_type") or \
                   self._find_child_by_type(node, "type_identifier") or \
                   self._find_child_by_types(node, ["type_qualifier", "sized_type_specifier"])

        if type_node:
            return self._node_to_source(type_node, source).strip()

        # 尝试查找 declarator 前面的类型
        for i, child in enumerate(node.children):
            if child.type == "function_declarator":
                # 前面的兄弟节点应该是类型
                if i > 0:
                    type_parts = []
                    for j in range(i):
                        type_parts.append(self._node_to_source(node.children[j], source))
                    return " ".join(type_parts).strip()

        return "void"  # 默认返回类型

    def _extract_parameters(self, node: Node, source: str) -> List[Dict[str, str]]:
        """提取参数列表"""
        parameters = []

        declarator = self._find_child_by_type(node, "function_declarator")
        if not declarator:
            return parameters

        param_list = self._find_child_by_type(declarator, "parameter_list")
        if not param_list:
            return parameters

        for param in param_list.children:
            if param.type == "parameter_declaration":
                param_info = self._parse_parameter(param, source)
                if param_info:
                    parameters.append(param_info)

        return parameters

    def _parse_parameter(self, node: Node, source: str) -> Optional[Dict[str, str]]:
        """解析单个参数"""
        param_type = ""
        param_name = ""

        for child in node.children:
            if child.type in ["primitive_type", "type_identifier", "sized_type_specifier"]:
                param_type = self._node_to_source(child, source).strip()
            elif child.type == "identifier":
                text = child.text
                param_name = text.decode("utf-8") if isinstance(text, bytes) else text
            elif child.type == "pointer_declarator":
                # 指针参数
                param_type += "*"
                name = self._find_identifier_in_node(child)
                if name:
                    param_name = name
            elif child.type == "abstract_pointer_declarator":
                param_type += "*"

        if param_type or param_name:
            return {"type": param_type or "unknown", "name": param_name or "unknown"}
        return None

    def _find_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        """查找指定类型的子节点"""
        for child in getattr(node, "children", []):
            if child.type == type_name:
                return child
        return None

    def _find_child_by_types(self, node: Node, type_names: List[str]) -> Optional[Node]:
        """查找指定类型列表中的任一类型的子节点"""
        for child in getattr(node, "children", []):
            if child.type in type_names:
                return child
        return None

    def _node_to_source(self, node: Node, source: str) -> str:
        """将 AST 节点转换回源码字符串"""
        if node is None:
            return ""

        start_byte = node.start_byte
        end_byte = node.end_byte

        if isinstance(source, str):
            source_bytes = source.encode("utf-8")
        else:
            source_bytes = source

        return source_bytes[start_byte:end_byte].decode("utf-8")

    def get_function_ast(self, func_info: FunctionInfo) -> Optional[Node]:
        """获取函数的 AST 节点"""
        return func_info.ast


def parse_c_functions(file_path: str) -> List[FunctionInfo]:
    """
    便捷函数：解析 C 文件提取函数

    Args:
        file_path: C 文件路径

    Returns:
        FunctionInfo 列表
    """
    parser = CASTParser()
    return parser.parse_file(file_path)
