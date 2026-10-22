"""
Prompt 模板加载模块

负责加载和管理 Prompt 模板文件
"""

from pathlib import Path
from typing import Optional, Dict, Any
from string import Template

from herta.utils.paths import project_relative
from herta.logging import get_logger

logger = get_logger(__name__)


class PromptLoader:
    """
    Prompt 模板加载器

    从 prompts/ 目录加载模板文件，支持变量替换
    """

    DEFAULT_PROMPTS_DIR = "prompts"

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        初始化 Prompt 加载器

        Args:
            prompts_dir: Prompt 模板目录，默认使用项目根目录下的 prompts/
        """
        if prompts_dir is None:
            self.prompts_dir = project_relative(self.DEFAULT_PROMPTS_DIR)
        else:
            self.prompts_dir = Path(prompts_dir)

        logger.debug(f"Prompt 目录: {self.prompts_dir}")

    def load(self, template_name: str) -> str:
        """
        加载 Prompt 模板文件

        Args:
            template_name: 模板文件名（如 "generate_config.md"）

        Returns:
            模板内容字符串

        Raises:
            FileNotFoundError: 模板文件不存在
        """
        template_path = self.prompts_dir / template_name

        if not template_path.exists():
            # 尝试添加 .md 后缀
            if not template_name.endswith('.md'):
                template_path = self.prompts_dir / f"{template_name}.md"

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt 模板不存在: {template_path}")

        try:
            content = template_path.read_text(encoding='utf-8')
            logger.debug(f"加载模板: {template_path} ({len(content)} 字符)")
            return content
        except Exception as e:
            logger.error(f"加载模板失败: {template_path}, 错误: {e}")
            raise

    def load_and_render(self, template_name: str, variables: Dict[str, str]) -> str:
        """
        加载模板并替换变量

        Args:
            template_name: 模板文件名
            variables: 变量字典，用于替换模板中的占位符

        Returns:
            渲染后的 Prompt 字符串

        Example:
            >>> loader = PromptLoader()
            >>> prompt = loader.load_and_render(
            ...     "generate_config.md",
            ...     {"target_summary": "...", "context": "..."}
            ... )
        """
        template = self.load(template_name)

        try:
            # 使用 Python string.Template 进行变量替换
            t = Template(template)
            rendered = t.safe_substitute(variables)
            logger.debug(f"模板渲染完成: {template_name}")
            return rendered
        except Exception as e:
            logger.error(f"模板渲染失败: {template_name}, 错误: {e}")
            raise

    def list_templates(self) -> list[str]:
        """
        列出所有可用的模板文件

        Returns:
            模板文件名列表
        """
        if not self.prompts_dir.exists():
            return []

        templates = []
        for f in self.prompts_dir.iterdir():
            if f.is_file() and f.suffix in ['.md', '.txt']:
                templates.append(f.name)

        return sorted(templates)


def load_prompt(template_name: str, prompts_dir: Optional[Path] = None) -> str:
    """
    便捷函数：加载 Prompt 模板

    Args:
        template_name: 模板文件名
        prompts_dir: 模板目录（可选）

    Returns:
        模板内容字符串

    Example:
        >>> prompt = load_prompt("generate_config")
        >>> print(prompt[:100])
    """
    loader = PromptLoader(prompts_dir)
    return loader.load(template_name)


def render_prompt(template_name: str, variables: Dict[str, str], prompts_dir: Optional[Path] = None) -> str:
    """
    便捷函数：加载并渲染 Prompt 模板

    Args:
        template_name: 模板文件名
        variables: 变量字典
        prompts_dir: 模板目录（可选）

    Returns:
        渲染后的 Prompt 字符串

    Example:
        >>> prompt = render_prompt("generate_config", {"target_summary": "..."})
    """
    loader = PromptLoader(prompts_dir)
    return loader.load_and_render(template_name, variables)
