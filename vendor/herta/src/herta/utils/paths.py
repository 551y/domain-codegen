"""
路径处理工具函数模块

提供目录创建、项目相对路径解析等功能
"""

import os
from pathlib import Path
from typing import Union, Optional


def ensure_dir(
    dir_path: Union[str, Path],
    mode: int = 0o755,
) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        dir_path: 目录路径（字符串或 Path 对象）
        mode: 目录权限模式，默认为 0o755

    Returns:
        目录的 Path 对象

    Raises:
        OSError: 创建目录失败时

    Example:
        >>> dir_path = ensure_dir("/path/to/new/dir")
        >>> print(dir_path.exists())  # True
    """
    path = Path(dir_path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True, mode=mode)
    elif not path.is_dir():
        raise NotADirectoryError(f"路径存在但不是目录: {path}")

    return path


def project_relative(
    relative_path: Union[str, Path],
    marker_file: str = "pyproject.toml",
    start_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    获取相对于项目根目录的完整路径

    通过向上查找 marker_file 来确定项目根目录

    Args:
        relative_path: 相对于项目根目录的路径
        marker_file: 用于识别项目根目录的标记文件，默认为 pyproject.toml
        start_dir: 搜索起始目录，默认为当前文件所在目录

    Returns:
        完整的绝对路径

    Raises:
        FileNotFoundError: 无法找到项目根目录时

    Example:
        >>> # 假设在项目根目录下
        >>> data_path = project_relative("data/config.json")
        >>> print(data_path)  # /full/path/to/project/data/config.json
    """
    # 确定搜索起始点
    if start_dir is None:
        # 默认从调用者所在文件目录开始
        # 如果没有 __file__，使用当前工作目录
        try:
            start = Path(__file__).resolve().parent
        except NameError:
            start = Path.cwd()
    else:
        start = Path(start_dir).resolve()

    # 向上查找标记文件
    current = start
    root = None

    while True:
        marker = current / marker_file
        if marker.exists():
            root = current
            break

        # 到达根目录，停止搜索
        parent = current.parent
        if parent == current:
            break
        current = parent

    if root is None:
        # 找不到标记文件，使用起始目录作为项目根
        root = start

    # 组合路径
    result = root / relative_path
    return result.resolve()


def get_data_dir(
    subdir: Optional[str] = None,
    data_root: str = "data",
) -> Path:
    """
    获取数据目录路径

    Args:
        subdir: 数据子目录名称
        data_root: 数据根目录名称，默认为 "data"

    Returns:
        数据目录的 Path 对象

    Example:
        >>> source_dir = get_data_dir("source_proto")
        >>> print(source_dir)  # /project/data/source_proto
    """
    base = project_relative(data_root)

    if subdir:
        path = base / subdir
        ensure_dir(path)
        return path

    return base


def get_output_dir(
    subdir: Optional[str] = None,
    output_root: str = "output",
) -> Path:
    """
    获取输出目录路径

    Args:
        subdir: 输出子目录名称
        output_root: 输出根目录名称，默认为 "output"

    Returns:
        输出目录的 Path 对象

    Example:
        >>> result_dir = get_output_dir("results")
        >>> print(result_dir)  # /project/output/results
    """
    base = project_relative(output_root)

    if subdir:
        path = base / subdir
        ensure_dir(path)
        return path

    return base
