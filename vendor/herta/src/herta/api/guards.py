"""路由输入校验守卫"""

import re

from fastapi import HTTPException

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def validate_name(value: str, field: str = "名称") -> str:
    """
    校验路径参数（source/target 等），防止 `..` 等路径穿越注入文件系统路径。
    仅允许字母、数字、下划线、短横线。
    """
    if not _NAME_RE.match(value or ""):
        raise HTTPException(status_code=400, detail=f"{field}含非法字符")
    return value
