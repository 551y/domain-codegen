"""
常量定义模块
集中管理固定配置项和默认值
"""

# 智谱 AI 默认模型配置
DEFAULT_CHAT_MODEL = "glm-5.1"
DEFAULT_EMBEDDING_MODEL = "embedding-3"
DEFAULT_EMBEDDING_DIMENSIONS = 1024

# LLM 供应商
PROVIDER_ZAI = "zai"        # 智谱 AI（zai-sdk 原生）
PROVIDER_OPENAI = "openai"  # OpenAI 兼容协议（DeepSeek / 硅基流动 / Moonshot 等）
SUPPORTED_PROVIDERS = (PROVIDER_ZAI, PROVIDER_OPENAI)
DEFAULT_CHAT_PROVIDER = PROVIDER_ZAI
DEFAULT_EMBEDDING_PROVIDER = PROVIDER_ZAI

# 默认目录路径
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_KB_DIR = "kb"

# 环境变量前缀
ENV_PREFIX = "HERTA_"

# 文件扩展名
EXT_DOCX = ".docx"
EXT_PDF = ".pdf"
EXT_JSON = ".json"
EXT_C = ".c"
EXT_H = ".h"
EXT_MD = ".md"

# 编码设置
DEFAULT_ENCODING = "utf-8"

# 日志格式
LOG_FORMAT_CONSOLE = "%(message)s"
LOG_FORMAT_FILE = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"

# 向量检索默认参数
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7
