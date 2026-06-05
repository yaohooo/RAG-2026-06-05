# 配置常量
DEFAULT_MODEL = "qwen3.6-plus"
SUPPORTED_MODELS = ["qwen3.6-plus", "qwen-turbo", "qwen-max"]
EMBEDDING_MODEL = "text-embedding-v2"
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# RAG 检索参数
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_RETRIEVE_K = 5
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_MIN_SCORE = 0.15
DEFAULT_TEMPERATURE = 0.3

# 文件处理配置
SUPPORTED_FILE_TYPES = [".txt", ".md", ".pdf", ".csv", ".svg", ".scvg"]
MIN_CHUNK_LENGTH = 20  # 最小文本块长度