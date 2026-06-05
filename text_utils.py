import re
import jieba
import numpy as np
import pandas as pd
from pypdf import PdfReader
import io
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MIN_CHUNK_LENGTH


def tokenize_mixed(text):
    """中英文混合分词（支持英文bigram、中文搜索引擎模式）"""
    tokens = []
    english_run = []

    def flush_english():
        # 写入单个英文词
        for w in english_run:
            tokens.append(w.lower())
        # 连续英文词额外生成 bigram（如 "john smith"）增强人名匹配
        if len(english_run) >= 2:
            for i in range(len(english_run) - 1):
                tokens.append(f"{english_run[i].lower()} {english_run[i+1].lower()}")
        english_run.clear()

    # 正则匹配连续的英文字母、数字、连字符（如 "John Smith", "GPT-4", "C++"）
    for segment in re.split(r'([a-zA-Z0-9][\w\-+.]*)', text):
        if not segment:
            continue
        if re.match(r'^[a-zA-Z0-9]', segment):
            english_run.append(segment)
        else:
            flush_english()
            # 中文段：用 jieba 搜索引擎模式分词
            tokens.extend(jieba.cut_for_search(segment))
    flush_english()
    return tokens


def extract_text_from_file(uploaded_file):
    """根据文件类型提取文本（txt/md/pdf/csv/svg）"""
    file_name = uploaded_file.name.lower()
    content_bytes = uploaded_file.read()

    if file_name.endswith(('.txt', '.md')):
        return content_bytes.decode('utf-8', errors='ignore')

    elif file_name.endswith('.pdf'):
        reader = PdfReader(io.BytesIO(content_bytes))
        text = ""
        for page_num, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text += f"\n\n[第{page_num+1}页]\n{t}\n"
        return text

    elif file_name.endswith('.csv'):
        # 使用 utf-8-sig 去除 BOM 头，避免列名出现乱码
        df = pd.read_csv(io.BytesIO(content_bytes), encoding='utf-8-sig')
        # 返回表结构信息用于向量化，让 LLM 知道有这个表格及列名
        summary = f"数据表名称：{uploaded_file.name}\n"
        summary += f"包含列：{', '.join(df.columns)}\n"
        summary += f"总行数：{len(df)}\n"
        summary += "前 2 行数据样本:\n" + df.head(2).to_string()
        return summary, df  # 返回文本摘要 + 原始DataFrame

    elif file_name.endswith(('.svg', '.scvg')):
        svg_content = content_bytes.decode('utf-8', errors='ignore')
        text_nodes = re.findall(r'<text[^>]*>(.*?)</text>', svg_content, re.DOTALL)
        clean_texts = [re.sub(r'<[^>]+>', '', t).strip() for t in text_nodes if t.strip()]
        return "\n".join(clean_texts)

    return ""


def split_text_semantic(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """语义化拆分文本（按句子边界，保留重叠）"""
    # 清洗多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 按句子边界拆分（中文 + 英文标点 + 换行）
    sentences = re.split(r'(?<=[。！？.!?。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # 如果单个句子就超过 chunk_size，需要强制拆分
        if sent_len > chunk_size:
            # 先保存当前累积的块
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0

            # 对超长句子按字符切分，保留重叠
            for i in range(0, sent_len, chunk_size - chunk_overlap):
                chunks.append(sentence[i:i + chunk_size])
            continue

        # 合并句子到当前块
        if current_length + sent_len > chunk_size and current_chunk:
            # 当前块已满，保存
            chunks.append("".join(current_chunk))
            # 保留重叠部分：取当前块末尾的句子
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sent_len

    # 保存最后一个块
    if current_chunk:
        chunks.append("".join(current_chunk))

    # 去重并过滤过短的块
    seen = set()
    filtered = []
    for c in chunks:
        c = c.strip()
        if len(c) > MIN_CHUNK_LENGTH and c not in seen:
            filtered.append(c)
            seen.add(c)

    return filtered