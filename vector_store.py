import numpy as np
import math
from collections import Counter
from text_utils import tokenize_mixed


class HybridVectorStore:
    def __init__(self):
        self.chunks = []          # 原始文本块
        self.embeddings = []      # 向量嵌入
        self.sources = []         # 来源文件名
        self.tokenized = []       # BM25 分词结果

    def add_chunk(self, chunk, embedding, source=""):
        """添加文本块、向量、来源信息"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
        self.sources.append(source)
        # BM25 预处理：中英文混合分词
        self.tokenized.append(tokenize_mixed(chunk))

    def _cosine_similarities(self, query_embedding):
        """计算余弦相似度"""
        if not self.embeddings:
            return np.array([])
        matrix = np.array(self.embeddings)
        q_vec = np.array(query_embedding)
        dot_product = np.dot(matrix, q_vec)
        norm_matrix = np.linalg.norm(matrix, axis=1)
        norm_q = np.linalg.norm(q_vec)
        return dot_product / (norm_matrix * norm_q + 1e-8)

    def _bm25_scores(self, query, k1=1.5, b=0.75):
        """计算BM25关键词相似度分数"""
        if not self.tokenized:
            return np.array([])

        query_tokens = tokenize_mixed(query)
        doc_lengths = [len(t) for t in self.tokenized]
        avg_doc_len = np.mean(doc_lengths) if doc_lengths else 1
        N = len(self.tokenized)

        # 计算 IDF
        doc_freq = Counter()
        for tokens in self.tokenized:
            for token in set(tokens):
                if token in query_tokens:
                    doc_freq[token] += 1

        idf = {}
        for token in query_tokens:
            df = doc_freq.get(token, 0)
            idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        # 计算 BM25 分数
        scores = np.zeros(N)
        for i, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            score = 0
            for token in query_tokens:
                if token in tf:
                    freq = tf[token]
                    denom = freq + k1 * (1 - b + b * doc_lengths[i] / avg_doc_len)
                    score += idf[token] * freq * (k1 + 1) / denom
            scores[i] = score
        return scores

    def similarity_search(self, query, query_embedding, k=4, vector_weight=0.7, min_score=0.15):
        """混合检索（向量+BM25）"""
        if not self.embeddings:
            return []

        # 1. 向量余弦相似度
        vec_scores = self._cosine_similarities(query_embedding)

        # 2. BM25 关键词打分
        bm25_raw = self._bm25_scores(query)

        # 3. 归一化 BM25 分数到 [0, 1]
        bm25_max = bm25_raw.max() if bm25_raw.size > 0 else 1
        bm25_scores = bm25_raw / (bm25_max + 1e-8) if bm25_max > 0 else bm25_raw

        # 4. 加权融合
        combined = vector_weight * vec_scores + (1 - vector_weight) * bm25_scores

        # 5. 按综合得分排序，过滤低分结果
        scored_indices = [(i, combined[i]) for i in range(len(combined))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # 6. 去重：合并高度相似的相邻文本块（来自同一文档的连续块）
        results = []
        seen_chunks = set()
        for idx, score in scored_indices:
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            # 简单去重：完全相同的文本块
            if chunk in seen_chunks:
                continue
            seen_chunks.add(chunk)
            results.append({
                "text": chunk,
                "source": self.sources[idx],
                "score": float(score),
                "vec_score": float(vec_scores[idx]),
                "bm25_score": float(bm25_scores[idx])
            })
            if len(results) >= k:
                break

        return results

    def clear(self):
        """清空向量库"""
        self.chunks = []
        self.embeddings = []
        self.sources = []
        self.tokenized = []

    @property
    def count(self):
        """返回文本块数量"""
        return len(self.chunks)