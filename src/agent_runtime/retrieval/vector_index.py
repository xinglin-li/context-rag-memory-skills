# src/agent_runtime/retrieval/vector_index.py
import math
from typing import List, Dict, Tuple, Any
from agent_runtime.retrieval.models import Chunk, RetrievalHit
from agent_runtime.retrieval.embeddings import BaseEmbeddingProvider

class VectorIndex:
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.provider = embedding_provider
        # 物理存储结构：内存字典映射
        self.storage: Dict[str, List[float]] = {} # chunk_id -> vector
        self.chunks: Dict[str, Chunk] = {}       # chunk_id -> Chunk
        
    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        """手写 L2 范数归一化，将向量长度缩放为 1.0"""
        sq_sum = sum(x ** 2 for x in vec)
        norm = math.sqrt(sq_sum)
        if norm == 0:
            return vec
        return [x / norm for x in vec]
    
    @staticmethod
    def _dot_product(vec_a: List[float], vec_b: List[float]) -> float:
        """手写两个向量的点积"""
        return sum(a * b for a, b in zip(vec_a, vec_b))
    
    def add_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            raw_vec = self.provider.get_embedding(chunk.text)
            # 预先进行 L2 归一化。这样后续检索时，余弦相似度将直接降维简化为极其高效的点积计算
            normalized_vec = self._l2_normalize(raw_vec)
            
            self.storage[chunk.chunk_id] = normalized_vec
            self.chunks[chunk.chunk_id] = chunk
    
    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """基于语义空间余弦相似度的精准Top-K检索"""
        if not self.storage or not query:
            return []
        
        # 1. 提取 Query 向量并做相同的归一化
        query_raw = self.provider.get_embedding(query)
        query_vec = self._l2_normalize(query_raw)
        
        scored_hits: List[Tuple[str, float]] = []
        
        # 2. 遍历全库进行几何计算
        for chunk_id, chunk_vec in self.storage.items():
            # 因为两边都提前做了 L2 归一化，分母 ||A||*||B|| 必然等于 1.0
            # 余弦相似度直接等于点积！计算速度飙升
            similarity = self._dot_product(query_vec, chunk_vec)
            scored_hits.append((chunk_id, similarity))
        
        # 3. 降序重排
        sorted_hits = sorted(scored_hits, key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_hits, start=1):
            chunk = self.chunks[chunk_id]
            results.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "score": round(score, 4),
                "rank": rank
            })
        return results
        