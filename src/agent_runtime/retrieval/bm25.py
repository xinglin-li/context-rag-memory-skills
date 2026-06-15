# src/agent_runtime/retrieval/bm25.py
import math
from typing import List, Dict, Set
from agent_runtime.retrieval.models import Chunk, RetrievalHit
from agent_runtime.retrieval.tokenizer import SimpleTokenizer

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, Chunk] = {}
        
        # 核心数据结构
        self.inverted_index: Dict[str, Set[str]] = {} # token -> set(chunk_ids)
        self.chunk_tf: Dict[str, Dict[str, int]] = {}   # chunk_id -> {token: count}
        self.chunk_lengths: Dict[str, int] = {}       # chunk_id -> length (in tokens)
        
        # 预计算全局元数据
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        
    def fit(self, chunks: List[Chunk]):
        self.chunks = {c.chunk_id: c for c in chunks}
        total_tokens_all_docs = 0
        
        # 1. 建立倒排索引与统计频次
        for chunk in chunks:
            tokens = SimpleTokenizer.tokenize(chunk.text)
            self.chunk_lengths[chunk.chunk_id] = len(tokens)
            total_tokens_all_docs += len(tokens)
            
            self.chunk_tf[chunk.chunk_id] = {}
            for token in tokens:
                # 更新 Chunk 内部词频 (TF)
                self.chunk_tf[chunk.chunk_id][token] = self.chunk_tf[chunk.chunk_id].get(token, 0) + 1
                
                # 更新全局倒排表
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(chunk.chunk_id)
                
        num_docs = len(chunks)
        self.avgdl = total_tokens_all_docs / num_docs if num_docs > 0 else 0.0
        
        # 2. 预计算每一个 Token 的标准 IDF (含有平滑项)
        for token, containing_chunks in self.inverted_index.items():
            n_q = len(containing_chunks)
            # 标准鲁棒 IDF 变体公式
            self.idf[token] = math.log((num_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        query_tokens = SimpleTokenizer.tokenize(query)
        scores: Dict[str, float] = {}
        
        if not query_tokens or not self.chunks:
            return []

        # 评分核心逻辑
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            # 只遍历包含该关键词的“倒排候选集”，拒绝全库爆破，极大提升检索性能
            target_chunk_ids = self.inverted_index.get(token, set())
            for chunk_id in target_chunk_ids:
                tf = self.chunk_tf[chunk_id].get(token, 0)
                doc_len = self.chunk_lengths[chunk_id]
                
                # BM25 核心精妙公式项：结合了长度惩罚与饱和度抑制
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                
                scores[chunk_id] = scores.get(chunk_id, 0.0) + self.idf[token] * (numerator / denominator)
                
        # 3. 排序与包装结果
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        hits = []
        for rank, (chunk_id, score) in enumerate(sorted_scores, start=1):
            chunk = self.chunks[chunk_id]
            hits.append(RetrievalHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=round(score, 4),
                rank=rank
            ))
        return hits