# src/agent_runtime/retrieval/pipeline.py
from typing import List, Dict, Any, Tuple
from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.vector_index import VectorIndex
from agent_runtime.retrieval.hybrid import ReciprocalRankFusion
from agent_runtime.retrieval.reranker import RuleBasedReranker
from agent_runtime.retrieval.citations import CitationBuilder

class HybridRetrievalPipeline:
    def __init__(self, bm25_retriever: BM25Retriever, vector_index: VectorIndex, vector_only_min_score: float = 0.75):
        self.bm25 = bm25_retriever
        self.v_index = vector_index
        self.fusion = ReciprocalRankFusion()
        self.vector_only_min_score = vector_only_min_score

    def execute_pipeline(self, query: str, top_k: int = 2) -> Tuple[str, Dict[str, Any]]:
        """端到端全链路闭环，内含无证据防线"""
        # 1. 触发双路召回
        bm25_hits = self.bm25.search(query, top_k=5)
        vector_hits = self.v_index.search(query, top_k=5)
        
        # 核心硬防护 1：Groundedness 铜墙铁壁。如果两路召回皆没有命中任何有效数据，
        # 说明查询已经彻底超出系统已知边界，严禁让大模型拍脑袋瞎编。直接熔断。
        if not bm25_hits and not vector_hits:
            return "insufficient_evidence", {}

        # 核心硬防护 2：只有向量召回命中时，必须要求极高的语义相似度，
        # 否则伪向量空间中的随机近邻会把无关问题错误放行。
        if not bm25_hits and vector_hits:
            top_vector_score = vector_hits[0].get("score", 0.0)
            if top_vector_score < self.vector_only_min_score:
                return "insufficient_evidence", {}
            
        # 2. 排序融合
        fused = self.fusion.fuse(bm25_hits, vector_hits)
        
        # 3. 重排洗牌
        reranked = RuleBasedReranker.rerank(fused)
        
        # 4. 截取 Top-K 并建立确定性引用
        final_candidates = reranked[:top_k]
        evidence_text, citation_map = CitationBuilder.build_citations(final_candidates)
        
        return evidence_text, citation_map