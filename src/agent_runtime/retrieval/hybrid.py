# src/agent_runtime/retrieval/hybrid.py
from typing import List, Dict, Any

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, bm25_hits: List[Any], vector_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        利用 RRF 算法将关键词排序和语义排序完美融合，彻底消除量纲差异。
        bm25_hits 结构: RetrievalHit (内含 chunk_id)
        vector_hits 结构: dict (内含 chunk_id)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        # 1. 注入第一路：BM25 名次倒数
        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = {"chunk_id": cid, "document_id": hit.document_id, "text": hit.text}

        # 2. 注入第二路：Vector 名次倒数
        for rank, hit in enumerate(vector_hits, start=1):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = {"chunk_id": cid, "document_id": hit["document_id"], "text": hit["text"]}

        # 3. 根据融合后的 RRF 得分降序排列
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for rank, (cid, score) in enumerate(sorted_chunks, start=1):
            meta = chunk_lookup[cid]
            meta["rrf_score"] = round(score, 6)
            meta["combined_rank"] = rank
            fused_results.append(meta)
            
        return fused_results