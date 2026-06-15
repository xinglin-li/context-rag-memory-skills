# src/agent_runtime/retrieval/reranker.py
from typing import List, Dict, Any

class RuleBasedReranker:
    @staticmethod
    def rerank(fused_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        高精度业务洗牌：如果文档来自于我们极其信赖的核心时序代码库/规范，
        在 RRF 分数的基础上给予战略加分，确保核心合规条目优先进入 Context。
        """
        for hit in fused_hits:
            # 假设对特定的核心资产库文件给予额外权重加分
            if "arima" in hit["chunk_id"] or "backtest" in hit["chunk_id"]:
                hit["rrf_score"] += 0.05
                
        # 重新根据洗牌后的最终得分排序
        return sorted(fused_hits, key=lambda x: x["rrf_score"], reverse=True)