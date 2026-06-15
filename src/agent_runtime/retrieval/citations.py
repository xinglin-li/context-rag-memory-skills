# src/agent_runtime/retrieval/citations.py
from typing import List, Dict, Any, Tuple

class CitationBuilder:
    @staticmethod
    def build_citations(top_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        由应用层确信地接管引用链的序列化。
        返回给模型的包装文本，以及用于后续 Trace 审计的地图。
        """
        formatted_evidence_blocks = []
        citation_map = {}
        
        for idx, chunk in enumerate(top_chunks, start=1):
            citation_id = f"Doc-{idx}"
            formatted_evidence_blocks.append(
                f"[{citation_id}] (Source Document ID: {chunk['document_id']})\nContent: {chunk['text']}"
            )
            citation_map[citation_id] = chunk
            
        evidence_text = "\n\n".join(formatted_evidence_blocks)
        return evidence_text, citation_map