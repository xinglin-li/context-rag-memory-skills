# src/agent_runtime/skills/selector.py
from typing import Dict, List, Optional

from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.vector_index import VectorIndex
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.models import ActivatedSkill, SkillMetadata


class SkillSelector:
    def __init__(self, loader: SkillLoader):
        self.loader = loader
        self.catalog: List[SkillMetadata] = self.loader.discover_catalog()
        self.skill_by_chunk_id: Dict[str, SkillMetadata] = {}

        self.bm25_retriever = BM25Retriever()
        self.embedding_provider = FakeEmbeddingProvider()
        self.vector_index = VectorIndex(self.embedding_provider)

        self._build_skill_indices()

    def _build_skill_indices(self) -> None:
        """Index lightweight skill metadata, never full SKILL.md bodies."""
        chunks = []
        for meta in self.catalog:
            chunk_id = f"skill_{meta.name}"
            text = f"Skill Name: {meta.name}. Description: {meta.description}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=meta.name,
                    text=text,
                    metadata={"skill_meta": meta},
                )
            )
            self.skill_by_chunk_id[chunk_id] = meta

        if not chunks:
            return

        self.bm25_retriever.fit(chunks)
        self.vector_index.add_chunks(chunks)

    def select_and_activate(self, query: str) -> Optional[ActivatedSkill]:
        """
        Select a skill from lightweight metadata, then load the full skill only
        after a match is established.
        """
        if not self.catalog:
            return None

        lowered_query = query.lower()

        # Fast path: explicit skill names should always win.
        for meta in self.catalog:
            if meta.name.lower() in lowered_query:
                return self.loader.load_full_skill(meta)

        bm25_results = self.bm25_retriever.search(query, top_k=2)
        if not bm25_results:
            return None

        vector_results = self.vector_index.search(query, top_k=2)
        score_map: Dict[str, float] = {}

        for rank, hit in enumerate(bm25_results):
            score_map[hit.chunk_id] = score_map.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)

        for rank, hit in enumerate(vector_results):
            chunk_id = hit["chunk_id"]
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        best_chunk_id = max(score_map, key=score_map.get)
        meta = self.skill_by_chunk_id.get(best_chunk_id)
        if meta is None:
            return None

        return self.loader.load_full_skill(meta)


SemanticSkillSelector = SkillSelector
