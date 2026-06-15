# tests/test_hybrid_retrieval.py
import pytest
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.bm25 import BM25Retriever
from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.vector_index import VectorIndex
from agent_runtime.retrieval.pipeline import HybridRetrievalPipeline

@pytest.fixture
def initialized_pipeline():
    # 物理注入测试基础数据
    chunks = [
        Chunk(chunk_id="c_arima", document_id="doc_1", text="ValueError occurred during training standard ARIMA forecast series."),
        Chunk(chunk_id="c_backtest", document_id="doc_2", text="Evaluating data leakage boundaries using rigorous quantitative backtesting.")
    ]
    
    bm25 = BM25Retriever()
    bm25.fit(chunks)
    
    v_index = VectorIndex(FakeEmbeddingProvider(dimensions=4))
    v_index.add_chunks(chunks)
    
    return HybridRetrievalPipeline(bm25, v_index)

def test_hybrid_pipeline_dedup_and_rank(initialized_pipeline):
    """测试情况 1：两路同时命中同一个 Chunk 时，RRF 应该能正确提升其排名，且绝不产生重复项"""
    # 这个提问同时包含 arima 的字面量和训练特征
    query = "ValueError in ARIMA series"
    evidence_text, citation_map = initialized_pipeline.execute_pipeline(query, top_k=1)
    
    assert evidence_text != "insufficient_evidence"
    # c_arima 必须在两路高分加持下胜出
    assert "Doc-1" in citation_map
    assert citation_map["Doc-1"]["chunk_id"] == "c_arima"

def test_insufficient_evidence_circuit_breaker(initialized_pipeline):
    """测试情况 2：致命黑天鹅测试。面对毫无根据的完全无关提问，系统必须守住防线，直接报 insufficient_evidence"""
    query = "How to make spicy noodle soup in kitchen?"
    evidence_text, citation_map = initialized_pipeline.execute_pipeline(query, top_k=2)
    
    # 锁死底线，拒绝回答
    assert evidence_text == "insufficient_evidence"
    assert citation_map == {}