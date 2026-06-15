# tests/test_vector_index.py
import pytest
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.embeddings import FakeEmbeddingProvider
from agent_runtime.retrieval.vector_index import VectorIndex

def test_semantic_similarity_alignment():
    """测试情况 1：验证完全不包含相同字面量，但语义映射一致时的召回能力"""
    provider = FakeEmbeddingProvider(dimensions=4)
    index = VectorIndex(provider)
    
    chunks = [
        Chunk(chunk_id="c_time", document_id="d1", text="Mathematical modeling for historical time series data."),
        Chunk(chunk_id="c_trade", document_id="d2", text="Executing automated backtesting routines for systematic trading systems.")
    ]
    index.add_chunks(chunks)
    
    # 提问：完全不包含上面的原词，使用同义概念词改写
    query = "forecast with stochastic sequential datasets" # 属于时序范畴
    hits = index.search(query, top_k=1)
    
    assert len(hits) > 0
    # 即使一个字都没重合，高维语义空间应该能够精准判定 c_time 距离更近！
    assert hits[0]["chunk_id"] == "c_time"

def test_perfect_and_orthogonal_vectors():
    """测试情况 2：数学边界测试。完全一致的文本相似度必须等于或接近 1.0"""
    provider = FakeEmbeddingProvider(dimensions=4)
    index = VectorIndex(provider)
    
    chunk = Chunk(chunk_id="c_exact", document_id="d_ex", text="Arbitrary quantitative formula placeholder.")
    index.add_chunks([chunk])
    
    # 完全相同的查询
    hits = index.search("Arbitrary quantitative formula placeholder.", top_k=1)
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)