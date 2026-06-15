# tests/test_bm25.py
import pytest
from agent_runtime.retrieval.models import Chunk
from agent_runtime.retrieval.bm25 import BM25Retriever

@pytest.fixture
def populated_retriever():
    chunks = [
        Chunk(chunk_id="c1", document_id="doc_arima", text="Throw a ValueError in ARIMA(1,1,1) integration model execution."),
        Chunk(chunk_id="c2", document_id="doc_backtest", text="Systematic backtesting workflow within XL-Systematic platform data integrity.")
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    return retriever

def test_precise_keyword_match(populated_retriever):
    """测试 1：精准变量/专有名词强匹配召回"""
    hits = populated_retriever.search("ARIMA(1,1,1)")
    assert len(hits) > 0
    # doc_arima 必须以压倒性最高分夺得榜首
    assert hits[0].document_id == "doc_arima"

def test_rare_vs_common_word_weight(populated_retriever):
    """测试 2：罕见词权重测试。'XL-Systematic'属于罕见词，'model'属于高频普通词"""
    # 这个提问同时包含了arima对应的model，和backtest对应的XL-Systematic
    query = "Where is the model of XL-Systematic?"
    hits = populated_retriever.search(query)
    
    # 因为 XL-Systematic 极其罕见，信息量巨大，其 IDF 权重应远高于 common 词 model
    # 索引首位应当精准召回包含罕见词的 c2
    assert hits[0].chunk_id == "c2"

def test_empty_and_no_match_query(populated_retriever):
    """测试 3：垃圾无效查询边界测试"""
    assert populated_retriever.search("") == []
    assert populated_retriever.search("completely unrelated garbage text") == []