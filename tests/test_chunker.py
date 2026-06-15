# tests/test_chunker.py
from agent_runtime.retrieval.models import Document
from agent_runtime.retrieval.chunker import SimpleChunker

def test_chunker_overlap_and_boundary():
    doc = Document(document_id="d1", title="Test", source_path="t.md", content="abcdefghijklmnopqrstuvwxyz")
    # 设置极短块大小强行切分
    chunker = SimpleChunker(chunk_size=10, overlap=3)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) > 1
    # 检查第一个块和第二个块是否由于 overlap 产生部分内容交织
    assert chunks[0].text[7:10] == chunks[1].text[0:3]