# tests/test_context_budget.py
import pytest
from agent_runtime.context.models import ContextItem
from agent_runtime.context.assembler import ContextAssembler

def test_context_within_budget():
    """测试情况 1：内容总量远低于预算时，必须全部保留，不得漏掉任何信息"""
    items = [
        ContextItem(item_id="1", kind="system_instruction", content="You are a system.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="2", kind="user_message", content="Help me buy a stock.", priority=95, trust_level="user_supplied")
    ]
    assembler = ContextAssembler(max_tokens=1000)
    bundle = assembler.assemble(items)
    
    assert len(bundle.items) == 2
    assert len(bundle.dropped_items) == 0

def test_context_over_budget_truncation():
    """测试情况 2：内容爆表。低优先级（RAG检索的未置信文本）必须优先被砍掉，且必须在 dropped_items 留下尸体痕迹"""
    items = [
        ContextItem(item_id="sys", kind="system_instruction", content="Keep safe.", priority=100, trust_level="system_trusted"),
        ContextItem(item_id="user", kind="user_message", content="Do task.", priority=95, trust_level="user_supplied"),
        # 这个低优异检索内容非常长，估算会耗费大量 token
        ContextItem(item_id="rag_junk", kind="retrieved_evidence", content="Junk text " * 100, priority=20, trust_level="retrieved_untrusted")
    ]
    
    # 设置一个极其严苛的 max_tokens 预算，逼迫其斩断垃圾检索结果
    assembler = ContextAssembler(max_tokens=30)
    bundle = assembler.assemble(items)
    
    # sys 和 user 必须活着
    retained_ids = [item.item_id for item in bundle.items]
    assert "sys" in retained_ids
    assert "user" in retained_ids
    
    # 垃圾 RAG 证据必须进入掉落列表
    dropped_ids = [item.item_id for item in bundle.dropped_items]
    assert "rag_junk" in dropped_ids

def test_system_instruction_is_undroppable():
    """测试情况 3：绝对底线测试。哪怕预算严重超支，系统最高指令（Priority=100）也绝对不能被丢弃"""
    items = [
        ContextItem(item_id="sys_heavy", kind="system_instruction", content="System Rule " * 50, priority=100, trust_level="system_trusted"),
        ContextItem(item_id="low_rag", kind="retrieved_evidence", content="Some text", priority=10, trust_level="retrieved_untrusted")
    ]
    
    assembler = ContextAssembler(max_tokens=10) # 极其狭窄的窗口
    bundle = assembler.assemble(items)
    
    # sys_heavy 必须强行保留，宁可突破预算也不能丢掉系统大脑的控制权
    assert "sys_heavy" in [item.item_id for item in bundle.items]
    assert "low_rag" in [item.item_id for item in bundle.dropped_items]
    