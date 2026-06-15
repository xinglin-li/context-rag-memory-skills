# tests/test_memory_store.py
import pytest
import uuid
from agent_runtime.memory.models import MemoryRecord
from agent_runtime.memory.sqlite_store import SQLiteMemoryStore

def test_namespace_strict_isolation():
    """测试情况 1：严格的 Namespace 跨租户/跨域隔离。 Xinglin 的偏好绝不能被恶意泄露给别人"""
    store = SQLiteMemoryStore(":memory:")
    
    rec_xinglin = MemoryRecord(
        memory_id="m1", namespace="users/xinglin/pref", key="baseline", memory_type="semantic", content="Use seasonal-naive"
    )
    rec_hacker = MemoryRecord(
        memory_id="m2", namespace="users/hacker/pref", key="baseline", memory_type="semantic", content="Use linear-trend"
    )
    
    store.put_memory(rec_xinglin)
    store.put_memory(rec_hacker)
    
    # 1. 以 Xinglin 的凭证去捞取
    xinglin_list = store.list_namespace_memories("users/xinglin/pref")
    assert len(xinglin_list) == 1
    assert xinglin_list[0].content == "Use seasonal-naive" # 必须只能看到自己的
    
    # 2. 确保在 Xinglin 的命名空间里绝对搜不到 hacker 的任何蛛丝马迹
    assert "Use linear-trend" not in [m.content for m in xinglin_list]

def test_upsert_on_conflict():
    """测试情况 2：同域同Key覆盖更新测试。旧记忆必须被新事实无情覆盖，防止产生多条冲突数据"""
    store = SQLiteMemoryStore(":memory:")
    r1 = MemoryRecord(memory_id="id1", namespace="proj_1", key="risk_limit", memory_type="semantic", content="Max limit 5%")
    r2 = MemoryRecord(memory_id="id2", namespace="proj_1", key="risk_limit", memory_type="semantic", content="Max limit 2%") # 调整风控线
    
    store.put_memory(r1)
    store.put_memory(r2) # 发生唯一键冲突，自动执行DO UPDATE
    
    res = store.get_memory_by_key("proj_1", "risk_limit")
    assert res.content == "Max limit 2%"  # 完美更新为最新事实