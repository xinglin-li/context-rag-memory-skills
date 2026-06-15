# src/agent_runtime/memory/policy.py
import uuid
from agent_runtime.memory.models import MemoryRecord
from agent_runtime.memory.sqlite_store import SQLiteMemoryStore

class MemoryWritePolicy:
    # 敏感的系统劫持黑特征词
    FORBIDDEN_KEYWORDS = {"ignore previous", "delete all", "system instruction", "override runtime"}

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store

    def inspect_and_commit(self, namespace: str, key: str, content: str, memory_type: str = "semantic") -> str:
        """
        上层应用调用的审查写入安全网
        返回状态: "committed", "rejected_sensitive", "skipped_noise"
        """
        lowered_content = content.lower()
        
        # 核心安全防线 1：持久化注入拦截（Persistent Injection Block）
        # 检查候选记忆是否试图冒充系统最高控制命令
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw in lowered_content:
                return "rejected_sensitive"
                
        # 核心防线 2：短期无意义噪音过滤
        if len(content.strip()) < 5:
            return "skipped_noise"
            
        # 3. 校验通过，允许安全落仓
        record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            namespace=namespace,
            key=key,
            memory_type=memory_type,
            content=content
        )
        self.store.put_memory(record)
        return "committed"