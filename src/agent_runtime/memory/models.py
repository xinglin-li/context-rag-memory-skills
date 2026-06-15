# src/agent_runtime/memory/models.py
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import UTC, datetime

class MemoryRecord(BaseModel):
    memory_id: str
    namespace: str         # 核心隔离域，例如 "users/xinglin_li/preferences"
    key: str               # 记忆检索键，例如 "baseline_rule"
    memory_type: str       # semantic, episodic
    content: str           # 记忆的literal文本
    importance: float = 1.0
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: Dict[str, str] = Field(default_factory=dict)