# src/agent_runtime/memory/models.py
from datetime import UTC, datetime
from typing import Dict

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    memory_id: str
    namespace: str
    key: str
    memory_type: str
    content: str
    importance: float = 1.0
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: Dict[str, str] = Field(default_factory=dict)
