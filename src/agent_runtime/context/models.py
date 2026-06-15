# src/agent_runtime/context/models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ContextItem(BaseModel):
    item_id: str = Field(..., description="Unique ID for this context slice")
    kind: str = Field(..., description="system_instruction, user_message, tool_result, retrieved_evidence, etc.")
    content: str = Field(..., description="The literal text payload.")
    priority: int = Field(..., description="Higher numbers mean higher retention priority (0-100).")
    provenance: Dict[str, str] = Field(default_factory=dict, description="Source trace, e.g., {'file': 'backtest.md', 'line': '12'}")
    trust_level: str = Field(..., description="system_trusted, user_supplied, retrieved_untrusted")

class ContextBundle(BaseModel):
    items: List[ContextItem]
    total_estimated_tokens: int
    max_tokens: int
    dropped_items: List[ContextItem] = Field(default_factory=list)