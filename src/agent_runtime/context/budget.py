# src/agent_runtime/context/budget.py
from agent_runtime.context.models import ContextItem

class TokenBudgetTracker:
    @staticmethod
    def estimate_tokens(content: str) -> int:
        """纯确定性的 Token 估算 baseline：4个字符约等于1个Token，空字符至少算1个"""
        if not content:
            return 1
        return max(1, len(content) // 4)