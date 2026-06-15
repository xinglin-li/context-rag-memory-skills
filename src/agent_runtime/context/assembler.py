# src/agent_runtime/context/assembler.py
from typing import List

from agent_runtime.context.budget import TokenBudgetTracker
from agent_runtime.context.dedup import ContextDeduplicator
from agent_runtime.context.models import ContextBundle, ContextItem


class ContextAssembler:
    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens

    def assemble(self, raw_items: List[ContextItem]) -> ContextBundle:
        deduped_items = ContextDeduplicator.deduplicate(raw_items)
        sorted_items = sorted(deduped_items, key=lambda x: x.priority, reverse=True)

        retained_items: List[ContextItem] = []
        dropped_items: List[ContextItem] = []
        current_tokens = 0

        for item in sorted_items:
            item_tokens = TokenBudgetTracker.estimate_tokens(item.content)

            if current_tokens + item_tokens > self.max_tokens:
                if item.priority >= 100:
                    # System instructions are retained even if they exceed the nominal budget.
                    retained_items.append(item)
                    current_tokens += item_tokens
                else:
                    dropped_items.append(item)
            else:
                retained_items.append(item)
                current_tokens += item_tokens

        return ContextBundle(
            items=retained_items,
            total_estimated_tokens=current_tokens,
            max_tokens=self.max_tokens,
            dropped_items=dropped_items,
        )
