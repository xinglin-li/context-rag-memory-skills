# src/agent_runtime/context/assembler.py
from typing import List
from agent_runtime.context.models import ContextItem, ContextBundle
from agent_runtime.context.budget import TokenBudgetTracker
from agent_runtime.context.dedup import ContextDeduplicator

class ContextAssembler:
    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens
    
    def assemble(self, raw_items: List[ContextItem]) -> ContextBundle:
        # 1. 基础去重
        deduped_items = ContextDeduplicator.deduplicate(raw_items)
        
        # 2. 严格按照优先级从大到小排序 (Priority Queue 心智)
        sorted_items = sorted(deduped_items, key=lambda x: x.priority, reverse=True)
        
        retained_items: List[ContextItem] = []
        dropped_items: List[ContextItem] = []
        current_tokens = 0
        
        for item in sorted_items:
            item_tokens = TokenBudgetTracker.estimate_tokens(item.content)
            
            # 防护底线：如果保留这个 item 会超出 max_tokens 预算
            if current_tokens + item_tokens > self.max_tokens:
                # 核心防线 1：系统最高指令（System Instructions，优先级设定为100）在任何情况下绝对不允许被丢弃！
                if item.priority >= 100:
                    # 强行塞入，即使超出预算也必须保护核心系统控制权
                    retained_items.append(item)
                    current_tokens += item_tokens
                else:
                    # 核心防线 2：其他非核心内容则直接丢弃，保护预算不被过度消耗
                    dropped_items.append(item)
            else:
                # 正常保留
                retained_items.append(item)
                current_tokens += item_tokens
        
        return ContextBundle(
            items=retained_items,
            total_estimated_tokens=current_tokens,
            max_tokens=self.max_tokens,
            dropped_items=dropped_items
        )