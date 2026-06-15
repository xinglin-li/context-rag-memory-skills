# src/agent_runtime/skills/selector.py
from typing import List, Optional
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.models import SkillMetadata, ActivatedSkill

class SkillSelector:
    def __init__(self, loader: SkillLoader):
        self.loader = loader
        # 内存常驻缓存：只保留极轻量级的元数据卡座
        self.catalog: List[SkillMetadata] = self.loader.discover_catalog()
    
    def select_and_activate(self, query: str) -> Optional[ActivatedSkill]:
        """
        基于确定性关键字匹配的 Selector Baseline。
        大模型可以通过 Catalog 简介匹配后，由 Runtime 物理拉起对应解冻函数。
        """
        lowered_query = query.lower()
        
        # 遍历常驻的极简 Catalog 简介
        for meta in self.catalog:
            # 核心防线：通过关键字触发精确的 Progressive Disclosure 激活
            if meta.name in lowered_query or any(word in lowered_query for word in meta.description.lower().split()[:5]):
                # 命中！开始第二层按需解冻加载
                return self.loader.load_full_skill(meta)
        return None