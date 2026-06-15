# tests/test_skill_loader.py
import pytest
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.selector import SkillSelector

def test_discovery_stage_only_metadata():
    """测试情况 1: 扫描发现阶段验证。全库扫描只能读取轻量简介，内存不能包含大段SOP正文"""
    loader = SkillLoader("skills")
    catalog = loader.discover_catalog()
    
    assert len(catalog) >= 2
    names = [m.name for m in catalog]
    assert "rolling-backtest" in names
    assert "seasonal-diagnostics" in names
    
    # 验证确定性防御边界：扫描出来的 Metadata 实体里绝不包含长篇累牍的说明正文字段
    assert not hasattr(catalog[0], "full_instructions")

def test_progressive_disclosure_activation():
    """测试情况 2: 渐进式按需解冻测试。只有触发匹配后，说明书正文才会被物理加载"""
    loader = SkillLoader("skills")
    selector = SkillSelector(loader)
    
    # 用户提问提到了回测相关逻辑
    query = "Please help me run a rolling-backtest for ARIMA"
    activated = selector.select_and_activate(query)
    
    assert activated is not None
    assert activated.name == "rolling-backtest"
    # SOP 说明书正文成功解冻加载入仓
    assert "Core Execution Steps" in activated.full_instructions
    assert "Define the clear forecasting horizon" in activated.full_instructions