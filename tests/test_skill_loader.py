# tests/test_skill_loader.py
import pytest
from agent_runtime.skills.loader import SkillLoader
from agent_runtime.skills.selector import SemanticSkillSelector, SkillSelector

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
    
def test_semantic_skill_fallback_and_activation():
    """
    测试模糊语义泛化激活：
    Query 中不包含任何 'rolling-backtest' 关键字，
    但包含高度相关的语义特征（chronological simulation, future leakage）。
    验证语义选择器能否通过双路召回成功将其激活并解冻。
    """
    loader = SkillLoader("skills")
    selector = SemanticSkillSelector(loader)
    
    # 故意避开技能名称，使用近义词和描述中的深层概念
    fuzzy_query = "Can you check my chronological simulation rows to ensure there is no future information leakage?"
    
    activated = selector.select_and_activate(fuzzy_query)
    
    assert activated is not None
    assert activated.name == "rolling-backtest"
    # 确保完整正文是在这里被解冻的
    assert "Core Execution Steps" in activated.full_instructions
    assert "Validate that chronological row splitting" in activated.full_instructions

def test_semantic_skill_seasonal_diagnostics():
    """验证另一个技能卡的语义泛化"""
    loader = SkillLoader("skills")
    selector = SemanticSkillSelector(loader)
    
    fuzzy_query = "Please analyze the repetitive variance and components in our high-frequency series."
    
    activated = selector.select_and_activate(fuzzy_query)
    
    assert activated is not None
    assert activated.name == "seasonal-diagnostics"
