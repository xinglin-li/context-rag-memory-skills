# src/agent_runtime/retrieval/embeddings.py
from abc import ABC, abstractmethod
from typing import List
import hashlib

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """将单条文本转换为稠密特征向量"""
        pass

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    def get_embedding(self, text: str) -> List[float]:
        """
        确定性伪 Embedding 生成算法：
        利用 MD5 哈希将输入文本转化为绝对可重复、可预测的伪固定维数向量，彻底摆脱网络和概率干扰。
        """
        if not text:
            return [0.0] * self.dimensions
            
        # 针对特定量化测试词汇进行语义“硬编码”，以便在测试中完美验证几何距离
        normalized = text.lower()
        if "arima" in normalized or "time series" in normalized:
            return [0.9, 0.1, 0.0, 0.0]  # 假设前两个维度代表时序特征
        if "backtest" in normalized or "trading" in normalized:
            return [0.0, 0.0, 0.8, 0.2]  # 假设后两个维度代表回测特征
            
class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    def get_embedding(self, text: str) -> List[float]:
        """
        确定性伪 Embedding 生成算法：
        利用 MD5 哈希将输入文本转化为绝对可重复、可预测的伪固定维数向量，彻底摆脱网络和概率干扰。
        """
        if not text:
            return [0.0] * self.dimensions
            
        # 针对特定量化测试词汇进行语义“硬编码”，以便在测试中完美验证几何距离
        normalized = text.lower()
        if "arima" in normalized or "time series" in normalized:
            return [0.9, 0.1, 0.0, 0.0]  # 假设前两个维度代表时序特征
        if "backtest" in normalized or "trading" in normalized:
            return [0.0, 0.0, 0.8, 0.2]  # 假设后两个维度代表回测特征
            
        # 兜底：利用字符哈希生成稳定的伪随机向量
        hasher = hashlib.md5(text.encode('utf-8'))
        digest = hasher.digest()
        
        vector = []
        for i in range(self.dimensions):
            # 取出不同位置的字节转化为 0-1 之间的浮点数
            val = digest[i % len(digest)] / 255.0
            vector.append(round(val, 4))
        return vector