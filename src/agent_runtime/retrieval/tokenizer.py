# src/agent_runtime/retrieval/tokenizer.py
import re
from typing import List

class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        # 1. 统一转为小写
        lowered = text.lower()
        # 2. 正则剥离非字母数字的噪音标点，保留基本字符
        cleaned = re.sub(r'[^\w\s\(\),-]', '', lowered)
        # 3. 按空格切分，过滤空字符
        tokens = []
        for token in cleaned.split():
            if not token:
                continue

            tokens.append(token)

            # 对连字符复合词同时保留整体词和子词，提升专有名词召回与权重区分。
            if '-' in token:
                parts = [part for part in token.split('-') if part]
                tokens.extend(parts)

        return tokens