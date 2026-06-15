# src/agent_runtime/memory/sqlite_store.py
import sqlite3
import json
from typing import List, Optional
from agent_runtime.memory.models import MemoryRecord

class SQLiteMemoryStore:
    def __init__(self, db_path: str = ":memory:"): # 单元测试默认用内存模式，生产可传本地文件路径
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(namespace, key) -- 强制锁死：同一个域下的同一个Key只能有一条记忆，防无限膨胀
                )
            """)
        self.conn.commit()

    def put_memory(self, record: MemoryRecord):
        """写入或更新记忆 (金融级强确定性 Upsert 语法)"""
        self.conn.execute("""
                INSERT INTO memories (memory_id, namespace, key, memory_type, content, importance, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at,
                    importance = excluded.importance,
                    metadata = excluded.metadata
            """, (
                record.memory_id, record.namespace, record.key, record.memory_type,
                record.content, record.importance, record.updated_at, json.dumps(record.metadata)
            ))
        self.conn.commit()

    def get_memory_by_key(self, namespace: str, key: str) -> Optional[MemoryRecord]:
        cursor = self.conn.cursor()
        cursor.execute(
                "SELECT memory_id, namespace, key, memory_type, content, importance, updated_at, metadata FROM memories WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
        row = cursor.fetchone()
        if not row:
            return None
        return MemoryRecord(
            memory_id=row[0], namespace=row[1], key=row[2], memory_type=row[3],
            content=row[4], importance=row[5], updated_at=row[6], metadata=json.loads(row[7])
        )

    def list_namespace_memories(self, namespace: str) -> List[MemoryRecord]:
        """安全边界：只捞取属于当前指定 Namespace 权限内的记忆"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT memory_id, namespace, key, memory_type, content, importance, updated_at, metadata FROM memories WHERE namespace = ?", (namespace,))
        rows = cursor.fetchall()
        return [
            MemoryRecord(
                memory_id=r[0], namespace=r[1], key=r[2], memory_type=r[3],
                content=r[4], importance=r[5], updated_at=r[6], metadata=json.loads(r[7])
            ) for r in rows
        ]

    def close(self):
        self.conn.close()
