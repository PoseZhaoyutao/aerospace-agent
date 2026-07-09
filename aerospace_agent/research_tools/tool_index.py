"""ToolVectorIndex — 工具语义向量索引。

将 105 个工具的 name + description + category + params 向量化存储，
用户任务描述 → 语义检索 → 只注入 top-K 最相关工具的 schema。

复用项目已有的 SimpleEmbedder + VectorStore（纯 numpy，无外部依赖）。

流程：
  1. 初始化时把所有工具注册到向量库
  2. 用户任务 → embed → 余弦检索 top-K
  3. 只把 top-K 工具的 schema 注入 system prompt
  4. 动态创建的新工具自动追加到向量库

性能：
  - 索引构建: ~0.3s（105 个工具）
  - 单次检索: <1ms（numpy 点积）
  - Token 节省: 2015 → ~200（top-5 只注入 5 个工具 schema）
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ..rag.vector_store import SimpleEmbedder, VectorStore
from ..core.bilingual_mapping import hybrid_search, keyword_match_score, translate_query

_logger = logging.getLogger(__name__)

__all__ = ["ToolVectorIndex"]


class ToolVectorIndex:
    """工具语义向量索引——按任务检索最相关工具。

    用法：
        idx = ToolVectorIndex()
        idx.build_from_registry(registry)       # 构建索引
        tools = idx.search("计算轨道速度", k=5)  # 检索 top-5
        idx.add_tool("new_tool", "描述", "cat")  # 追加新工具
    """

    def __init__(self, persist_path: Optional[str] = None):
        """初始化工具向量索引。

        Args:
            persist_path: 持久化路径。如果指定，build 后自动保存，下次 init 自动加载。
        """
        self._embedder = SimpleEmbedder(dim=256, ngram=3, seed=42)
        self._store = VectorStore(self._embedder)
        self._persist_path = persist_path
        self._built = False
        # 缓存所有工具的 (name, description) 用于关键词匹配
        self._all_tools: List[Tuple[str, str]] = []

        # 尝试加载已有索引
        if persist_path and os.path.exists(persist_path):
            try:
                self._store.load(persist_path)
                self._built = True
                # 从 metadatas 恢复 _all_tools
                self._all_tools = [
                    (m.get("name", ""), m.get("description", ""))
                    for m in self._store.metadatas
                ]
                _logger.info("工具向量索引已从 %s 加载 (%d 个工具)",
                             persist_path, len(self._store))
            except Exception as e:
                _logger.warning("加载工具向量索引失败: %s", e)

    @property
    def is_built(self) -> bool:
        return self._built

    def _tool_to_text(self, name: str, description: str,
                      category: str, params: List[Dict]) -> str:
        """把工具元信息转为可嵌入的文本。

        把 name / description / category / param names 拼接成一段
        富信息文本，让语义检索能匹配到参数级别。
        """
        param_names = [p.get("name", "") for p in params] if params else []
        param_str = " ".join(param_names)
        return f"[{category}] {name}: {description} 参数: {param_str}"

    def build_from_registry(self, registry) -> None:
        """从 ResearchToolRegistry 构建完整索引。

        Args:
            registry: ResearchToolRegistry 实例
        """
        self._store.clear()
        self._all_tools = []
        for name in registry.list_all():
            tool = registry.get(name)
            if tool is None:
                continue
            # 构建可嵌入文本
            text = self._tool_to_text(
                name=tool.name,
                description=tool.description,
                category=tool.category,
                params=[{"name": p.name, "type": p.type}
                        for p in tool.params],
            )
            self._store.add(text, {
                "name": name,
                "category": tool.category,
                "description": tool.description,
            })
            # 缓存用于关键词匹配
            self._all_tools.append((name, tool.description))
        self._store.reindex()
        self._built = True

        # 持久化
        if self._persist_path:
            try:
                self._store.save(self._persist_path)
                _logger.info("工具向量索引已保存到 %s", self._persist_path)
            except Exception as e:
                _logger.warning("保存工具向量索引失败: %s", e)

        _logger.info("工具向量索引构建完成: %d 个工具", len(self._store))

    def add_tool(self, name: str, description: str,
                 category: str, params: Optional[List[Dict]] = None) -> None:
        """追加单个工具到索引（动态创建的新工具用）。"""
        text = self._tool_to_text(name, description, category, params or [])
        self._store.add(text, {
            "name": name,
            "category": category,
            "description": description,
        })
        # 缓存用于关键词匹配
        self._all_tools.append((name, description))
        # 增量添加后重新索引保证 IDF 一致
        self._store.reindex()

        if self._persist_path:
            try:
                self._store.save(self._persist_path)
            except Exception:
                pass

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索最相关的 top-K 工具（混合检索：向量 + 关键词）。

        策略：
        1. 先做向量检索，取 top-k*3 候选
        2. 同时对全部工具做中英双语关键词匹配
        3. 用 hybrid_search 合并两路分数（关键词权重 0.6，向量权重 0.4）
        4. 如果向量 top-1 分数 < 0.15（跨语言失败信号），
           纯关键词匹配结果优先

        Args:
            query: 用户任务描述
            k: 返回工具数（默认 5）

        Returns:
            [{"name":..., "category":..., "description":..., "score":...}, ...]
            按相关度降序
        """
        if not self._built or len(self._store) == 0:
            return []

        # 1. 向量检索 — 取更多候选用于合并
        vector_k = min(k * 3, len(self._store))
        raw_results = self._store.search(query, top_k=vector_k)
        vector_results = [(meta.get("name", ""), score) for score, _text, meta in raw_results]

        # 向量 top-1 分数
        vec_top_score = vector_results[0][1] if vector_results else 0.0

        # 2. 混合检索
        combined = hybrid_search(
            query=query,
            tools=self._all_tools,
            vector_results=vector_results,
            k=k * 2,  # 取更多再裁剪
            keyword_weight=0.8,
            vector_weight=0.2,
        )

        # 3. 如果向量分数很低（跨语言失败），用纯关键词匹配补充
        if vec_top_score < 0.15 and self._all_tools:
            kw_results = []
            for name, desc in self._all_tools:
                ks = keyword_match_score(query, name, desc)
                if ks > 0:
                    kw_results.append({"name": name, "score": ks})
            kw_results.sort(key=lambda x: x["score"], reverse=True)
            # 合并去重
            existing_names = {r["name"] for r in combined}
            for kr in kw_results[:k]:
                if kr["name"] not in existing_names:
                    combined.append({
                        "name": kr["name"],
                        "score": kr["score"],
                        "source": "keyword_fallback",
                    })

        # 4. 构建 metadata 查找表
        meta_lookup: Dict[str, Dict] = {}
        for m in self._store.metadatas:
            name = m.get("name", "")
            if name:
                meta_lookup[name] = m

        # 5. 最终排序取 top-k
        combined.sort(key=lambda x: x["score"], reverse=True)

        result = []
        seen = set()
        for item in combined[:k]:
            name = item["name"]
            if name in seen:
                continue
            seen.add(name)
            meta = meta_lookup.get(name, {})
            result.append({
                "name": name,
                "category": meta.get("category", ""),
                "description": meta.get("description", ""),
                "score": item["score"],
            })

        return result

    def search_with_schemas(self, query: str, k: int = 5,
                            registry=None) -> List[str]:
        """检索并返回 top-K 工具的完整 schema（供 system prompt 注入）。

        Args:
            query: 用户任务描述
            k: 返回工具数
            registry: ResearchToolRegistry（用于获取工具 schema）

        Returns:
            schema 字符串列表，如 ["- save_file(path:str, content:str): 保存文件到文件", ...]
        """
        hits = self.search(query, k=k)
        if not hits:
            return []

        if registry is None:
            # 返回简短格式
            return [f"- {h['name']}: {h['description']}" for h in hits]

        schemas = []
        for h in hits:
            tool = registry.get(h["name"])
            if tool:
                schemas.append(tool.to_schema())
            else:
                schemas.append(f"- {h['name']}: {h['description']}")
        return schemas

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计。"""
        return {
            "total_tools": len(self._store),
            "is_built": self._built,
            "embed_dim": self._embedder.dim,
            "persisted": self._persist_path is not None,
        }
