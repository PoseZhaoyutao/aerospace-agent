"""航天知识库管理 (Aerospace Knowledge Base)。

封装整个 RAG: 统一管理 :class:`VectorStore` + :class:`KeywordIndex` +
:class:`KnowledgeGraph`, 提供批量/单条索引入口、内置预填充航天知识,
以及统一的 ``query`` 检索入口。

内置的 ``index_default_knowledge()`` 提供 21 条核心航天知识, 物理正确且与
``aerospace_agent.physics`` 子包一致, 并与知识图谱节点对齐 (metadata 带
``node_id`` / ``type``), 使重排器能识别公式类节点。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .keyword_index import KeywordIndex
from .knowledge_graph import KnowledgeGraph
from .reranker import Reranker
from .retriever import HybridRetriever, RetrievalResult
from .vector_store import VectorStore

__all__ = ["AerospaceKnowledgeBase"]

DEFAULT_DATA_DIR = "/workspace/data"


class AerospaceKnowledgeBase:
    """航天知识库: 向量 + 关键词 + 图谱 三路混合 RAG 的统一管理器。"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        autoload: bool = True,
        auto_default_knowledge: bool = True,
    ):
        self.data_dir = data_dir
        self.vector_store = VectorStore()
        self.keyword_index = KeywordIndex()
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.prepopulate()  # 图谱默认预填充

        # 文档注册表: doc_id -> (text, metadata), 与向量库/倒排索引对齐
        self.documents: List[Tuple[str, dict]] = []
        self.retriever = HybridRetriever(
            self.vector_store, self.keyword_index, self.knowledge_graph
        )

        loaded = False
        if autoload:
            loaded = self.load()
        # 若全新 (无持久化), 自动索引内置核心知识, 保证开箱即用
        if not loaded and auto_default_knowledge and len(self.documents) == 0:
            self.index_default_knowledge()

    # ------------------------------------------------------------------ 索引
    def index_text(self, text: str, source: str = "manual",
                   metadata: Optional[dict] = None) -> int:
        """索引单段文本, 返回 doc_id。"""
        meta = {"source": source}
        if metadata:
            meta.update(metadata)
        doc_id = len(self.documents)
        self.vector_store.add(text, meta)
        self.keyword_index.add(doc_id, text)
        self.documents.append((text, meta))
        return doc_id

    def index_directory(
        self, dir_path: str, extensions: Optional[List[str]] = None
    ) -> int:
        """批量索引目录下文档, 按段落切块。返回索引文档数。"""
        if extensions is None:
            extensions = [".md", ".txt", ".py"]
        ext_set = {e.lower() for e in extensions}
        count = 0
        for root, _dirs, files in os.walk(dir_path):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in ext_set:
                    continue
                fpath = os.path.join(root, fn)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                except OSError:
                    continue
                for chunk in self._chunk(raw):
                    self.index_text(chunk, source=fpath)
                    count += 1
        # 批量索引后用最终 IDF 重嵌, 保证一致性
        self.vector_store.reindex()
        return count

    @staticmethod
    def _chunk(text: str, min_len: int = 40, max_len: int = 1200) -> List[str]:
        """按空行切段; 过短合并; 过长截断。"""
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        # 也兼容单换行分段
        if len(parts) <= 1:
            parts = [p.strip() for p in text.splitlines() if p.strip()]
        chunks: List[str] = []
        buf = ""
        for p in parts:
            if len(buf) + len(p) < max_len:
                buf = (buf + "\n" + p).strip() if buf else p
            else:
                if buf:
                    chunks.append(buf)
                buf = p if len(p) <= max_len else p[:max_len]
            if len(buf) >= max_len:
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        return [c for c in chunks if len(c) >= min_len] or chunks

    # ------------------------------------------------- 内置预填充航天知识
    def index_default_knowledge(self) -> int:
        """内置预填充航天核心知识 (21 条), 与知识图谱节点对齐。

        覆盖: 开普勒方程、vis-viva、Hohmann、Lambert、拼凑圆锥、SOI、C3、
        TLI、LOI、发射窗口、相位角、地月转移流程、二体问题、轨道根数、
        能量/角动量守恒, 以及 orekit/gmat/spiceypy/basilisk/stk 工具说明。
        """
        entries: List[Tuple[str, str, str, str]] = [
            # (node_id, type, content, source)
            ("vis_viva", "formula",
             "vis-viva (活力) 方程: v² = μ(2/r − 1/a)。任意圆锥轨道任意点半径处的速度, "
             "由能量守恒 ε=v²/2−μ/r=−μ/(2a) 导出。是 Hohmann 转移、TLI、LOI 等 Δv 计算的核心公式。",
             "default"),
            ("kepler_equation", "formula",
             "开普勒方程 M = E − e·sinE。给定平近点角 M (与时间线性) 迭代求偏近点角 E, "
             "再由 tan(ν/2)=√((1+e)/(1−e))·tan(E/2) 得真近点角 ν。用于二体位置/时间转换, 依赖二体问题。",
             "default"),
            ("hohmann", "concept",
             "Hohmann 转移: 共面圆轨道间最省能量的双脉冲椭圆转移。转移椭圆半长轴 a_t=(r1+r2)/2, "
             "Δv1=√(μ(2/r1−1/a_t))−√(μ/r1), Δv2=√(μ/r2)−√(μ(2/r2−1/a_t)), "
             "飞行时间 t=π√(a_t³/μ)。由 vis-viva 推导。",
             "default"),
            ("lambert", "concept",
             "Lambert 问题: 已知两端点位置向量 r1, r2 与飞行时间 Δt, 求连接轨道 (速度 v1, v2)。"
             "是 Hohmann 的通用化 (支持非共面/任意时间), 依赖二体问题; 常用普适变量法求解。",
             "default"),
            ("patched_conic", "concept",
             "拼凑圆锥近似 (patched conic): 把多体问题拆成若干二体段, 在引力作用球 (SOI) 边界"
             "做速度匹配。地月转移分地心段 (双曲逃逸) 与月心段 (双曲接近+近月制动), "
             "是地月转移的核心简化方法。",
             "default"),
            ("soi", "concept",
             "引力作用球 SOI: r_SOI = a·(μ_moon/μ_earth)^(2/5) (Laplace)。月球 SOI ≈ 66183 km。"
             "SOI 是拼凑圆锥的分段边界, 约束该方法适用范围。",
             "default"),
            ("c3", "concept",
             "特征能量 C3 = v_∞², 即逃逸双曲剩余速度的平方。由 vis-viva 双曲情形 v_∞²=−μ/a (a<0) 给出。"
             "C3 衡量发射能量需求, 约束 TLI 所需 Δv; C3 越小越省运载能力。",
             "default"),
            ("tli", "mission",
             "地月注入 TLI (Trans-Lunar Injection): 在地球停泊轨道近地点施加脉冲, "
             "把航天器送入地月转移轨道 (双曲逃逸地心段)。Δv 由 vis-viva 估算, 需满足 C3 要求; "
             "时机由发射窗口/相位角决定。",
             "default"),
            ("loi", "mission",
             "环月注入 LOI (Lunar Orbit Insertion): 航天器经月心双曲接近段抵达近月点时反向制动, "
             "把双曲轨道捕获为环月轨道。捕获 Δv = v_p − v_circ, v_p²=v_∞²+2μ_moon/r_p (vis-viva)。",
             "default"),
            ("launch_window", "concept",
             "发射窗口: 满足相位角/光照/测控等约束的可发射时段。地月转移要求发射时月球处于使航天器"
             "到达 SOI 时与月球相遇的相位, 由相位角与轨道根数决定; 约束整个地月转移任务。",
             "default"),
            ("phase_angle", "concept",
             "相位角: 发射时刻目标天体相对出发点的角位置。Hohmann 转移相位角由两轨道半径比决定; "
             "决定转移时机, 是发射窗口的核心约束, 依赖轨道根数。",
             "default"),
            ("earth_moon_transfer", "mission",
             "地月转移: 从地球到月球的轨道设计。核心方法为拼凑圆锥近似 + Hohmann (或 Lambert 精确解); "
             "流程: LEO 停泊 → TLI 注入 → 地心双曲段 → SOI 边界 → 月心双曲段 → LOI 捕获。"
             "需发射窗口(相位角)匹配, Δv 预算由 vis-viva 估算。",
             "default"),
            ("two_body", "concept",
             "二体问题: 两质点在万有引力下的运动, 解析解为圆锥曲线 (圆/椭圆/抛物/双曲)。"
             "基于角动量守恒 (h=r×v) 与能量守恒 (ε=v²/2−μ/r), 是开普勒方程、vis-viva、"
             "Hohmann、Lambert、拼凑圆锥的共同基础。",
             "default"),
            ("orbital_elements", "concept",
             "经典轨道根数: 半长轴 a、偏心率 e、倾角 i、升交点赤经 Ω、近地点幅角 ω、真近点角 ν, "
             "共六要素完整描述二体轨道。相位角与发射窗口计算均依赖轨道根数。",
             "default"),
            ("energy_conservation", "concept",
             "能量守恒 (二体): 比机械能 ε = v²/2 − μ/r = −μ/(2a) 守恒。"
             "由此直接导出 vis-viva 方程, 是所有 Δv 估算的物理根基。",
             "default"),
            ("angular_momentum", "concept",
             "角动量守恒 (中心力场): 比角动量 h = r × v 守恒, 故轨道位于固定平面内, "
             "且面积速度守恒 (开普勒第二定律)。是二体问题平面性与时变规律的根基。",
             "default"),
            ("delta_v", "concept",
             "速度增量 Δv 预算: 任务总冲量需求。地月转移 Δv ≈ TLI Δv + LOI Δv, "
             "各段由 vis-viva 估算; Δv 是地月转移任务的核心约束之一。",
             "default"),
            ("orekit", "tool",
             "Orekit: CNES 开源的航天动力学基础库 (Java 核心, Python 封装)。"
             "支持二体/数值传播、历书、坐标变换; 本项目 orekit_tool/astropy_tool 封装之, "
             "适合工程级轨道计算。",
             "default"),
            ("gmat", "tool",
             "GMAT (General Mission Analysis Tool): NASA Goddard 开源任务分析工具。"
             "支持拼凑圆锥、Lambert 优化、数值积分、脚本; gmat_tool 封装之, 适合复杂任务设计。",
             "default"),
            ("spiceypy", "tool",
             "SPICE (NAIF): NASA 行星历书/姿态/事件工具包, spiceypy 为 Python 封装。"
             "提供月球/行星精确星历 (星历内核), 用于发射窗口与相位角的精确计算; "
             "spiceypy_tool 封装之。",
             "default"),
            ("basilisk", "tool",
             "Basilisk: University of Colorado 开发的多体/航天器 GNC 仿真框架。"
             "支持六自由度动力学、多体引力、传感器/执行器闭环仿真; basilisk_tool 封装之, "
             "适合 GNC 验证。",
             "default"),
            ("stk", "tool",
             "STK (Systems Tool Kit): AGI 商业航天任务分析软件, 行业标准。"
             "支持全任务链分析、覆盖、链路、可视化; stk_tool 封装之, 适合工程总体设计与汇报。",
             "default"),
        ]
        # 去重: 避免重复 index_default_knowledge 时重复入栈
        existing = {m.get("node_id") for _, m in self.documents}
        added = 0
        for node_id, ntype, content, source in entries:
            if node_id in existing:
                continue
            self.index_text(
                content, source=source,
                metadata={"node_id": node_id, "type": ntype},
            )
            added += 1
        # 用最终 IDF 重嵌, 保证一致性 + 可复现
        self.vector_store.reindex()
        return added

    # ------------------------------------------------------------------ 检索
    def query(
        self,
        query: str,
        top_k: int = 5,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        use_reranker: bool = True,
    ) -> List[RetrievalResult]:
        """统一检索入口: 三路混合 + (可选) 规则重排。"""
        return self.retriever.retrieve(
            query, top_k=top_k, weights=weights, use_reranker=use_reranker
        )

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict:
        return {
            "num_documents": len(self.documents),
            "num_graph_nodes": self.knowledge_graph.num_nodes,
            "num_graph_edges": self.knowledge_graph.num_edges,
            "embed_dim": self.vector_store.embedder.dim,
            "data_dir": self.data_dir,
        }

    # ------------------------------------------------------------------ 持久化
    def _paths(self) -> Dict[str, str]:
        return {
            "vector": os.path.join(self.data_dir, "vector_store.npz"),
            "keyword": os.path.join(self.data_dir, "keyword_index.json"),
            "graph": os.path.join(self.data_dir, "knowledge_graph.json"),
            "registry": os.path.join(self.data_dir, "kb_registry.json"),
        }

    def save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        p = self._paths()
        self.vector_store.save(p["vector"])
        self.keyword_index.save(p["keyword"])
        self.knowledge_graph.save(p["graph"])
        # 文档注册表单独存 (元数据含 node_id/type, 便于 status 与去重)
        import json
        with open(p["registry"], "w", encoding="utf-8") as f:
            json.dump(
                [{"text": t, "metadata": m} for t, m in self.documents],
                f, ensure_ascii=False, indent=2,
            )

    def load(self) -> bool:
        """从 data_dir 加载; 文件不全则返回 False (保持内存中现状)。"""
        p = self._paths()
        if not (
            os.path.exists(p["vector"]) and os.path.exists(p["vector"][:-4] + ".meta.json")
            and os.path.exists(p["keyword"])
            and os.path.exists(p["graph"])
        ):
            return False
        try:
            self.vector_store.load(p["vector"])
            self.keyword_index.load(p["keyword"])
            self.knowledge_graph.load(p["graph"])
            # 恢复文档注册表
            if os.path.exists(p["registry"]):
                import json
                with open(p["registry"], "r", encoding="utf-8") as f:
                    items = json.load(f)
                self.documents = [(it["text"], it["metadata"]) for it in items]
            else:
                # 从向量库重建
                self.documents = list(zip(self.vector_store.texts, self.vector_store.metadatas))
            # retriever 持有的是引用, 自动同步
            return True
        except Exception as e:  # 加载失败不致命
            print(f"[AerospaceKnowledgeBase] load 失败, 使用全新实例: {e}")
            return False

    def clear(self) -> None:
        """清空文档与倒排索引 (保留知识图谱预填充)。"""
        self.vector_store.clear()
        self.keyword_index.clear()
        self.documents = []


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        kb = AerospaceKnowledgeBase(data_dir=d, autoload=False)
        n = kb.index_default_knowledge()
        print(f"[index_default_knowledge] 索引 {n} 条, 文档总数 {len(kb.documents)}")
        print(f"[status] {kb.status()}")

        q = "地月转移用什么方法"
        print(f"\n[query] {q}")
        for r in kb.query(q, top_k=5):
            tag = r.metadata.get("node_id", r.metadata.get("source", "?"))
            print(f"  [{r.source:>16}] {r.score:.4f}  <{tag}>  {r.text[:48]}")

        # 验收: 结果应含 Hohmann / 拼凑圆锥 / vis-viva
        blob = " ".join(r.text for r in kb.query(q, top_k=5))
        assert "Hohmann" in blob, "缺 Hohmann"
        assert "拼凑圆锥" in blob, "缺 拼凑圆锥"
        assert "vis-viva" in blob or "vis_viva" in blob.lower(), "缺 vis-viva"
        print("\n[ok] 验收通过: 结果包含 Hohmann / 拼凑圆锥 / vis-viva")

        # 持久化往返
        kb.save()
        kb2 = AerospaceKnowledgeBase(data_dir=d, autoload=True, auto_default_knowledge=False)
        print(f"[ok] 持久化: 加载后文档数 {len(kb2.documents)} "
              f"(原 {len(kb.documents)}), 图谱节点 {kb2.knowledge_graph.num_nodes}")
