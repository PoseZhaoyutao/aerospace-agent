"""航天知识图谱 (Aerospace Knowledge Graph) —— 三路混合检索的核心创新。

向量库只懂「字面相似」, 关键词索引只懂「词项命中」, 二者都无法表达
**概念间的推导依赖**。例如问「地月转移用什么方法」, 向量库可能只命中
含「地月转移」字面的文档, 却不会主动联想到 Hohmann → vis-viva → 能量守恒
这条推导链。知识图谱用显式的 ``depends_on`` / ``used_by`` 边补上这一环。

数据模型
--------
节点 (Node):
    * type ∈ {concept, formula, tool, mission}
    * content: 自然语言描述 (会被检索器当作候选文档)
    * metadata: 任意附加信息, 推荐 ``aliases`` 字段放中英文别名以利概念匹配

边 (Edge):
    * relation ∈ {depends_on, used_by, alternative_to, constraint_of}
    * weight: 关系强度 (默认 1.0)

物理正确性
----------
``prepopulate()`` 内置的公式与依赖关系均与 ``aerospace_agent.physics`` 子包
一致 (vis-viva / Hohmann / 拼凑圆锥 / 开普勒方程 / SOI 等), 见各节点 content。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

__all__ = ["KnowledgeGraph"]

# 节点 / 边类型常量
NODE_CONCEPT = "concept"
NODE_FORMULA = "formula"
NODE_TOOL = "tool"
NODE_MISSION = "mission"

REL_DEPENDS_ON = "depends_on"        # A 依赖于 B (推导上需要 B)
REL_USED_BY = "used_by"              # A 被 B 使用
REL_ALTERNATIVE_TO = "alternative_to"  # A 与 B 互为替代
REL_CONSTRAINT_OF = "constraint_of"  # A 是 B 的约束


class KnowledgeGraph:
    """邻接表式航天知识图谱。"""

    def __init__(self):
        # nodes[id] = {"type":..., "content":..., "metadata":...}
        self.nodes: Dict[str, dict] = {}
        # 邻接表: adj[src][relation] = [(dst, weight), ...]
        self.adj: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # 反向邻接表 (便于 used_by 等反向遍历)
        self.radj: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------ 节点
    def add_node(
        self, id: str, type: str, content: str, metadata: Optional[dict] = None
    ) -> None:
        """添加/更新节点。"""
        self.nodes[id] = {
            "type": type,
            "content": content,
            "metadata": dict(metadata) if metadata else {},
        }
        # 保证邻接表条目存在
        _ = self.adj[id]
        _ = self.radj[id]

    def has_node(self, id: str) -> bool:
        return id in self.nodes

    # ------------------------------------------------------------------ 边
    def add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0) -> None:
        """添加有向边 src -[relation]-> dst。"""
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(
                f"边端点不存在: src={src!r} dst={dst!r}; 请先 add_node"
            )
        self.adj[src][relation].append((dst, float(weight)))
        self.radj[dst][relation].append((src, float(weight)))

    def neighbors(self, id: str, relation: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """返回 id 的出边邻居 [(dst, relation, weight), ...]。"""
        out: List[Tuple[str, str, float]] = []
        for rel, lst in self.adj.get(id, {}).items():
            if relation is not None and rel != relation:
                continue
            for dst, w in lst:
                out.append((dst, rel, w))
        return out

    # ------------------------------------------------------------------ BFS 子图
    def query(self, start_id: str, depth: int = 2) -> dict:
        """从 start_id 出发 BFS, 取 depth 跳内的子图。

        返回 ``{"nodes": [id,...], "edges": [(src,dst,rel,weight),...],
        "distances": {id: 距离}}``。找不到 start_id 返回空子图。
        """
        if start_id not in self.nodes:
            return {"nodes": [], "edges": [], "distances": {}}
        visited: Set[str] = {start_id}
        distances = {start_id: 0}
        edges: List[Tuple[str, str, str, float]] = []
        q: deque = deque([(start_id, 0)])
        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            for dst, rel, w in self.neighbors(cur):
                edges.append((cur, dst, rel, w))
                if dst not in visited:
                    visited.add(dst)
                    distances[dst] = d + 1
                    q.append((dst, d + 1))
        return {
            "nodes": sorted(visited),
            "edges": edges,
            "distances": distances,
        }

    # ------------------------------------------------------------------ 最短路径
    def find_path(self, src: str, dst: str) -> List[str]:
        """BFS 找 src 到 dst 的最短路径 (跨所有边类型)。返回节点 id 列表; 无路径返回 []。"""
        if src not in self.nodes or dst not in self.nodes:
            return []
        if src == dst:
            return [src]
        prev: Dict[str, str] = {}
        visited: Set[str] = {src}
        q: deque = deque([src])
        while q:
            cur = q.popleft()
            for nxt, _rel, _w in self.neighbors(cur):
                if nxt in visited:
                    continue
                visited.add(nxt)
                prev[nxt] = cur
                if nxt == dst:
                    # 回溯路径
                    path = [dst]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        return []

    def explain(self, topic: str, max_depth: int = 4) -> str:
        """生成某主题的「知识链」解释文本。

        沿 ``depends_on`` 边自顶向下展开推导依赖, 形成
        例如「地月转移 → 拼凑圆锥 → 二体问题 → 能量守恒」的可读链。
        同时附带 used_by / constraint_of 关系概览。
        """
        start = self._resolve_id(topic)
        if start is None:
            return f"(未在知识图谱中找到主题: {topic})"

        lines: List[str] = []
        node = self.nodes[start]
        lines.append(f"【{start}】 [{node['type']}] {node['content']}")

        # 1) depends_on 推导链 (DFS, 去环)
        lines.append("  推导依赖链 (depends_on):")
        seen: Set[str] = set()

        def _dfs(nid: str, depth: int, prefix: str):
            if depth > max_depth:
                return
            deps = [d for d, r, _w in self.neighbors(nid) if r == REL_DEPENDS_ON]
            for i, d in enumerate(deps):
                if d in seen:
                    continue
                seen.add(d)
                last = (i == len(deps) - 1)
                branch = "└─" if last else "├─"
                dnode = self.nodes[d]
                lines.append(
                    f"  {prefix}{branch}→ {d} [{dnode['type']}] {dnode['content']}"
                )
                _dfs(d, depth + 1, prefix + ("   " if last else "│  "))

        seen.add(start)
        _dfs(start, 0, "")

        # 2) 关系概览 (反向+正向)
        #    used_by 边 X -> start 表示「X 被 start 使用」, 即 start 使用了 X
        uses = [s for s, _r, _w in self._in_edges(start, REL_USED_BY)]
        #    constraint_of 边 X -> start 表示「X 约束 start」
        constrained_by = [s for s, _r, _w in self._in_edges(start, REL_CONSTRAINT_OF)]
        #    alternative_to 双向, 取出邻即可
        alts = [d for d, r, _w in self.neighbors(start) if r == REL_ALTERNATIVE_TO]
        if uses:
            lines.append(f"  使用 (used_by 入边): {', '.join(uses)}")
        if constrained_by:
            lines.append(f"  受约束于 (constraint_of 入边): {', '.join(constrained_by)}")
        if alts:
            lines.append(f"  替代 (alternative_to): {', '.join(alts)}")
        return "\n".join(lines)

    def _in_edges(self, id: str, relation: str) -> List[Tuple[str, str, float]]:
        """返回指向 id 且关系为 relation 的入边 [(src, relation, weight), ...]。

        radj[id][relation] = [(src, weight), ...]
        """
        lst = self.radj.get(id, {}).get(relation, [])
        return [(src, relation, w) for (src, w) in lst]

    # ------------------------------------------------------------------ 概念匹配
    def _resolve_id(self, topic: str) -> Optional[str]:
        """把自然语言 topic 解析为节点 id: 先精确 id, 再别名/子串匹配。"""
        if topic in self.nodes:
            return topic
        t = topic.lower().strip()
        # 别名精确匹配
        for nid, n in self.nodes.items():
            for alias in n["metadata"].get("aliases", []):
                if alias.lower() == t:
                    return nid
        # 子串匹配 (中文按字, 英文按词)
        for nid, n in self.nodes.items():
            cand = (nid + " " + n["content"] + " " + " ".join(n["metadata"].get("aliases", []))).lower()
            if topic and topic.lower() in cand:
                return nid
        return None

    def match_concepts(self, query: str) -> List[Tuple[str, float]]:
        """在 query 中识别命中的图谱概念, 返回 [(node_id, score), ...]。

        用于混合检索的「图谱路」: 对每个概念取 BFS 邻域作为候选。
        score = 1.0 (精确 id/别名命中) 或按字面覆盖度递减。
        """
        q = query.lower()
        hits: List[Tuple[str, float]] = []
        for nid, n in self.nodes.items():
            aliases = [nid] + list(n["metadata"].get("aliases", []))
            best = 0.0
            for a in aliases:
                al = a.lower()
                if not al:
                    continue
                if al in q:
                    # 别名越长且命中, 分数越高
                    best = max(best, min(1.0, 0.4 + 0.1 * len(al)))
            if best > 0:
                hits.append((nid, best))
        hits.sort(key=lambda kv: kv[1], reverse=True)
        return hits

    # ------------------------------------------------------------------ 统计
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return sum(
            len(lst) for rels in self.adj.values() for lst in rels.values()
        )

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        edges = []
        for src, rels in self.adj.items():
            for rel, lst in rels.items():
                for dst, w in lst:
                    edges.append({"src": src, "dst": dst, "relation": rel, "weight": w})
        data = {"nodes": self.nodes, "edges": edges}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.nodes = {}
        self.adj = defaultdict(lambda: defaultdict(list))
        self.radj = defaultdict(lambda: defaultdict(list))
        for nid, n in data.get("nodes", {}).items():
            self.add_node(nid, n["type"], n["content"], n.get("metadata", {}))
        for e in data.get("edges", []):
            self.add_edge(e["src"], e["dst"], e["relation"], e.get("weight", 1.0))

    def clear(self) -> None:
        self.nodes = {}
        self.adj = defaultdict(lambda: defaultdict(list))
        self.radj = defaultdict(lambda: defaultdict(list))

    # ------------------------------------------------------------------ 预填充
    def prepopulate(self) -> None:
        """内置航天知识图谱预填充 (物理正确, 与 physics 子包一致)。

        节点 25 个, 覆盖: 开普勒方程、vis-viva、Hohmann 转移、Lambert 问题、
        拼凑圆锥、SOI、C3 能量、TLI、LOI、发射窗口、相位角、orekit/gmat/
        spiceypy/basilisk/stk、二体问题、角动量守恒、能量守恒、地月转移等。
        """
        # ---- 节点 ----
        nodes = [
            # 基础物理 (concept)
            ("two_body", NODE_CONCEPT,
             "二体问题: 两质点在万有引力下的运动, 是所有轨道力学的基础; "
             "解析解为圆锥曲线轨道",
             {"aliases": ["二体", "二体问题", "two-body", "two body"]}),
            ("angular_momentum", NODE_CONCEPT,
             "角动量守恒: 中心引力场下比角动量 h = r × v 守恒, 轨道位于固定平面内",
             {"aliases": ["角动量", "角动量守恒", "angular momentum"]}),
            ("energy_conservation", NODE_CONCEPT,
             "能量守恒: 比机械能 ε = v²/2 - μ/r = -μ/(2a) 守恒",
             {"aliases": ["能量守恒", "机械能", "energy conservation"]}),
            ("orbital_elements", NODE_CONCEPT,
             "经典轨道根数: 半长轴 a, 偏心率 e, 倾角 i, 升交点赤经 Ω, "
             "近地点幅角 ω, 真近点角 ν, 共六要素描述轨道",
             {"aliases": ["轨道根数", "轨道要素", "orbital elements", "六要素"]}),
            ("soi", NODE_CONCEPT,
             "引力作用球 SOI: r_SOI = a·(μ_moon/μ_earth)^(2/5), "
             "某天体引力主导的范围, 是拼凑圆锥的分段边界",
             {"aliases": ["soi", "引力作用球", "作用球", "sphere of influence"]}),
            ("c3", NODE_CONCEPT,
             "特征能量 C3 = v_inf²: 逃逸双曲剩余速度的平方, 衡量发射能量需求",
             {"aliases": ["c3", "特征能量", "characteristic energy", "v_inf"]}),
            ("delta_v", NODE_CONCEPT,
             "速度增量 Δv 预算: 任务总冲量需求, 由 vis-viva 方程估算各段",
             {"aliases": ["delta-v", "delta v", "速度增量", "dv", "Δv"]}),
            ("phase_angle", NODE_CONCEPT,
             "相位角: 发射时刻目标天体相对出发点的角位置, 决定转移时机与发射窗口",
             {"aliases": ["相位角", "phase angle", "相位"]}),
            ("launch_window", NODE_CONCEPT,
             "发射窗口: 满足相位角/光照/测控等约束的可发射时段",
             {"aliases": ["发射窗口", "launch window", "窗口"]}),
            ("hohmann", NODE_CONCEPT,
             "Hohmann 转移: 共面圆轨道间最省能量的双脉冲椭圆转移, "
             "半长轴 a_t = (r1+r2)/2, 由 vis-viva 求 Δv",
             {"aliases": ["hohmann", "霍曼", "霍曼转移", "hohmann转移", "hohmann transfer"]}),
            ("lambert", NODE_CONCEPT,
             "Lambert 问题: 已知两端点位置向量与飞行时间, 求连接轨道; "
             "Hohmann 的通用化 (非共面/任意时间)",
             {"aliases": ["lambert", "兰伯特", "lambert问题", "lambert problem"]}),
            ("patched_conic", NODE_CONCEPT,
             "拼凑圆锥近似: 把多体问题拆成二体段, 在 SOI 边界做速度匹配, "
             "地月转移的核心简化方法",
             {"aliases": ["拼凑圆锥", "patched conic", "圆锥拼接", "拼接圆锥"]}),
            ("true_anomaly", NODE_CONCEPT,
             "真近点角 ν: 近地点到当前位置的真角, 直接描述位置",
             {"aliases": ["真近点角", "true anomaly", "ν"]}),
            ("mean_anomaly", NODE_CONCEPT,
             "平近点角 M = n·(t-tp): 平均运动乘时间, 与时间线性, 输入开普勒方程",
             {"aliases": ["平近点角", "mean anomaly", "M"]}),
            ("eccentric_anomaly", NODE_CONCEPT,
             "偏近点角 E: 辅助变量, 经开普勒方程 M = E - e·sinE 与 M 关联",
             {"aliases": ["偏近点角", "eccentric anomaly", "E"]}),
            # 公式 (formula)
            ("kepler_equation", NODE_FORMULA,
             "开普勒方程: M = E - e·sinE, 由平近点角 M 迭代求偏近点角 E, "
             "再得真近点角 ν, 用于二体位置/时间转换",
             {"aliases": ["开普勒方程", "kepler equation", "kepler"]}),
            ("vis_viva", NODE_FORMULA,
             "vis-viva (活力) 方程: v² = μ(2/r - 1/a), 任意圆锥轨道任意点半径处速度, "
             "由能量守恒导出, 是 Δv 计算核心",
             {"aliases": ["vis-viva", "vis viva", "活力公式", "活力方程", "visviva"]}),
            # 工具 (tool)
            ("orekit", NODE_TOOL,
             "Orekit: CNES 开源的航天动力学基础库 (Java/Python), 支持二体/数值传播/"
             "历书, 本项目 astropy_tool/orekit_tool 即封装之",
             {"aliases": ["orekit"]}),
            ("gmat", NODE_TOOL,
             "GMAT: NASA 开源任务分析工具, 支持拼凑圆锥/Lambert 优化/数值积分, "
             "gmat_tool 封装之",
             {"aliases": ["gmat"]}),
            ("spiceypy", NODE_TOOL,
             "SPICE (NAIF): NASA 行星历书/姿态/事件工具, spiceypy 为 Python 封装, "
             "提供月球/行星精确星历, 用于发射窗口计算",
             {"aliases": ["spiceypy", "spice", "历书"]}),
            ("basilisk", NODE_TOOL,
             "Basilisk: CU 多体/航天器 GNC 仿真框架, 支持六自由度与多体动力学, "
             "basilisk_tool 封装之",
             {"aliases": ["basilisk"]}),
            ("stk", NODE_TOOL,
             "STK: AGI 商业航天任务分析软件, 行业标准, stk_tool 封装之",
             {"aliases": ["stk"]}),
            # 任务 (mission)
            ("tli", NODE_MISSION,
             "地月注入 TLI (Trans-Lunar Injection): 把航天器从地球停泊轨道送入"
             "地月转移轨道的脉冲, 需 C3/v_inf 与 vis-viva 估算 Δv",
             {"aliases": ["tli", "地月注入", "trans-lunar injection", "trans lunar"]}),
            ("loi", NODE_MISSION,
             "环月注入 LOI (Lunar Orbit Insertion): 在近月点制动, "
             "把双曲接近轨道捕获为环月轨道, Δv 由 vis-viva 计算",
             {"aliases": ["loi", "环月注入", "lunar orbit insertion"]}),
            ("earth_moon_transfer", NODE_MISSION,
             "地月转移: 从地球到月球的轨道设计, 核心方法为拼凑圆锥近似 + "
             "Hohmann/精确实例; 流程 TLI → 地心段 → SOI → 月心段 → LOI, "
             "需发射窗口(相位角)匹配",
             {"aliases": ["地月转移", "earth-moon transfer", "earth moon transfer",
                          "登月转移", "月球转移"]}),
        ]
        for nid, typ, content, meta in nodes:
            self.add_node(nid, typ, content, meta)

        # ---- 边 (物理正确的推导依赖) ----
        # depends_on: A 推导上需要 B
        dep_edges = [
            ("two_body", "angular_momentum"),
            ("two_body", "energy_conservation"),
            ("orbital_elements", "two_body"),
            ("kepler_equation", "two_body"),
            ("kepler_equation", "mean_anomaly"),
            ("kepler_equation", "eccentric_anomaly"),
            ("true_anomaly", "two_body"),
            ("mean_anomaly", "two_body"),
            ("eccentric_anomaly", "two_body"),
            ("vis_viva", "energy_conservation"),
            ("vis_viva", "two_body"),
            ("hohmann", "vis_viva"),
            ("hohmann", "kepler_equation"),
            ("hohmann", "two_body"),
            ("lambert", "two_body"),
            ("patched_conic", "two_body"),
            ("patched_conic", "soi"),
            ("c3", "vis_viva"),
            ("delta_v", "vis_viva"),
            ("tli", "vis_viva"),
            ("tli", "c3"),
            ("loi", "vis_viva"),
            ("phase_angle", "orbital_elements"),
            ("launch_window", "phase_angle"),
            ("launch_window", "orbital_elements"),
            ("earth_moon_transfer", "patched_conic"),
            ("earth_moon_transfer", "hohmann"),
            ("earth_moon_transfer", "tli"),
            ("earth_moon_transfer", "loi"),
            ("earth_moon_transfer", "launch_window"),
            ("earth_moon_transfer", "phase_angle"),
        ]
        for a, b in dep_edges:
            self.add_edge(a, b, REL_DEPENDS_ON, 1.0)

        # used_by: A 被 B 使用 (反向写: B used_by A -> 我们存 A->B 的 used_by
        # 语义即“A 的使用者是 B”不直观; 这里统一存 src used_by dst 表示
        # “src 被 dst 使用”。为可读, 下面直接表达“工具/公式 被谁用”)
        used_by_edges = [
            ("kepler_equation", "hohmann"),
            ("vis_viva", "hohmann"),
            ("vis_viva", "tli"),
            ("vis_viva", "loi"),
            ("vis_viva", "c3"),
            ("vis_viva", "delta_v"),
            ("patched_conic", "earth_moon_transfer"),
            ("hohmann", "earth_moon_transfer"),
            ("lambert", "earth_moon_transfer"),
            ("spiceypy", "earth_moon_transfer"),
            ("kepler_equation", "orekit"),
            ("vis_viva", "orekit"),
            ("lambert", "gmat"),
            ("patched_conic", "gmat"),
            ("kepler_equation", "gmat"),
        ]
        for a, b in used_by_edges:
            self.add_edge(a, b, REL_USED_BY, 1.0)

        # alternative_to: 互为替代
        alt_edges = [
            ("lambert", "hohmann"),
            ("hohmann", "lambert"),
            ("basilisk", "orekit"),
            ("orekit", "basilisk"),
            ("stk", "gmat"),
            ("gmat", "stk"),
        ]
        for a, b in alt_edges:
            self.add_edge(a, b, REL_ALTERNATIVE_TO, 0.8)

        # constraint_of: A 是 B 的约束
        constraint_edges = [
            ("c3", "tli"),
            ("phase_angle", "launch_window"),
            ("launch_window", "earth_moon_transfer"),
            ("delta_v", "earth_moon_transfer"),
            ("soi", "patched_conic"),
        ]
        for a, b in constraint_edges:
            self.add_edge(a, b, REL_CONSTRAINT_OF, 0.9)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    kg = KnowledgeGraph()
    kg.prepopulate()
    print(f"[预填充] 节点数 = {kg.num_nodes}, 边数 = {kg.num_edges}")

    # BFS 子图
    sub = kg.query("earth_moon_transfer", depth=2)
    print(f"[BFS depth=2 from earth_moon_transfer] 节点: {len(sub['nodes'])}, "
          f"边: {len(sub['edges'])}")

    # 最短路径
    path = kg.find_path("earth_moon_transfer", "energy_conservation")
    print(f"[path] earth_moon_transfer -> energy_conservation: {' -> '.join(path)}")

    # 概念匹配
    for q in ["地月转移用什么方法", "地月转移发射窗口", "vis-viva 怎么推导"]:
        hits = kg.match_concepts(q)
        print(f"[match] {q!r} -> {hits[:5]}")

    # 知识链解释
    print("\n[explain 地月转移]")
    print(kg.explain("地月转移"))

    # 持久化
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kg.json")
        kg.save(p)
        kg2 = KnowledgeGraph()
        kg2.load(p)
        print(f"\n[ok] 持久化: 加载后节点={kg2.num_nodes}, 边={kg2.num_edges} "
              f"(原 {kg.num_nodes}/{kg.num_edges})")
