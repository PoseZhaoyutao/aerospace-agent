"""动态知识云图可视化模块 (Dynamic Knowledge Cloud Visualization)。

把 :class:`KnowledgeGraph` 转换为自包含的交互式 HTML 力导向图,
内嵌 D3.js v7 (从 CDN 加载, 带静态表格 fallback) 与图谱数据 JSON。

核心能力
--------
* :class:`KnowledgeCloudGenerator.generate` —— 生成单帧力导向云图
* :class:`KnowledgeCloudGenerator.generate_temporal_cloud` —— 带时间轴的演进云图
* :class:`KnowledgeCloudGenerator.export_graph_data` —— 图谱 -> JSON 可序列化结构

可视化特性
----------
* 力导向布局: 节点排斥 + 边拉拢 + 中心引力 + 碰撞检测
* 节点按类型着色, 按度数缩放大小
* 鼠标悬停高亮邻居, 点击弹出详情面板, 双击展开/折叠邻居
* 边按 relation 着色与线型区分, 悬停显示关系标签
* 可拖拽节点、滚轮缩放、切换边标签、按类型筛选、搜索高亮
* 深色航天科技风格, 响应式, 入场动画与边渐显

颜色方案与现有报告一致 (紫色品牌色为主)。
"""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["KnowledgeCloudGenerator", "NODE_COLORS", "RELATION_STYLES"]

# ---------------------------------------------------------------------------
# 节点类型 -> 颜色 (与品牌紫色系一致)
# ---------------------------------------------------------------------------
NODE_COLORS: Dict[str, str] = {
    "concept": "#4B3FE3",   # 紫色 (品牌主色)
    "formula": "#22A5F7",   # 蓝色
    "tool": "#27D2BF",      # 青色
    "mission": "#F87454",   # 橙色
    "paper": "#1DC981",     # 绿色
}

# 节点类型中文名
TYPE_NAMES: Dict[str, str] = {
    "concept": "概念",
    "formula": "公式",
    "tool": "工具",
    "mission": "任务",
    "paper": "文献",
}

# ---------------------------------------------------------------------------
# 边关系 -> 样式 (线型 / 颜色 / 中文名)
# ---------------------------------------------------------------------------
RELATION_STYLES: Dict[str, Dict[str, Any]] = {
    "depends_on": {
        "color": "#58a6ff",
        "dash": "none",        # 实线
        "width": 1.5,
        "name": "依赖",
    },
    "used_by": {
        "color": "#d2a8ff",
        "dash": "6,4",         # 虚线
        "width": 1.5,
        "name": "被使用",
    },
    "alternative_to": {
        "color": "#f0883e",
        "dash": "2,4",         # 点线
        "width": 1.5,
        "name": "替代",
    },
    "constraint_of": {
        "color": "#ff7b72",
        "dash": "none",        # 粗线
        "width": 3.0,
        "name": "约束",
    },
}


class KnowledgeCloudGenerator:
    """把 :class:`KnowledgeGraph` 转成交互式 HTML 力导向云图。"""

    DEFAULT_OUTPUT = os.path.join(os.getcwd(), "reports", "knowledge_cloud.html")

    # ==================================================================
    # 数据导出
    # ==================================================================
    def export_graph_data(self, knowledge_graph: Any) -> Dict[str, Any]:
        """把图谱转为 JSON 可序列化的 ``{nodes, links, stats}``。

        - nodes: ``[{id, type, content, metadata, degree, label}, ...]``
        - links: ``[{source, target, relation, weight}, ...]``
        - stats: ``{total_nodes, total_edges, type_counts, top5_connected}``
        """
        # 度数统计 (入度+出度)
        degree_map: Dict[str, int] = defaultdict(int)
        links: List[Dict[str, Any]] = []

        for src, rels in knowledge_graph.adj.items():
            for rel, lst in rels.items():
                for dst, w in lst:
                    links.append({
                        "source": src,
                        "target": dst,
                        "relation": rel,
                        "weight": round(float(w), 3),
                    })
                    degree_map[src] += 1
                    degree_map[dst] += 1

        nodes: List[Dict[str, Any]] = []
        for nid, n in knowledge_graph.nodes.items():
            aliases = n.get("metadata", {}).get("aliases", [])
            # 标签: 优先取第一个中文别名, 否则用 id
            label = nid
            for a in aliases:
                if any("\u4e00" <= ch <= "\u9fff" for ch in a):
                    label = a
                    break
            nodes.append({
                "id": nid,
                "type": n["type"],
                "content": n.get("content", ""),
                "metadata": n.get("metadata", {}),
                "degree": degree_map.get(nid, 0),
                "label": label,
            })

        # 统计
        type_counts: Dict[str, int] = defaultdict(int)
        for n in knowledge_graph.nodes.values():
            type_counts[n["type"]] += 1

        top_nodes = sorted(nodes, key=lambda x: x["degree"], reverse=True)[:5]
        top5 = [
            {"id": n["id"], "type": n["type"], "degree": n["degree"],
             "label": n["label"]}
            for n in top_nodes
        ]

        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "type_counts": dict(type_counts),
            "top5_connected": top5,
        }

        return {"nodes": nodes, "links": links, "stats": stats}

    # ==================================================================
    # 主入口: 单帧云图
    # ==================================================================
    def generate(
        self,
        knowledge_graph: Any,
        output_path: str = DEFAULT_OUTPUT,
        title: str = "航天知识云图",
    ) -> str:
        """生成自包含 HTML 力导向云图, 返回文件绝对路径。"""
        data = self.export_graph_data(knowledge_graph)
        html_content = self._build_html(data, title, snapshots=None)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return os.path.abspath(output_path)

    # ==================================================================
    # 带时间轴的演进云图
    # ==================================================================
    def generate_temporal_cloud(
        self,
        knowledge_graph: Any,
        history: Optional[Sequence[Any]] = None,
        output_path: str = os.path.join(os.getcwd(), "reports", "knowledge_cloud_temporal.html"),
        title: str = "航天知识云图 (时间演进)",
    ) -> str:
        """生成带时间轴的演进版本。

        ``history`` 为知识图谱快照列表, 每项可为:
        - :class:`KnowledgeGraph` 对象
        - ``{"timestamp": str, "graph": KnowledgeGraph}`` 字典

        若 ``history`` 为空则等同于 :meth:`generate`。
        """
        if not history:
            return self.generate(knowledge_graph, output_path, title)

        snapshots: List[Dict[str, Any]] = []
        for i, item in enumerate(history):
            kg = None
            label = None
            if hasattr(item, "nodes") and hasattr(item, "adj"):
                kg = item
                label = f"快照 {i + 1}"
            elif isinstance(item, dict):
                kg = item.get("graph") or item.get("knowledge_graph")
                label = (item.get("timestamp") or item.get("label")
                         or item.get("name") or f"快照 {i + 1}")
            if kg is not None:
                snapshots.append({
                    "label": str(label),
                    "data": self.export_graph_data(kg),
                })

        # 末尾追加当前图谱
        snapshots.append({
            "label": "当前",
            "data": self.export_graph_data(knowledge_graph),
        })

        data = snapshots[-1]["data"]  # 默认显示最后一帧
        html_content = self._build_html(data, title, snapshots=snapshots)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return os.path.abspath(output_path)

    # ==================================================================
    # HTML 构建
    # ==================================================================
    def _build_html(
        self,
        data: Dict[str, Any],
        title: str,
        snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """组装自包含 HTML (CSS + 内嵌数据 + D3 JS + fallback)。"""
        # 安全嵌入 JSON (防止 </script> 破坏)
        data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        colors_json = json.dumps(NODE_COLORS, ensure_ascii=False)
        type_names_json = json.dumps(TYPE_NAMES, ensure_ascii=False)
        rel_styles_json = json.dumps(RELATION_STYLES, ensure_ascii=False)

        snapshots_json = "null"
        if snapshots:
            snapshots_json = json.dumps(snapshots, ensure_ascii=False).replace(
                "</", "<\\/")

        stats_html = self._stats_html(data.get("stats", {}))
        legend_html = self._legend_html()
        fallback_html = self._fallback_table_html(data, title)

        doc = _HTML_TEMPLATE
        doc = doc.replace("%%TITLE%%", html.escape(title))
        doc = doc.replace("%%CSS%%", _CLOUD_CSS)
        doc = doc.replace("%%STATS_HTML%%", stats_html)
        doc = doc.replace("%%LEGEND_HTML%%", legend_html)
        doc = doc.replace("%%FALLBACK_HTML%%", fallback_html)
        # 先插入 JS, 再替换 JS 内部的数据占位符 (%%DATA_JSON%% 等在 _CLOUD_JS 中)
        doc = doc.replace("%%JS%%", _CLOUD_JS)
        doc = doc.replace("%%DATA_JSON%%", data_json)
        doc = doc.replace("%%COLORS_JSON%%", colors_json)
        doc = doc.replace("%%TYPE_NAMES_JSON%%", type_names_json)
        doc = doc.replace("%%REL_STYLES_JSON%%", rel_styles_json)
        doc = doc.replace("%%SNAPSHOTS_JSON%%", snapshots_json)
        return doc

    # ==================================================================
    # 统计区 HTML (静态, 即使 D3 失败也能显示)
    # ==================================================================
    def _stats_html(self, stats: Dict[str, Any]) -> str:
        total_n = stats.get("total_nodes", 0)
        total_e = stats.get("total_edges", 0)
        tc = stats.get("type_counts", {})

        type_chips = []
        for t, name in TYPE_NAMES.items():
            cnt = tc.get(t, 0)
            color = NODE_COLORS.get(t, "#888")
            type_chips.append(
                f'<span class="type-chip" style="border-color:{color};'
                f'color:{color}">{name} <b>{cnt}</b></span>'
            )

        top5 = stats.get("top5_connected", [])
        top5_items = ""
        for i, n in enumerate(top5, 1):
            color = NODE_COLORS.get(n.get("type", ""), "#888")
            top5_items += (
                f'<li><span class="rank">{i}</span>'
                f'<span class="dot" style="background:{color}"></span>'
                f'<span class="nm">{html.escape(str(n.get("label", n.get("id", ""))))}</span>'
                f'<span class="dg">度数 {n.get("degree", 0)}</span></li>'
            )

        return f"""
        <div class="stats-row">
          <div class="stat-box"><span class="sv">{total_n}</span><span class="sl">节点</span></div>
          <div class="stat-box"><span class="sv">{total_e}</span><span class="sl">边</span></div>
          <div class="type-chips">{''.join(type_chips)}</div>
        </div>
        <div class="top5-box">
          <div class="top5-title">最连接节点 Top 5</div>
          <ol class="top5-list">{top5_items}</ol>
        </div>"""

    # ==================================================================
    # 图例 HTML
    # ==================================================================
    def _legend_html(self) -> str:
        # 节点类型图例
        type_items = []
        for t, name in TYPE_NAMES.items():
            color = NODE_COLORS.get(t, "#888")
            type_items.append(
                f'<span class="leg-node"><span class="leg-dot" '
                f'style="background:{color}"></span>{name}</span>'
            )
        # 关系图例
        rel_items = []
        for rel, st in RELATION_STYLES.items():
            dash = st["dash"]
            width = st["width"]
            rel_items.append(
                f'<span class="leg-rel"><svg width="40" height="10">'
                f'<line x1="2" y1="5" x2="38" y2="5" stroke="{st["color"]}" '
                f'stroke-width="{width}" stroke-dasharray="{dash}"/></svg>'
                f'{st["name"]} ({rel})</span>'
            )
        return (
            f'<div class="legend-node">{"&nbsp;".join(type_items)}</div>'
            f'<div class="legend-rel">{"&nbsp;&nbsp;".join(rel_items)}</div>'
        )

    # ==================================================================
    # 静态表格 fallback (D3 加载失败时显示)
    # ==================================================================
    def _fallback_table_html(self, data: Dict[str, Any], title: str) -> str:
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        stats = data.get("stats", {})

        # 节点表
        node_rows = ""
        for n in sorted(nodes, key=lambda x: x.get("degree", 0), reverse=True):
            color = NODE_COLORS.get(n["type"], "#888")
            node_rows += (
                f'<tr><td><span class="fb-dot" style="background:{color}"></span>'
                f'{html.escape(str(n["id"]))}</td>'
                f'<td>{html.escape(TYPE_NAMES.get(n["type"], n["type"]))}</td>'
                f'<td class="num">{n.get("degree", 0)}</td>'
                f'<td>{html.escape(str(n.get("content", ""))[:80])}</td></tr>'
            )
        # 边表
        link_rows = ""
        for l in links:
            st = RELATION_STYLES.get(l["relation"], {})
            link_rows += (
                f'<tr><td>{html.escape(str(l["source"]))}</td>'
                f'<td>{html.escape(str(l["target"]))}</td>'
                f'<td>{html.escape(st.get("name", l["relation"]))}</td>'
                f'<td class="num">{l.get("weight", 1)}</td></tr>'
            )

        return f"""
        <div id="fallback-table" style="display:none;">
          <div class="fb-banner">D3.js 未能从 CDN 加载, 以下为静态表格版本。</div>
          <h3>节点列表 (共 {stats.get("total_nodes", len(nodes))} 个)</h3>
          <table class="fb-table"><thead><tr>
            <th>ID</th><th>类型</th><th>度数</th><th>内容</th>
          </tr></thead><tbody>{node_rows}</tbody></table>
          <h3>边列表 (共 {stats.get("total_edges", len(links))} 条)</h3>
          <table class="fb-table"><thead><tr>
            <th>源节点</th><th>目标节点</th><th>关系</th><th>权重</th>
          </tr></thead><tbody>{link_rows}</tbody></table>
        </div>"""


# ---------------------------------------------------------------------------
# 内联 CSS (深色航天科技风格)
# ---------------------------------------------------------------------------
_CLOUD_CSS = r"""
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #e6edf3;
  font-family: -apple-system, "Segoe UI", "Noto Sans", "Noto Sans CJK SC",
               "WenQuanYi Micro Hei", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px; line-height: 1.6; height: 100vh; overflow: hidden;
  display: flex; flex-direction: column;
}
.app { display: flex; flex-direction: column; height: 100vh; width: 100%; }

/* 顶部栏 */
.topbar {
  background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
  border-bottom: 2px solid #4B3FE3;
  padding: 12px 20px; flex-shrink: 0; z-index: 10;
}
.topbar h1 {
  font-size: 1.3em; color: #a5b4fc; letter-spacing: 1px;
  margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.topbar h1::before { content: ""; display:inline-block; width:10px; height:10px;
  background: #4B3FE3; border-radius: 50%; box-shadow: 0 0 10px #4B3FE3; }
.controls {
  display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin-bottom: 8px;
}
#search-input {
  background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
  color: #e6edf3; padding: 6px 12px; font-size: 0.9em; width: 220px;
  transition: border-color .2s;
}
#search-input:focus { outline: none; border-color: #4B3FE3; box-shadow: 0 0 0 2px rgba(75,63,227,.2); }
.type-filters { display: flex; gap: 8px; flex-wrap: wrap; }
.type-filters label {
  display: flex; align-items: center; gap: 4px; font-size: 0.82em;
  cursor: pointer; padding: 3px 8px; border-radius: 12px;
  border: 1px solid #30363d; transition: all .2s; user-select: none;
}
.type-filters label:hover { border-color: #58a6ff; }
.type-filters input { accent-color: #4B3FE3; width: 13px; height: 13px; }
.switch { font-size: 0.82em; cursor: pointer; display: flex; align-items: center; gap: 4px;
  padding: 3px 8px; border: 1px solid #30363d; border-radius: 12px; user-select: none; }
.switch:hover { border-color: #58a6ff; }
.switch input { accent-color: #4B3FE3; }

/* 统计区 */
.stats-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.stat-box { display: flex; flex-direction: column; align-items: center;
  background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 4px 14px; }
.stat-box .sv { font-size: 1.5em; font-weight: 700; color: #58a6ff; font-family: monospace; }
.stat-box .sl { font-size: 0.7em; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.type-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.type-chip { font-size: 0.76em; padding: 2px 8px; border-radius: 10px;
  border: 1px solid; background: #161b22; }
.type-chip b { font-family: monospace; }
.top5-box { position: absolute; right: 20px; top: 12px; max-width: 240px;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; }
.top5-title { font-size: 0.78em; color: #d2a8ff; margin-bottom: 4px;
  text-transform: uppercase; letter-spacing: 1px; }
.top5-list { list-style: none; }
.top5-list li { display: flex; align-items: center; gap: 6px; font-size: 0.8em; padding: 1px 0; }
.top5-list .rank { width: 16px; height: 16px; background: #4B3FE3; color: #fff;
  border-radius: 50%; text-align: center; font-size: 0.7em; line-height: 16px; font-weight: 700; }
.top5-list .dot { width: 8px; height: 8px; border-radius: 50%; }
.top5-list .nm { flex: 1; color: #c9d1d9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.top5-list .dg { color: #8b949e; font-size: 0.85em; font-family: monospace; }

/* 主区域 */
.main { flex: 1; display: flex; position: relative; overflow: hidden; }
.svg-wrap { flex: 1; position: relative; background: #0d1117; }
#cloud-svg { width: 100%; height: 100%; display: block; cursor: grab; }
#cloud-svg:active { cursor: grabbing; }
.svg-wrap .edge-tooltip {
  position: absolute; pointer-events: none; background: #161b22;
  border: 1px solid #4B3FE3; border-radius: 6px; padding: 4px 10px;
  font-size: 0.8em; color: #e6edf3; box-shadow: 0 4px 12px rgba(0,0,0,.5);
  opacity: 0; transition: opacity .15s; z-index: 5; white-space: nowrap;
}
.node-label { pointer-events: none; text-shadow: 0 1px 3px #0d1117, 0 0 4px #0d1117; }
.edge-label { pointer-events: none; }

/* 详情面板 */
.detail-panel {
  width: 340px; flex-shrink: 0; background: #161b22; border-left: 1px solid #30363d;
  overflow-y: auto; transform: translateX(100%); transition: transform .3s ease;
  display: flex; flex-direction: column;
}
.detail-panel.open { transform: translateX(0); }
.detail-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px; border-bottom: 1px solid #30363d; position: sticky; top: 0;
  background: #161b22; z-index: 2;
}
.detail-header h3 { color: #a5b4fc; font-size: 1.05em; word-break: break-all; }
#detail-close { background: none; border: none; color: #8b949e; font-size: 1.4em;
  cursor: pointer; line-height: 1; padding: 0 4px; }
#detail-close:hover { color: #ff7b72; }
.detail-body { padding: 14px 16px; }
.detail-body .field { margin-bottom: 12px; }
.detail-body .field-label { font-size: 0.72em; color: #6e7681; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 3px; }
.detail-body .field-value { color: #c9d1d9; font-size: 0.92em; word-break: break-word; }
.detail-body .type-badge { display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 0.78em; font-weight: 600; }
.detail-body .conn-list { list-style: none; }
.detail-body .conn-list li { padding: 4px 0; border-bottom: 1px solid #21262d;
  font-size: 0.85em; display: flex; align-items: center; gap: 6px; }
.detail-body .conn-list .rel-tag { font-size: 0.75em; padding: 1px 6px; border-radius: 8px;
  background: #21262d; color: #8b949e; }
.detail-body .conn-list .conn-id { color: #58a6ff; cursor: pointer; }
.detail-body .conn-list .conn-id:hover { text-decoration: underline; }
.detail-body .alias-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.detail-body .alias-chip { font-size: 0.75em; padding: 1px 8px; border-radius: 8px;
  background: #21262d; color: #d2a8ff; }

/* 时间轴 (temporal) */
.timeline-bar {
  display: none; align-items: center; gap: 12px; padding: 8px 20px;
  background: #161b22; border-top: 1px solid #30363d; flex-shrink: 0;
}
.timeline-bar.show { display: flex; }
.timeline-bar label { font-size: 0.82em; color: #8b949e; white-space: nowrap; }
#timeline-slider { flex: 1; accent-color: #4B3FE3; }
#timeline-label { color: #a5b4fc; font-weight: 600; min-width: 80px; }

/* 底部图例 */
.legend {
  background: #161b22; border-top: 1px solid #30363d; padding: 8px 20px;
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  font-size: 0.8em; flex-shrink: 0; justify-content: center;
}
.legend-node, .legend-rel { display: flex; align-items: center; gap: 4px; color: #c9d1d9; }
.leg-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* fallback 表格 */
#fallback-table { padding: 20px; overflow-y: auto; height: 100vh; }
.fb-banner { background: #4a1f1f; color: #ff7b72; padding: 10px 16px;
  border-radius: 6px; margin-bottom: 16px; font-size: 0.9em; }
#fallback-table h3 { color: #a5b4fc; margin: 16px 0 8px; }
.fb-table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin-bottom: 20px; }
.fb-table th, .fb-table td { border: 1px solid #30363d; padding: 6px 10px; text-align: left; }
.fb-table th { background: #21262d; color: #58a6ff; }
.fb-table tr:nth-child(even) td { background: #0d1117; }
.fb-table td.num { text-align: right; font-family: monospace; }
.fb-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: middle; }

/* 响应式 */
@media (max-width: 720px) {
  .top5-box { display: none; }
  .detail-panel { width: 100%; position: absolute; right: 0; top: 0; height: 100%; z-index: 20; }
  #search-input { width: 140px; }
  .controls { gap: 8px; }
}
"""


# ---------------------------------------------------------------------------
# 内联 JavaScript (D3.js v7 力导向图)
# ---------------------------------------------------------------------------
_CLOUD_JS = r"""
(function(){
  "use strict";
  // ---- 注入数据 ----
  var GRAPH_DATA = %%DATA_JSON%%;
  var NODE_COLORS = %%COLORS_JSON%%;
  var TYPE_NAMES = %%TYPE_NAMES_JSON%%;
  var REL_STYLES = %%REL_STYLES_JSON%%;
  var SNAPSHOTS = %%SNAPSHOTS_JSON%%;

  // ---- 全局状态 ----
  var sim, svgSel, zoomLayer, linkSel, linkLabelSel, nodeSel;
  var allNodes, allLinks, neighborMap;
  var showEdgeLabels = false;
  var activeTypes = {};        // type -> bool
  var searchQuery = '';
  var collapsedSet = {};       // id -> true (邻居被折叠)
  var hiddenSet = {};          // id -> true (当前隐藏)
  var selectedId = null;
  var currentWidth, currentHeight;

  // 初始化类型筛选全选
  Object.keys(NODE_COLORS).forEach(function(t){ activeTypes[t] = true; });

  // ---- D3 加载检测 ----
  function boot(){
    if (typeof d3 === 'undefined') {
      document.getElementById('fallback-table').style.display = 'block';
      document.getElementById('cloud-app').style.display = 'none';
      return;
    }
    if (SNAPSHOTS) {
      document.getElementById('timeline-bar').classList.add('show');
      var slider = document.getElementById('timeline-slider');
      slider.max = SNAPSHOTS.length - 1;
      slider.value = SNAPSHOTS.length - 1;
      updateTimelineLabel();
      slider.addEventListener('input', onTimelineChange);
      initCloud(SNAPSHOTS[SNAPSHOTS.length - 1].data);
    } else {
      initCloud(GRAPH_DATA);
    }
    bindControls();
  }

  // ---- 初始化力导向图 ----
  function initCloud(data){
    // 深拷贝避免污染原数据
    allNodes = data.nodes.map(function(n){ return JSON.parse(JSON.stringify(n)); });
    allLinks = data.links.map(function(l){ return JSON.parse(JSON.stringify(l)); });

    // 计算度数
    var deg = {};
    allNodes.forEach(function(n){ deg[n.id] = 0; });
    allLinks.forEach(function(l){
      deg[l.source] = (deg[l.source]||0)+1;
      deg[l.target] = (deg[l.target]||0)+1;
    });
    allNodes.forEach(function(n){ n.degree = deg[n.id]||0; });

    // 邻居表
    neighborMap = {};
    allNodes.forEach(function(n){ neighborMap[n.id] = {}; });
    allLinks.forEach(function(l){
      neighborMap[l.source][l.target] = true;
      neighborMap[l.target][l.source] = true;
    });

    // 清空旧画布
    var svgEl = document.getElementById('cloud-svg');
    while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);

    var wrap = document.getElementById('svg-wrap');
    currentWidth = wrap.clientWidth || 900;
    currentHeight = wrap.clientHeight || 560;

    svgSel = d3.select('#cloud-svg');
    svgSel.attr('viewBox', '0 0 ' + currentWidth + ' ' + currentHeight);

    // 缩放层
    zoomLayer = svgSel.append('g').attr('class', 'zoom-layer');
    var zoom = d3.zoom()
      .scaleExtent([0.15, 6])
      .on('zoom', function(e){ zoomLayer.attr('transform', e.transform); });
    svgSel.call(zoom);

    // 边
    var linkG = zoomLayer.append('g').attr('class', 'links');
    linkSel = linkG.selectAll('line').data(allLinks).enter().append('line')
      .attr('stroke', function(d){ return (REL_STYLES[d.relation]||{}).color || '#666'; })
      .attr('stroke-width', function(d){ return (REL_STYLES[d.relation]||{}).width || 1.5; })
      .attr('stroke-dasharray', function(d){ return (REL_STYLES[d.relation]||{}).dash || 'none'; })
      .attr('opacity', 0)
      .on('mouseover', showEdgeTip)
      .on('mousemove', moveEdgeTip)
      .on('mouseout', hideEdgeTip);

    // 边标签
    var labelG = zoomLayer.append('g').attr('class', 'link-labels');
    linkLabelSel = labelG.selectAll('text').data(allLinks).enter().append('text')
      .attr('class', 'edge-label')
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', '#8b949e')
      .attr('opacity', 0)
      .text(function(d){ return (REL_STYLES[d.relation]||{}).name || d.relation; });

    // 节点
    var nodeG = zoomLayer.append('g').attr('class', 'nodes');
    var maxDeg = d3.max(allNodes, function(d){ return d.degree; }) || 1;
    function rScale(d){ return 7 + 16 * Math.sqrt(d.degree / maxDeg); }

    nodeSel = nodeG.selectAll('g.node').data(allNodes).enter().append('g')
      .attr('class', 'node')
      .style('opacity', 0)
      .call(d3.drag()
        .on('start', dragStart)
        .on('drag', dragging)
        .on('end', dragEnd));

    nodeSel.append('circle')
      .attr('r', rScale)
      .attr('fill', function(d){ return NODE_COLORS[d.type] || '#888'; })
      .attr('stroke', '#0d1117')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer');

    // 节点外发光环 (高亮时用)
    nodeSel.append('circle')
      .attr('class', 'halo')
      .attr('r', function(d){ return rScale(d) + 6; })
      .attr('fill', 'none')
      .attr('stroke', function(d){ return NODE_COLORS[d.type] || '#888'; })
      .attr('stroke-width', 2)
      .attr('opacity', 0);

    nodeSel.append('text')
      .attr('class', 'node-label')
      .attr('dy', function(d){ return -rScale(d) - 5; })
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('fill', '#e6edf3')
      .text(function(d){ return d.label || d.id; });

    // 入场动画: 从中心散开
    allNodes.forEach(function(n){ n.x = currentWidth/2; n.y = currentHeight/2; });

    // 力导向模拟
    sim = d3.forceSimulation(allNodes)
      .force('link', d3.forceLink(allLinks).id(function(d){ return d.id; })
        .distance(85).strength(0.35))
      .force('charge', d3.forceManyBody().strength(-380))
      .force('center', d3.forceCenter(currentWidth/2, currentHeight/2).strength(0.08))
      .force('collide', d3.forceCollide().radius(function(d){ return rScale(d)+8; }).strength(0.9))
      .force('x', d3.forceX(currentWidth/2).strength(0.045))
      .force('y', d3.forceY(currentHeight/2).strength(0.045))
      .alpha(1)
      .alphaDecay(0.022)
      .on('tick', ticked);

    // 节点入场渐显
    nodeSel.transition().duration(700).delay(function(d,i){ return i*25; })
      .style('opacity', 1);
    // 边渐显
    linkSel.transition().duration(900).delay(350).attr('opacity', 0.45);

    // 交互事件
    nodeSel.on('mouseover', function(e,d){
      if (selectedId) return;
      highlightNode(d.id);
    }).on('mouseout', function(e,d){
      if (selectedId) return;
      resetHighlight();
    }).on('click', function(e,d){
      e.stopPropagation();
      selectedId = d.id;
      showDetail(d);
      highlightNode(d.id);
    }).on('dblclick', function(e,d){
      e.stopPropagation();
      e.preventDefault();
      toggleCollapse(d.id);
    });

    svgSel.on('click', function(){
      selectedId = null;
      hideDetail();
      resetHighlight();
    });

    // 初始可见性
    applyVisibility();
    updateEdgeLabels();
  }

  // ---- tick 回调 ----
  function ticked(){
    linkSel
      .attr('x1', function(d){ return d.source.x; })
      .attr('y1', function(d){ return d.source.y; })
      .attr('x2', function(d){ return d.target.x; })
      .attr('y2', function(d){ return d.target.y; });
    nodeSel.attr('transform', function(d){ return 'translate('+d.x+','+d.y+')'; });
    linkLabelSel
      .attr('x', function(d){ return (d.source.x+d.target.x)/2; })
      .attr('y', function(d){ return (d.source.y+d.target.y)/2; });
  }

  // ---- 拖拽 ----
  function dragStart(e, d){
    if (!e.active) sim.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragging(e, d){ d.fx = e.x; d.fy = e.y; }
  function dragEnd(e, d){
    if (!e.active) sim.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  // ---- 高亮节点及邻居 ----
  function highlightNode(id){
    var nbrs = neighborMap[id] || {};
    nodeSel.style('opacity', function(d){
      if (d.id === id || nbrs[d.id]) return 1;
      return 0.15;
    });
    nodeSel.select('circle.halo').attr('opacity', function(d){
      return (d.id === id) ? 0.6 : 0;
    });
    nodeSel.select('circle:not(.halo)').attr('stroke', function(d){
      return (d.id === id) ? '#fff' : '#0d1117';
    });
    linkSel.style('opacity', function(d){
      var s = d.source.id || d.source, t = d.target.id || d.target;
      if (s === id || t === id) return 0.9;
      return 0.06;
    });
    linkLabelSel.style('opacity', function(d){
      if (!showEdgeLabels) return 0;
      var s = d.source.id || d.source, t = d.target.id || d.target;
      return (s === id || t === id) ? 1 : 0;
    });
  }

  function resetHighlight(){
    nodeSel.style('opacity', function(d){
      return isVisible(d) ? 1 : 0.1;
    });
    nodeSel.select('circle.halo').attr('opacity', 0);
    nodeSel.select('circle:not(.halo)').attr('stroke', '#0d1117');
    linkSel.style('opacity', function(d){
      if (!isVisibleById(d.source.id||d.source) || !isVisibleById(d.target.id||d.target)) return 0;
      return 0.45;
    });
    linkLabelSel.style('opacity', function(d){
      if (!showEdgeLabels) return 0;
      if (!isVisibleById(d.source.id||d.source) || !isVisibleById(d.target.id||d.target)) return 0;
      return 0.7;
    });
    applySearchHighlight();
  }

  // ---- 可见性 (类型筛选 + 折叠) ----
  function isVisible(d){
    if (!activeTypes[d.type]) return false;
    if (hiddenSet[d.id]) return false;
    return true;
  }
  function isVisibleById(id){
    for (var i=0;i<allNodes.length;i++){
      if (allNodes[i].id === id) return isVisible(allNodes[i]);
    }
    return false;
  }

  function applyVisibility(){
    hiddenSet = {};
    // 折叠: 隐藏被折叠节点的直接邻居 (仅当该邻居不被其他可见节点连接时保留)
    Object.keys(collapsedSet).forEach(function(cid){
      if (!collapsedSet[cid]) return;
      var nbrs = neighborMap[cid] || {};
      Object.keys(nbrs).forEach(function(nid){
        // 仅隐藏叶子邻居 (度数低且只被折叠节点连接)
        hiddenSet[nid] = true;
      });
    });
    nodeSel.style('opacity', function(d){ return isVisible(d) ? 1 : 0.08; })
           .style('pointer-events', function(d){ return isVisible(d) ? 'all' : 'none'; });
    linkSel.style('opacity', function(d){
      var s = d.source.id||d.source, t = d.target.id||d.target;
      if (!isVisibleById(s) || !isVisibleById(t)) return 0;
      return 0.45;
    });
    linkLabelSel.style('opacity', function(d){
      if (!showEdgeLabels) return 0;
      var s = d.source.id||d.source, t = d.target.id||d.target;
      if (!isVisibleById(s) || !isVisibleById(t)) return 0;
      return 0.7;
    });
    applySearchHighlight();
  }

  // ---- 双击折叠/展开邻居 ----
  function toggleCollapse(id){
    if (collapsedSet[id]) {
      delete collapsedSet[id];
    } else {
      collapsedSet[id] = true;
    }
    applyVisibility();
    sim.alpha(0.5).restart();
  }

  // ---- 搜索高亮 ----
  function applySearchHighlight(){
    if (!searchQuery) return;
    var q = searchQuery.toLowerCase();
    nodeSel.select('circle:not(.halo)').attr('stroke', function(d){
      var hay = (d.id+' '+d.label+' '+d.content+' '+
        (d.metadata.aliases||[]).join(' ')).toLowerCase();
      if (hay.indexOf(q) >= 0) return '#fbbf24';
      return '#0d1117';
    });
    nodeSel.select('circle:not(.halo)').attr('stroke-width', function(d){
      var hay = (d.id+' '+d.label+' '+d.content+' '+
        (d.metadata.aliases||[]).join(' ')).toLowerCase();
      return (hay.indexOf(q) >= 0) ? 3 : 2;
    });
  }

  // ---- 详情面板 ----
  function showDetail(d){
    var panel = document.getElementById('detail-panel');
    panel.classList.add('open');
    document.getElementById('detail-title').textContent = d.label || d.id;
    var color = NODE_COLORS[d.type] || '#888';
    var typeName = TYPE_NAMES[d.type] || d.type;

    // 连接列表
    var conns = [];
    allLinks.forEach(function(l){
      var s = l.source.id||l.source, t = l.target.id||l.target;
      if (s === d.id) conns.push({id:t, rel:l.relation, dir:'→'});
      else if (t === d.id) conns.push({id:s, rel:l.relation, dir:'←'});
    });
    var connHtml = conns.map(function(c){
      var rn = (REL_STYLES[c.rel]||{}).name || c.rel;
      var rc = (REL_STYLES[c.rel]||{}).color || '#666';
      return '<li><span class="rel-tag" style="color:'+rc+'">'+c.dir+' '+rn+'</span>'
        +'<span class="conn-id" data-id="'+esc(c.id)+'">'+esc(c.id)+'</span></li>';
    }).join('');

    // 别名
    var aliases = (d.metadata && d.metadata.aliases) || [];
    var aliasHtml = aliases.map(function(a){
      return '<span class="alias-chip">'+esc(a)+'</span>';
    }).join('');

    var body = document.getElementById('detail-body');
    body.innerHTML =
      '<div class="field"><div class="field-label">类型</div>'
      +'<div class="field-value"><span class="type-badge" style="background:'+color+'22;color:'+color+'">'
      +esc(typeName)+' ('+esc(d.type)+')</span></div></div>'
      +'<div class="field"><div class="field-label">节点 ID</div>'
      +'<div class="field-value"><code>'+esc(d.id)+'</code></div></div>'
      +'<div class="field"><div class="field-label">内容</div>'
      +'<div class="field-value">'+esc(d.content||'')+'</div></div>'
      + (aliasHtml ? '<div class="field"><div class="field-label">别名</div>'
        +'<div class="field-value alias-chips">'+aliasHtml+'</div></div>' : '')
      +'<div class="field"><div class="field-label">度数</div>'
      +'<div class="field-value">'+d.degree+'</div></div>'
      +'<div class="field"><div class="field-label">关联节点 ('+conns.length+')</div>'
      +'<ul class="conn-list">'+connHtml+'</ul></div>';

    // 连接节点可点击跳转
    body.querySelectorAll('.conn-id').forEach(function(el){
      el.addEventListener('click', function(){
        var targetId = el.getAttribute('data-id');
        var target = allNodes.find(function(n){ return n.id === targetId; });
        if (target) {
          selectedId = targetId;
          showDetail(target);
          highlightNode(targetId);
        }
      });
    });
  }

  function hideDetail(){
    document.getElementById('detail-panel').classList.remove('open');
  }

  document.getElementById('detail-close').addEventListener('click', function(e){
    e.stopPropagation();
    selectedId = null;
    hideDetail();
    resetHighlight();
  });

  // ---- 边 tooltip ----
  var tipEl = document.getElementById('edge-tooltip');
  function showEdgeTip(e, d){
    var rn = (REL_STYLES[d.relation]||{}).name || d.relation;
    var s = d.source.id||d.source, t = d.target.id||d.target;
    tipEl.innerHTML = '<b style="color:#a5b4fc">'+esc(rn)+'</b> ('+esc(d.relation)+')<br>'
      + esc(s)+' → '+esc(t) + (d.weight!=null ? '<br>权重: '+d.weight : '');
    tipEl.style.opacity = 1;
  }
  function moveEdgeTip(e){
    var rect = document.getElementById('svg-wrap').getBoundingClientRect();
    tipEl.style.left = (e.clientX - rect.left + 12) + 'px';
    tipEl.style.top = (e.clientY - rect.top + 12) + 'px';
  }
  function hideEdgeTip(){ tipEl.style.opacity = 0; }

  // ---- 边标签开关 ----
  function updateEdgeLabels(){
    var op = showEdgeLabels ? 0.7 : 0;
    linkLabelSel.style('opacity', function(d){
      if (!isVisibleById(d.source.id||d.source) || !isVisibleById(d.target.id||d.target)) return 0;
      return op;
    });
  }

  // ---- 绑定控件 ----
  function bindControls(){
    // 搜索
    document.getElementById('search-input').addEventListener('input', function(e){
      searchQuery = e.target.value.trim();
      if (searchQuery) {
        applySearchHighlight();
        // 淡化不匹配
        var q = searchQuery.toLowerCase();
        nodeSel.style('opacity', function(d){
          if (!isVisible(d)) return 0.08;
          var hay = (d.id+' '+d.label+' '+d.content+' '+
            (d.metadata.aliases||[]).join(' ')).toLowerCase();
          return (hay.indexOf(q) >= 0) ? 1 : 0.2;
        });
      } else {
        applyVisibility();
      }
    });

    // 类型筛选
    document.querySelectorAll('.type-filters input').forEach(function(cb){
      cb.addEventListener('change', function(){
        activeTypes[cb.getAttribute('data-type')] = cb.checked;
        applyVisibility();
      });
    });

    // 边标签开关
    document.getElementById('toggle-edge-labels').addEventListener('change', function(e){
      showEdgeLabels = e.target.checked;
      updateEdgeLabels();
    });

    // 重置布局
    document.getElementById('btn-reset').addEventListener('click', function(){
      if (sim) {
        allNodes.forEach(function(n){ n.fx=null; n.fy=null; });
        sim.alpha(1).restart();
      }
    });

    // 窗口缩放
    window.addEventListener('resize', function(){
      var wrap = document.getElementById('svg-wrap');
      var w = wrap.clientWidth || 900, h = wrap.clientHeight || 560;
      if (sim && (w !== currentWidth || h !== currentHeight)) {
        currentWidth = w; currentHeight = h;
        svgSel.attr('viewBox', '0 0 '+w+' '+h);
        sim.force('center', d3.forceCenter(w/2, h/2).strength(0.08));
        sim.force('x', d3.forceX(w/2).strength(0.045));
        sim.force('y', d3.forceY(h/2).strength(0.045));
        sim.alpha(0.5).restart();
      }
    });
  }

  // ---- 时间轴 ----
  function onTimelineChange(){
    var idx = parseInt(document.getElementById('timeline-slider').value, 10);
    updateTimelineLabel();
    initCloud(SNAPSHOTS[idx].data);
  }
  function updateTimelineLabel(){
    var idx = parseInt(document.getElementById('timeline-slider').value, 10);
    var snap = SNAPSHOTS[idx];
    document.getElementById('timeline-label').textContent =
      snap.label + ' (' + snap.data.stats.total_nodes + '节点)';
  }

  // ---- HTML 转义 ----
  function esc(s){
    if (s==null) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // ---- 启动 ----
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
"""


# ---------------------------------------------------------------------------
# HTML 模板 (用占位符注入, 避免 JS 花括号与 format 冲突)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>%%TITLE%%</title>
<style>%%CSS%%</style>
</head>
<body>
<div id="cloud-app" class="app">
  <!-- 顶部栏 -->
  <header class="topbar">
    <h1>%%TITLE%%</h1>
    <div class="controls">
      <input type="search" id="search-input" placeholder="搜索节点 (ID / 内容 / 别名)..." autocomplete="off">
      <div class="type-filters">
        <label><input type="checkbox" data-type="concept" checked><span style="color:#4B3FE3">●</span> 概念</label>
        <label><input type="checkbox" data-type="formula" checked><span style="color:#22A5F7">●</span> 公式</label>
        <label><input type="checkbox" data-type="tool" checked><span style="color:#27D2BF">●</span> 工具</label>
        <label><input type="checkbox" data-type="mission" checked><span style="color:#F87454">●</span> 任务</label>
        <label><input type="checkbox" data-type="paper" checked><span style="color:#1DC981">●</span> 文献</label>
      </div>
      <label class="switch"><input type="checkbox" id="toggle-edge-labels"> 边标签</label>
      <button class="switch" id="btn-reset" style="border:none;cursor:pointer;background:#21262d;color:#c9d1d9">重置布局</button>
    </div>
    <div style="position:relative;">
      %%STATS_HTML%%
    </div>
  </header>

  <!-- 主区域 -->
  <main class="main">
    <div class="svg-wrap" id="svg-wrap">
      <svg id="cloud-svg"></svg>
      <div class="edge-tooltip" id="edge-tooltip"></div>
    </div>
    <aside class="detail-panel" id="detail-panel">
      <div class="detail-header">
        <h3 id="detail-title">节点详情</h3>
        <button id="detail-close">&times;</button>
      </div>
      <div class="detail-body" id="detail-body"></div>
    </aside>
  </main>

  <!-- 时间轴 (temporal 模式) -->
  <div class="timeline-bar" id="timeline-bar">
    <label>时间轴</label>
    <input type="range" id="timeline-slider" min="0" max="0" step="1" value="0">
    <span id="timeline-label">—</span>
  </div>

  <!-- 底部图例 -->
  <footer class="legend">
    %%LEGEND_HTML%%
  </footer>
</div>

<!-- D3 加载失败时的静态表格 -->
%%FALLBACK_HTML%%

<!-- D3.js v7 (CDN) -->
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<!-- 云图逻辑 (数据内嵌) -->
<script>%%JS%%</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 自测: 用预填充知识图谱生成云图
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .knowledge_graph import KnowledgeGraph

    print("=== knowledge_cloud 自测 ===")
    kg = KnowledgeGraph()
    kg.prepopulate()
    print(f"知识图谱: 节点={kg.num_nodes}, 边={kg.num_edges}")

    gen = KnowledgeCloudGenerator()

    # 导出数据
    data = gen.export_graph_data(kg)
    print(f"导出数据: nodes={len(data['nodes'])}, links={len(data['links'])}")
    print(f"统计: {data['stats']}")

    # 生成云图
    out = gen.generate(kg)
    size_kb = os.path.getsize(out) / 1024
    print(f"\n知识云图已生成: {out}")
    print(f"文件大小: {size_kb:.1f} KB")

    # 核心 Top 5
    print("\n核心概念 Top 5 (按度数):")
    for i, n in enumerate(data["stats"]["top5_connected"], 1):
        print(f"  {i}. {n['label']} [{n['type']}] 度数={n['degree']}")

    print("\nknowledge_cloud 自测通过.")
