"""文献搜索与获取模块 (Literature Fetcher)。

本模块实现航天文献的自动搜索、下载与全文提取, 是 RAG 系统的「数据入口」:

* :class:`CSTCloudAuthenticator` — 中国科技云 (CSTCloud) OpenAI 兼容 API 认证器,
  支持从环境变量读取密钥、验证有效性, 未配置时自动回退到 ``create_llm()``。
* :class:`ArxivFetcher`           — arXiv 文献搜索器, 调用 arXiv 公开 API (Atom XML),
  支持字段检索 (all/ti/au/cat) 与 AND/OR 组合, 严格遵守 3 秒/次限速。
* :class:`Paper`                  — 论文数据类, 封装 arXiv 条目的全部元信息。
* :func:`extract_text_from_pdf`   — PDF 全文提取, 优先 PyPDF2/pypdf, 回退 pdftotext。

arXiv API 说明
--------------
* 查询接口: ``http://export.arxiv.org/api/query?search_query=...&start=0&max_results=...``
* 返回 Atom XML (默认命名空间 ``http://www.w3.org/2005/Atom``,
  arxiv 命名空间 ``http://arxiv.org/schemas/atom``)
* 限速: 每次请求间隔至少 3 秒
* PDF 下载: ``https://arxiv.org/pdf/{arxiv_id}``
"""

from __future__ import annotations

import os
import re
import time
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aerospace_agent.local_runtime import run_command

__all__ = [
    "Paper",
    "CSTCloudAuthenticator",
    "ArxivFetcher",
    "extract_text_from_pdf",
]

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
DEFAULT_PAPER_DIR = os.path.join(os.getcwd(), "data", "papers")

# arXiv API 基址
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"

# arXiv 限速: 请求间隔 (秒), 官方要求至少 3 秒
ARXIV_RATE_LIMIT_SECONDS = 3.0

# Atom XML 命名空间
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


# ---------------------------------------------------------------------------
# Paper 数据类
# ---------------------------------------------------------------------------
@dataclass
class Paper:
    """论文数据类, 封装 arXiv 条目的全部元信息。

    Attributes:
        id:               arXiv ID, 如 ``2501.17867v1``
        title:            论文标题
        authors:          作者列表
        abstract:         摘要全文
        categories:       arXiv 分类列表, 如 ``["astro-ph.IM", "astro-ph.EP"]``
        published_date:   发表日期 (ISO 8601 字符串)
        pdf_url:          PDF 下载链接
        arxiv_url:        arXiv 摘要页面链接
        doi:              DOI (可能为空)
        primary_category: 主分类
    """

    id: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    categories: List[str] = field(default_factory=list)
    published_date: str = ""
    pdf_url: str = ""
    arxiv_url: str = ""
    doi: Optional[str] = None
    primary_category: Optional[str] = None

    def __repr__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" 等{len(self.authors)}人"
        return (
            f"Paper(id={self.id!r}, title={self.title[:60]!r}, "
            f"authors=[{authors_str}], primary={self.primary_category})"
        )

    @property
    def short_id(self) -> str:
        """不带版本号的 arXiv ID, 如 ``2501.17867``。"""
        return re.sub(r"v\d+$", "", self.id)


# ---------------------------------------------------------------------------
# CSTCloud 认证器
# ---------------------------------------------------------------------------
class CSTCloudAuthenticator:
    """中国科技云 (CSTCloud) OpenAI 兼容 API 认证器。

    从环境变量读取 ``CSTCLOUD_API_KEY`` / ``CSTCLOUD_BASE_URL``,
    验证密钥有效性。若未配置 CSTCloud 密钥, 自动回退到
    :func:`aerospace_agent.core.llm_interface.create_llm`。

    用法::

        auth = CSTCloudAuthenticator()
        if auth.login():
            print("CSTCloud 认证成功")
        llm = auth.get_llm()  # 返回 LLMInterface (CSTCloud 或回退)
    """

    DEFAULT_BASE_URL = "https://api.cstcloud.cn/v1"

    def __init__(self):
        self.api_key: Optional[str] = os.environ.get("CSTCLOUD_API_KEY")
        self.base_url: str = (
            os.environ.get("CSTCLOUD_BASE_URL") or self.DEFAULT_BASE_URL
        )
        self._authenticated: bool = False

    # ------------------------------------------------------------------ 登录
    def login(self, api_key: Optional[str] = None,
              base_url: Optional[str] = None) -> bool:
        """验证 API key 有效性。

        发送一个简单请求 (GET /models) 测试密钥。成功返回 True。
        若未提供密钥 (环境变量也没有), 直接返回 False (回退模式)。

        Args:
            api_key:  可选, 覆盖环境变量
            base_url: 可选, 覆盖环境变量

        Returns:
            认证是否成功
        """
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url

        # 未配置密钥 -> 回退模式
        if not self.api_key:
            print("[CSTCloudAuthenticator] 未配置 CSTCLOUD_API_KEY, "
                  "回退到默认 LLM (create_llm)")
            self._authenticated = False
            return False

        # 发送测试请求: GET {base_url}/models
        url = self.base_url.rstrip("/") + "/models"
        try:
            req = urllib.request.Request(
                url,
                headers=self.get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    self._authenticated = True
                    print(f"[CSTCloudAuthenticator] 认证成功: {self.base_url}")
                    return True
                else:
                    print(f"[CSTCloudAuthenticator] 认证失败: HTTP {resp.status}")
                    self._authenticated = False
                    return False
        except urllib.error.HTTPError as e:
            print(f"[CSTCloudAuthenticator] 认证失败: HTTP {e.code} {e.reason}")
            self._authenticated = False
            return False
        except urllib.error.URLError as e:
            print(f"[CSTCloudAuthenticator] 网络错误: {e.reason}")
            self._authenticated = False
            return False
        except Exception as e:
            print(f"[CSTCloudAuthenticator] 认证异常: {e}")
            self._authenticated = False
            return False

    # ------------------------------------------------------------------ 请求头
    def get_headers(self) -> dict:
        """返回带 Authorization Bearer 的请求头。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------ 状态
    @property
    def is_authenticated(self) -> bool:
        """是否已通过 CSTCloud 认证。"""
        return self._authenticated

    # ------------------------------------------------------------------ 获取 LLM
    def get_llm(self):
        """返回 LLM 接口实例。

        若已通过 CSTCloud 认证, 返回基于 CSTCloud 凭据的
        ``OpenAICompatibleLLM``; 否则回退到 ``create_llm()``
        (可能使用 AEROSPACE_LLM_API_KEY 或 MockLLM)。
        """
        # 延迟导入避免循环依赖
        from ..core.llm_interface import OpenAICompatibleLLM, create_llm

        if self._authenticated and self.api_key:
            return OpenAICompatibleLLM(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        # 回退到默认 LLM
        return create_llm()


# ---------------------------------------------------------------------------
# arXiv 文献搜索器
# ---------------------------------------------------------------------------
class ArxivFetcher:
    """arXiv 文献搜索器。

    调用 arXiv 公开 API (Atom XML), 支持字段检索与 AND/OR 组合,
    严格遵守 3 秒/次限速。

    用法::

        fetcher = ArxivFetcher()
        papers = fetcher.search("lunar transfer orbit", max_results=5)
        path = fetcher.download_pdf(papers[0])
    """

    # 类级共享: 上次请求时间 (arXiv 限速是按 IP 的)
    _last_request_time: float = 0.0

    def __init__(self, paper_dir: str = DEFAULT_PAPER_DIR):
        self.paper_dir = paper_dir
        os.makedirs(self.paper_dir, exist_ok=True)

    # ------------------------------------------------------------------ 限速
    @classmethod
    def rate_limit(cls) -> None:
        """保证两次请求间隔至少 ``ARXIV_RATE_LIMIT_SECONDS`` 秒。"""
        now = time.time()
        elapsed = now - cls._last_request_time
        if elapsed < ARXIV_RATE_LIMIT_SECONDS:
            wait = ARXIV_RATE_LIMIT_SECONDS - elapsed
            time.sleep(wait)
        cls._last_request_time = time.time()

    # ------------------------------------------------------------------ 搜索
    def search(
        self,
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
    ) -> List[Paper]:
        """搜索 arXiv 文献。

        Args:
            query:       搜索关键词。支持以下格式:
                         * ``"lunar transfer orbit"``  (普通字符串 -> 自动转为 all:"...")
                         * ``all:lunar transfer``      (带字段前缀, 原样使用)
                         * ``ti:orbit AND au:smith``   (AND/OR 组合, 原样使用)
            max_results: 最大返回条数
            category:    可选, arXiv 分类过滤 (如 ``astro-ph.IM``), 追加为 ``AND cat:...``

        Returns:
            论文列表; 网络不可用时返回空列表
        """
        # 构建 search_query
        search_query = self._build_search_query(query, category)
        params = urllib.parse.urlencode({
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        })
        url = f"{ARXIV_API_URL}?{params}"
        print(f"[ArxivFetcher] 搜索: {search_query}")
        print(f"[ArxivFetcher] 请求: {url}")

        # 限速
        self.rate_limit()

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AerospaceAgent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            print(f"[ArxivFetcher] HTTP 错误: {e.code} {e.reason}")
            return []
        except urllib.error.URLError as e:
            print(f"[ArxivFetcher] 网络错误 (arXiv 不可达): {e.reason}")
            return []
        except Exception as e:
            print(f"[ArxivFetcher] 搜索异常: {e}")
            return []

        # 解析 Atom XML
        papers = self._parse_atom(xml_data)
        print(f"[ArxivFetcher] 找到 {len(papers)} 篇文献")
        return papers

    # ------------------------------------------------------------------ 构建查询
    @staticmethod
    def _build_search_query(query: str, category: Optional[str]) -> str:
        """构建 arXiv search_query 字符串。

        规则:
        * 若 query 已含字段前缀 (``all:``/``ti:``/``au:``/``cat:``/``abs:``) 或
          布尔运算符 (AND/OR), 原样使用;
        * 若 query 用引号包裹 (如 ``"lunar transfer orbit"``), 做精确短语匹配
          ``all:"..."``;
        * 否则作为普通多词查询, 拆词后用 AND 连接
          (如 ``lunar transfer orbit`` -> ``all:lunar AND all:transfer AND all:orbit``);
        * 若指定 category, 追加 ``AND cat:{category}``。
        """
        has_prefix = bool(re.search(r"\b(all|ti|au|cat|abs):", query))
        has_operator = bool(re.search(r"\b(AND|OR)\b", query, re.IGNORECASE))

        if has_prefix or has_operator:
            # 用户已指定格式, 原样使用
            sq = query
        elif query.startswith('"') and query.endswith('"'):
            # 引号包裹 -> 精确短语
            sq = f"all:{query}"
        else:
            # 普通多词查询 -> AND 连接
            words = query.split()
            if len(words) == 1:
                sq = f"all:{words[0]}"
            else:
                sq = " AND ".join(f"all:{w}" for w in words)

        if category:
            sq = f"{sq} AND cat:{category}"

        return sq

    # ------------------------------------------------------------------ XML 解析
    @staticmethod
    def _parse_atom(xml_data: str) -> List[Paper]:
        """解析 arXiv Atom XML 响应, 返回 Paper 列表。

        XML 结构::

            <feed xmlns="http://www.w3.org/2005/Atom"
                  xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <id>http://arxiv.org/abs/2501.17867v1</id>
                <title>...</title>
                <summary>...</summary>
                <published>2025-01-15T01:26:37Z</published>
                <link rel="alternate" href="..." type="text/html"/>
                <link rel="related" href="..." type="application/pdf" title="pdf"/>
                <author><name>...</name></author>
                <category term="astro-ph.IM"/>
                <arxiv:primary_category term="astro-ph.IM"/>
                <arxiv:doi>10.xxx/xxx</arxiv:doi>
              </entry>
            </feed>
        """
        papers: List[Paper] = []
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            print(f"[ArxivFetcher] XML 解析失败: {e}")
            return papers

        # 查找所有 entry 元素 (在 atom 命名空间下)
        for entry in root.findall("atom:entry", _NS):
            try:
                paper = ArxivFetcher._parse_entry(entry)
                if paper:
                    papers.append(paper)
            except Exception as e:
                print(f"[ArxivFetcher] 解析条目异常: {e}")
                continue

        return papers

    @staticmethod
    def _parse_entry(entry: ET.Element) -> Optional[Paper]:
        """解析单个 <entry> 元素为 Paper 对象。"""
        # id: http://arxiv.org/abs/2501.17867v1 -> 2501.17867v1
        id_elem = entry.find("atom:id", _NS)
        if id_elem is None or not id_elem.text:
            return None
        full_id = id_elem.text.strip()
        arxiv_id = full_id.rsplit("/", 1)[-1]

        # title
        title_elem = entry.find("atom:title", _NS)
        title = (title_elem.text or "").strip() if title_elem is not None else ""
        # 去除多余空白
        title = re.sub(r"\s+", " ", title)

        # summary (摘要)
        summary_elem = entry.find("atom:summary", _NS)
        abstract = ""
        if summary_elem is not None and summary_elem.text:
            abstract = re.sub(r"\s+", " ", summary_elem.text.strip())

        # published 日期
        pub_elem = entry.find("atom:published", _NS)
        published_date = (pub_elem.text or "").strip() if pub_elem is not None else ""

        # authors
        authors: List[str] = []
        for author_elem in entry.findall("atom:author", _NS):
            name_elem = author_elem.find("atom:name", _NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # categories
        categories: List[str] = []
        for cat_elem in entry.findall("atom:category", _NS):
            term = cat_elem.get("term")
            if term:
                categories.append(term)

        # primary category (arxiv 命名空间)
        primary_cat_elem = entry.find("arxiv:primary_category", _NS)
        primary_category = None
        if primary_cat_elem is not None:
            primary_category = primary_cat_elem.get("term")

        # DOI (arxiv 命名空间)
        doi_elem = entry.find("arxiv:doi", _NS)
        doi = None
        if doi_elem is not None and doi_elem.text:
            doi = doi_elem.text.strip()

        # links: rel="alternate" -> arxiv_url, rel="related" title="pdf" -> pdf_url
        arxiv_url = full_id  # 默认用 id URL
        pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}"
        for link_elem in entry.findall("atom:link", _NS):
            rel = link_elem.get("rel", "")
            href = link_elem.get("href", "")
            link_title = link_elem.get("title", "")
            if rel == "alternate":
                arxiv_url = href
            elif rel == "related" and (link_title == "pdf" or "pdf" in href):
                pdf_url = href

        return Paper(
            id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            published_date=published_date,
            pdf_url=pdf_url,
            arxiv_url=arxiv_url,
            doi=doi,
            primary_category=primary_category,
        )

    # ------------------------------------------------------------------ 下载 PDF
    def download_pdf(self, paper: Paper,
                     save_dir: str = DEFAULT_PAPER_DIR) -> Optional[str]:
        """下载论文 PDF 到本地。

        Args:
            paper:    论文对象
            save_dir: 保存目录, 默认 ``/workspace/data/papers``

        Returns:
            成功返回 PDF 本地路径; 失败返回 None
        """
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, f"{paper.id}.pdf")

        # 若已下载, 直接返回
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            print(f"[ArxivFetcher] PDF 已存在: {pdf_path}")
            return pdf_path

        # 限速
        self.rate_limit()

        url = paper.pdf_url or f"{ARXIV_PDF_BASE}/{paper.id}"
        print(f"[ArxivFetcher] 下载 PDF: {url}")

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AerospaceAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdf_data = resp.read()

            if len(pdf_data) < 1000:
                print(f"[ArxivFetcher] PDF 内容过小 ({len(pdf_data)} 字节), 可能无效")
                return None

            with open(pdf_path, "wb") as f:
                f.write(pdf_data)

            print(f"[ArxivFetcher] PDF 已保存: {pdf_path} "
                  f"({len(pdf_data) / 1024:.0f} KB)")
            return pdf_path

        except urllib.error.HTTPError as e:
            print(f"[ArxivFetcher] 下载失败 HTTP {e.code}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            print(f"[ArxivFetcher] 下载网络错误: {e.reason}")
            return None
        except Exception as e:
            print(f"[ArxivFetcher] 下载异常: {e}")
            return None


# ---------------------------------------------------------------------------
# PDF 全文提取
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str, max_chars: int = 50000) -> str:
    """从 PDF 文件提取全文文本。

    提取策略 (按优先级):
    1. ``PyPDF2``  (若已安装)
    2. ``pypdf``   (PyPDF2 的继任者, API 兼容)
    3. ``fitz``    (PyMuPDF, 若已安装)
    4. ``pdftotext`` (subprocess 调用命令行工具)
    5. 回退: 返回 PDF 文件元信息 (路径、大小) 作为占位文本

    Args:
            pdf_path:  PDF 文件路径
        max_chars: 提取文本最大字符数 (防止超大 PDF 撑爆内存)

    Returns:
        提取到的全文文本; 全部方法失败时返回文件元信息字符串
    """
    if not os.path.exists(pdf_path):
        return f"(PDF 文件不存在: {pdf_path})"

    # ---- 策略 1: PyPDF2 ----
    try:
        import PyPDF2  # type: ignore
        text = _extract_with_pypdf2_like(pdf_path, PyPDF2, max_chars)
        if text and len(text.strip()) > 50:
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[extract_text_from_pdf] PyPDF2 提取失败: {e}")

    # ---- 策略 2: pypdf (PyPDF2 继任者) ----
    try:
        import pypdf  # type: ignore
        text = _extract_with_pypdf2_like(pdf_path, pypdf, max_chars)
        if text and len(text.strip()) > 50:
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[extract_text_from_pdf] pypdf 提取失败: {e}")

    # ---- 策略 3: fitz (PyMuPDF) ----
    try:
        import fitz  # type: ignore
        text = _extract_with_fitz(pdf_path, max_chars)
        if text and len(text.strip()) > 50:
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[extract_text_from_pdf] fitz 提取失败: {e}")

    # ---- 策略 4: pdftotext (subprocess) ----
    text = _extract_with_pdftotext(pdf_path, max_chars)
    if text and len(text.strip()) > 50:
        return text

    # ---- 策略 5: 回退到文件元信息 ----
    file_size = os.path.getsize(pdf_path)
    print(f"[extract_text_from_pdf] 所有提取方法均失败, 返回文件元信息")
    return (
        f"(无法提取 PDF 全文, 仅文件元信息)\n"
        f"文件路径: {pdf_path}\n"
        f"文件大小: {file_size / 1024:.0f} KB\n"
        f"提取方法: PyPDF2/pypdf/fitz/pdftotext 均不可用或失败"
    )


def _extract_with_pypdf2_like(pdf_path: str, module, max_chars: int) -> str:
    """使用 PyPDF2 或 pypdf (API 兼容) 提取文本。"""
    reader = module.PdfReader(pdf_path)
    parts: List[str] = []
    total = 0
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        parts.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def _extract_with_fitz(pdf_path: str, max_chars: int) -> str:
    """使用 PyMuPDF (fitz) 提取文本。"""
    import fitz  # type: ignore
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    total = 0
    for page in doc:
        page_text = page.get_text()
        parts.append(page_text)
        total += len(page_text)
        if total >= max_chars:
            break
    doc.close()
    return "\n".join(parts)[:max_chars]


def _extract_with_pdftotext(pdf_path: str, max_chars: int) -> str:
    """使用 pdftotext 命令行工具提取文本。"""
    try:
        result = run_command(
            ["pdftotext", "-q", pdf_path, "-"],
            timeout=60,
        )
        if result.ok and result.stdout:
            return result.stdout[:max_chars]
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        print("[extract_text_from_pdf] pdftotext 超时")
    except Exception as e:
        print(f"[extract_text_from_pdf] pdftotext 异常: {e}")
    return ""


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ArxivFetcher 自测 (查询: lunar transfer orbit)")
    print("=" * 70)

    fetcher = ArxivFetcher()

    # 测试 1: arXiv 搜索
    papers = fetcher.search("lunar transfer orbit", max_results=5)
    print(f"\n--- 搜索结果: {len(papers)} 篇 ---")
    for i, p in enumerate(papers, 1):
        print(f"  {i}. [{p.id}] {p.title}")
        print(f"     作者: {', '.join(p.authors[:3])}")
        print(f"     分类: {p.primary_category} | 发表: {p.published_date[:10]}")
        print(f"     PDF:  {p.pdf_url}")
        print(f"     摘要: {p.abstract[:120]}...")
        print()

    if not papers:
        print("(未找到文献, 可能网络不可用)")
    else:
        # 测试 2: PDF 下载 (仅下载第一篇)
        print("--- PDF 下载测试 ---")
        pdf_path = fetcher.download_pdf(papers[0])
        if pdf_path:
            print(f"下载成功: {pdf_path}")

            # 测试 3: PDF 文本提取
            print("\n--- PDF 文本提取测试 ---")
            text = extract_text_from_pdf(pdf_path)
            print(f"提取文本长度: {len(text)} 字符")
            print(f"前 300 字符:\n{text[:300]}")
        else:
            print("下载失败 (网络可能受限)")

    # 测试 4: CSTCloud 认证器
    print("\n" + "=" * 70)
    print("CSTCloudAuthenticator 自测")
    print("=" * 70)
    auth = CSTCloudAuthenticator()
    ok = auth.login()
    print(f"认证状态: {auth.is_authenticated}")
    if not auth.is_authenticated:
        llm = auth.get_llm()
        print(f"回退 LLM 类型: {type(llm).__name__}")
