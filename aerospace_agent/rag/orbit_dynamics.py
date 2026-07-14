"""Seed corpus for an orbit-dynamics expert RAG library."""

from __future__ import annotations

from typing import Any, Dict, List


ORBIT_DYNAMICS_SEED_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "topic": "two_body_dynamics",
        "slug": "two-body-orbital-dynamics",
        "title": "Two-body orbital dynamics",
        "related_topics": ["perturbations", "validation"],
        "text": (
            "Two-body dynamics models spacecraft motion under a central gravity field. "
            "The governing acceleration is -mu r / |r|^3. Keplerian elements are useful "
            "for compact orbit description, while Cartesian states are preferred for "
            "numerical propagation and covariance operations. Assumptions: point-mass "
            "gravity, no drag, no third-body perturbation, no finite burn."
        ),
    },
    {
        "topic": "frames_and_time",
        "slug": "reference-frames-and-time-scales",
        "title": "Reference frames and time scales",
        "related_topics": ["orbit_determination", "sensor_truth_mapping"],
        "text": (
            "Orbit software must state frame and time scale explicitly. Common frames "
            "include ECI/GCRF, ECEF/ITRF, TEME, LVLH, and sensor frames. Common time "
            "scales include UTC, TAI, TT, and TDB. Frame transforms and time conversions "
            "are not optional metadata; they affect position, velocity, pointing, and "
            "truth/image alignment."
        ),
    },
    {
        "topic": "perturbations",
        "slug": "perturbed-orbit-propagation",
        "title": "Perturbed orbit propagation",
        "related_topics": ["two_body_dynamics", "validation"],
        "text": (
            "High-fidelity orbit propagation should declare force models: gravity degree "
            "and order, J2/J3 terms, atmospheric drag, solar radiation pressure, third-body "
            "gravity, solid tides, maneuvers, and integrator tolerances. Unsupported force "
            "models must be reported as unavailable rather than silently mocked."
        ),
    },
    {
        "topic": "orbit_determination",
        "slug": "orbit-determination-and-measurements",
        "title": "Orbit determination and measurements",
        "related_topics": ["frames_and_time", "validation"],
        "text": (
            "Orbit determination estimates state and uncertainty from observations such "
            "as range, range-rate, optical angles, bearings, or image detections. A useful "
            "RAG answer should separate measurement model, dynamic model, estimator, "
            "prior assumptions, residuals, and validation data."
        ),
    },
    {
        "topic": "validation",
        "slug": "propagation-validation",
        "title": "Propagation validation",
        "related_topics": ["perturbations", "orbit_determination"],
        "text": (
            "Orbit propagation validation should compare independent engines or analytic "
            "limits when possible. Minimal checks include units, epoch consistency, frame "
            "consistency, conserved energy for two-body cases, expected nodal precession "
            "for J2 cases, and bounded interpolation error."
        ),
    },
    {
        "topic": "sensor_truth_mapping",
        "slug": "truth-to-sensor-mapping",
        "title": "Truth to sensor mapping",
        "related_topics": ["frames_and_time", "orbit_determination"],
        "text": (
            "Space-based image simulation must preserve traceability from propagated truth "
            "states to camera-frame line-of-sight vectors, detector coordinates, PSF, SNR, "
            "exposure, gain, noise model, and generated image pixels. Claims about weak "
            "target detectability require explicit photometric assumptions."
        ),
    },
]


def build_orbit_dynamics_seed_texts() -> List[str]:
    return [
        f"[{doc['topic']}] {doc['title']}\n{doc['text']}"
        for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS
    ]


def index_orbit_dynamics_corpus(rag: Any) -> Dict[str, Any]:
    """Index the orbit-dynamics seed corpus into a RAG object."""

    if rag is None or not hasattr(rag, "index"):
        return {
            "status": "unavailable",
            "error_code": "RAG_NOT_AVAILABLE",
            "indexed_count": 0,
            "topics": [doc["topic"] for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS],
        }

    indexed = 0
    for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS:
        text = f"[{doc['topic']}] {doc['title']}\n{doc['text']}"
        try:
            rag.index(text, source=f"orbit_dynamics:{doc['topic']}")
            indexed += 1
        except TypeError:
            rag.index(text)
            indexed += 1

    return {
        "status": "ok",
        "indexed_count": indexed,
        "topics": [doc["topic"] for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS],
    }


def ingest_pdf_to_rag(
    pdf_path: str,
    rag: Any,
    topic: str = "imported",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """从 PDF 文件提取文本并索引入 RAG。

    支持 PyPDF2 和 pdfplumber 两种后端，自动回退。
    文本按 chunk_size 分块，块间有 chunk_overlap 重叠。

    Args:
        pdf_path: PDF 文件路径
        rag: RAG 实例（需有 index() 方法）
        topic: 索引入的主题标签
        chunk_size: 每块最大字符数
        chunk_overlap: 块间重叠字符数

    Returns:
        {status, chunks_indexed, errors}
    """
    import os

    if not os.path.exists(pdf_path):
        return {"status": "error", "reason": f"PDF 文件不存在: {pdf_path}"}

    if rag is None or not hasattr(rag, "index"):
        return {"status": "error", "reason": "RAG 不可用或缺少 index() 方法"}

    # 尝试提取文本
    text = _extract_pdf_text(pdf_path)
    if not text.strip():
        return {"status": "error", "reason": "PDF 提取文本为空"}

    # 分块
    chunks = _chunk_text(text, chunk_size, chunk_overlap)
    indexed = 0
    errors = []

    for i, chunk in enumerate(chunks):
        try:
            title = os.path.basename(pdf_path).replace(".pdf", "")
            rag.index(
                f"[{topic}] {title} (chunk {i+1}/{len(chunks)})\n{chunk}",
                source=f"pdf:{topic}:{os.path.basename(pdf_path)}:chunk{i}",
            )
            indexed += 1
        except TypeError:
            try:
                rag.index(
                    f"[{topic}] {title} (chunk {i+1}/{len(chunks)})\n{chunk}",
                )
                indexed += 1
            except Exception as e:
                errors.append(f"chunk {i}: {e}")
        except Exception as e:
            errors.append(f"chunk {i}: {e}")

    return {
        "status": "ok" if indexed > 0 else "error",
        "pdf_path": pdf_path,
        "total_chunks": len(chunks),
        "chunks_indexed": indexed,
        "errors": errors,
    }


def _extract_pdf_text(pdf_path: str) -> str:
    """从 PDF 提取文本，自动选择可用后端。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取的文本字符串
    """
    # 尝试 pdfplumber（质量更好）
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception:
        pass

    # 回退到 PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception:
        pass

    # 最后回退到 pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        return ""
    except Exception:
        return ""


def _chunk_text(
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list:
    """将文本按字符数分块。

    Args:
        text: 输入文本
        chunk_size: 每块最大字符数
        chunk_overlap: 块间重叠字符数

    Returns:
        文本块列表
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 尽量在完整句子边界断开
        if end < len(text):
            # 往回找最近的句号、换行
            for sep in ["\n\n", "\n", "。", ". "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + chunk_size // 2:
                    end = last_sep + len(sep)
                    break

        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start >= len(text):
            break
        # 防止无限循环：当 end 已经是末尾时强制退出
        if end >= len(text):
            break

    return chunks
