"""文件与 IO 操作工具集（15 个原子工具）。

第一性原理：
  1. 文件系统操作是不可分解的科研基础设施（读/写/查/压缩）
  2. 所有返回值 JSON 可序列化——工具输出可直接链式传入下一个工具
  3. 标准库实现（os/shutil/zipfile/glob），零第三方依赖
  4. 统一错误协议：失败时返回 {"status":"error","reason":"..."}
"""
from __future__ import annotations

import os
import glob as _glob
import shutil
import zipfile
from datetime import datetime

from aerospace_agent.research_tools.base import register_tool


@register_tool("save_file", "保存文本内容到文件", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"},
                       {"name": "content", "type": "str", "description": "文件内容"},
                       {"name": "encoding", "type": "str", "description": "编码",
                        "required": False, "default": "utf-8"}])
def save_file(path, content, encoding="utf-8"):
    """保存文本内容到文件。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return {"status": "success", "path": os.path.abspath(path), "size": len(content)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("read_file", "读取文件文本内容", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"},
                       {"name": "encoding", "type": "str", "description": "编码",
                        "required": False, "default": "utf-8"}])
def read_file(path, encoding="utf-8"):
    """读取文件文本内容。"""
    try:
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        return {"status": "success", "path": os.path.abspath(path),
                "content": content, "size": len(content)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("append_file", "追加内容到文件末尾", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"},
                       {"name": "content", "type": "str", "description": "要追加的内容"},
                       {"name": "encoding", "type": "str", "description": "编码",
                        "required": False, "default": "utf-8"}])
def append_file(path, content, encoding="utf-8"):
    """追加内容到文件末尾（文件不存在则创建）。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding=encoding) as f:
            f.write(content)
        size = os.path.getsize(path)
        return {"status": "success", "path": os.path.abspath(path),
                "appended_size": len(content), "total_size": size}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("delete_file", "删除文件", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"}])
def delete_file(path):
    """删除指定文件。"""
    try:
        if not os.path.exists(path):
            return {"status": "error", "reason": f"文件不存在: {path}"}
        if os.path.isdir(path):
            return {"status": "error", "reason": f"目标是目录而非文件: {path}"}
        os.remove(path)
        return {"status": "success", "path": os.path.abspath(path), "deleted": True}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("copy_file", "复制文件", "file_io",
               params=[{"name": "src", "type": "str", "description": "源文件路径"},
                       {"name": "dst", "type": "str", "description": "目标文件路径"}])
def copy_file(src, dst):
    """复制文件到新位置。"""
    try:
        if not os.path.exists(src):
            return {"status": "error", "reason": f"源文件不存在: {src}"}
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        shutil.copy2(src, dst)
        return {"status": "success", "src": os.path.abspath(src),
                "dst": os.path.abspath(dst), "size": os.path.getsize(dst)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("move_file", "移动或重命名文件", "file_io",
               params=[{"name": "src", "type": "str", "description": "源文件路径"},
                       {"name": "dst", "type": "str", "description": "目标文件路径"}])
def move_file(src, dst):
    """移动或重命名文件。"""
    try:
        if not os.path.exists(src):
            return {"status": "error", "reason": f"源文件不存在: {src}"}
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        shutil.move(src, dst)
        return {"status": "success", "src": os.path.abspath(src),
                "dst": os.path.abspath(dst)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("list_directory", "列出目录内容", "file_io",
               params=[{"name": "path", "type": "str", "description": "目录路径"},
                       {"name": "pattern", "type": "str", "description": "文件名匹配模式",
                        "required": False, "default": "*"},
                       {"name": "include_hidden", "type": "bool", "description": "是否包含隐藏文件",
                        "required": False, "default": False}])
def list_directory(path=".", pattern="*", include_hidden=False):
    """列出目录内容。"""
    try:
        if not os.path.isdir(path):
            return {"status": "error", "reason": f"目录不存在: {path}"}
        entries = []
        for name in sorted(os.listdir(path)):
            if not include_hidden and name.startswith("."):
                continue
            if not _glob.fnmatch.fnmatch(name, pattern):
                continue
            full = os.path.join(path, name)
            entries.append({
                "name": name,
                "type": "dir" if os.path.isdir(full) else "file",
                "size": os.path.getsize(full) if os.path.isfile(full) else None,
            })
        return {"status": "success", "path": os.path.abspath(path),
                "count": len(entries), "entries": entries}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("create_directory", "创建目录", "file_io",
               params=[{"name": "path", "type": "str", "description": "目录路径"},
                       {"name": "parents", "type": "bool", "description": "是否递归创建父目录",
                        "required": False, "default": True}])
def create_directory(path, parents=True):
    """创建目录。"""
    try:
        if parents:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)
        return {"status": "success", "path": os.path.abspath(path), "created": True}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("file_exists", "检查文件或目录是否存在", "file_io",
               params=[{"name": "path", "type": "str", "description": "路径"}])
def file_exists(path):
    """检查文件或目录是否存在。"""
    try:
        exists = os.path.exists(path)
        return {"status": "success", "path": os.path.abspath(path),
                "exists": exists,
                "type": "dir" if os.path.isdir(path) else
                        ("file" if os.path.isfile(path) else None)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("get_file_info", "获取文件元信息（大小、修改时间等）", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"}])
def get_file_info(path):
    """获取文件元信息。"""
    try:
        if not os.path.exists(path):
            return {"status": "error", "reason": f"路径不存在: {path}"}
        stat = os.stat(path)
        return {
            "status": "success",
            "path": os.path.abspath(path),
            "type": "dir" if os.path.isdir(path) else "file",
            "size": stat.st_size,
            "mode": oct(stat.st_mode),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "ctime": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "atime": datetime.fromtimestamp(stat.st_atime).isoformat(),
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("compress_file", "压缩文件为 zip", "file_io",
               params=[{"name": "src", "type": "str", "description": "源文件或目录路径"},
                       {"name": "dst", "type": "str", "description": "目标 zip 文件路径"},
                       {"name": "compression", "type": "str", "description": "压缩方式 (deflated/stored)",
                        "required": False, "default": "deflated"}])
def compress_file(src, dst, compression="deflated"):
    """将文件或目录压缩为 zip。"""
    try:
        if not os.path.exists(src):
            return {"status": "error", "reason": f"源路径不存在: {src}"}
        comp = zipfile.ZIP_DEFLATED if compression == "deflated" else zipfile.ZIP_STORED
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        count = 0
        with zipfile.ZipFile(dst, "w", compression=comp) as zf:
            if os.path.isdir(src):
                for root, _dirs, files in os.walk(src):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        arc = os.path.relpath(fp, src)
                        zf.write(fp, arc)
                        count += 1
            else:
                zf.write(src, os.path.basename(src))
                count = 1
        return {"status": "success", "src": os.path.abspath(src),
                "dst": os.path.abspath(dst), "files": count,
                "size": os.path.getsize(dst)}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("decompress_file", "解压 zip 文件", "file_io",
               params=[{"name": "src", "type": "str", "description": "zip 文件路径"},
                       {"name": "dst", "type": "str", "description": "解压目标目录"}])
def decompress_file(src, dst):
    """解压 zip 文件到目标目录。"""
    try:
        if not os.path.isfile(src):
            return {"status": "error", "reason": f"zip 文件不存在: {src}"}
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            members = zf.namelist()
            zf.extractall(dst)
        return {"status": "success", "src": os.path.abspath(src),
                "dst": os.path.abspath(dst), "files": len(members), "members": members}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("search_files", "按 glob 模式搜索文件", "file_io",
               params=[{"name": "pattern", "type": "str", "description": "glob 匹配模式"},
                       {"name": "path", "type": "str", "description": "搜索根目录",
                        "required": False, "default": "."},
                       {"name": "recursive", "type": "bool", "description": "是否递归搜索",
                        "required": False, "default": False}])
def search_files(pattern, path=".", recursive=False):
    """按 glob 模式搜索文件。"""
    try:
        if recursive:
            matches = _glob.glob(os.path.join(path, "**", pattern), recursive=True)
        else:
            matches = _glob.glob(os.path.join(path, pattern))
        results = []
        for m in sorted(matches):
            results.append({
                "path": os.path.abspath(m),
                "type": "dir" if os.path.isdir(m) else "file",
                "size": os.path.getsize(m) if os.path.isfile(m) else None,
            })
        return {"status": "success", "pattern": pattern, "path": os.path.abspath(path),
                "count": len(results), "files": results}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("read_file_lines", "读取文件指定行范围", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"},
                       {"name": "start", "type": "int", "description": "起始行号（从1开始）",
                        "required": False, "default": 1},
                       {"name": "end", "type": "int", "description": "结束行号（包含）",
                        "required": False, "default": -1},
                       {"name": "encoding", "type": "str", "description": "编码",
                        "required": False, "default": "utf-8"}])
def read_file_lines(path, start=1, end=-1, encoding="utf-8"):
    """读取文件指定行范围（1-based，end=-1 表示到末尾）。"""
    try:
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
        total = len(lines)
        s = max(1, start)
        e = total if end == -1 else min(end, total)
        if s > e:
            return {"status": "error", "reason": f"起始行 {s} 大于结束行 {e}"}
        selected = [ln.rstrip("\n").rstrip("\r") for ln in lines[s - 1:e]]
        return {"status": "success", "path": os.path.abspath(path),
                "total_lines": total, "start": s, "end": e,
                "count": len(selected), "lines": selected}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@register_tool("write_binary", "写入二进制文件", "file_io",
               params=[{"name": "path", "type": "str", "description": "文件路径"},
                       {"name": "data", "type": "str", "description": "二进制数据的 hex 或 base64 字符串"},
                       {"name": "encoding", "type": "str", "description": "输入编码方式 (hex/base64)",
                        "required": False, "default": "base64"}])
def write_binary(path, data, encoding="base64"):
    """写入二进制文件（data 为 hex 或 base64 字符串）。"""
    try:
        import base64
        if encoding == "hex":
            raw = bytes.fromhex(data)
        elif encoding == "base64":
            raw = base64.b64decode(data)
        else:
            return {"status": "error", "reason": f"不支持的编码方式: {encoding}（请用 hex 或 base64）"}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
        return {"status": "success", "path": os.path.abspath(path),
                "size": len(raw), "input_encoding": encoding}
    except Exception as e:
        return {"status": "error", "reason": str(e)}
