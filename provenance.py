"""
溯源 / 指纹工具（V1.2 引入）：哈希、参数规范化指纹、模型指纹。

用途（对应《DS 修改指南》§5 / §7.1）：
- A/B 公平门控：比较参数规范化 JSON、w(t)/schedule 哈希、模型哈希等，
  任一不同就拒绝比较，防止把新旧混合/不公平数据算成抑制率。
- fit_w 产物：meta 里记录源 CSV / 三个输出文件 / 模型文件的 SHA-256。
- manifest / run 指纹：写成功状态与所需哈希。

约定：
- 所有 sha256 都返回 64 位小写 hex。
- “规范化指纹”对 dict 做键排序 + ensure_ascii + 紧凑 json，保证跨平台一致。
- 路径指纹一律解析成绝对路径再读字节；不存在的文件 -> 记成缺失串。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import config


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """文件字节 SHA-256；不存在/读失败则返回形如 MISSING:<reason> 的非空串。"""
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError as exc:
        return f"MISSING:{exc.__class__.__name__}"


def json_fingerprint(obj: dict[str, Any]) -> str:
    """参数的规范化指纹：只含 JSON 原生类型、键排序、ASCII 紧凑。"""
    return sha256_bytes(
        json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                   default=str).encode("utf-8"))


def model_fingerprint() -> dict[str, Any]:
    """scene.xml / ur10e.xml 的哈希（模型指纹的权威来源）。"""
    scene = config.MODEL_XML_PATH
    ur10e = config.ASSET_DIR / "ur10e.xml"
    return {
        "scene.xml": {"path": str(scene), "sha256": sha256_file(scene)},
        "ur10e.xml": {"path": str(ur10e), "sha256": sha256_file(ur10e)},
    }


def file_record(path: str | Path) -> dict[str, str]:
    """给一个文件生成 {path, sha256} 小记录（用于 w/schedule/e_des 等）。"""
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p)}


def short_hash(hex64: str, n: int = 12) -> str:
    """取哈希前 n 位作人读短标识。"""
    return hex64[:n]
