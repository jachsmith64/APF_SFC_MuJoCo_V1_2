"""
等效 Y 向扰动 w(t) 的唯一入口（V1.2）：只允许“预实验导出的冻结文件”。

《DS 修改指南》§1.1 / §5.1：
- 删除合成正弦、随机种子、幅值、缩放等正式分支；正式扰动只能来自文件。
- 文件列：[t(s), 世界Y向力(N)]；t 严格递增、力全部有限。
- 文件覆盖时间必须不短于本次 schedule/duration；不足直接报错，禁止循环/补零/替换。
- 加载即算 SHA-256，供运行日志与 manifest。

接口：
    load_fixed(params) -> Disturbance   （一次加载，做全部校验，返回对象）
    Disturbance.w(t)                      t -> 力(N)
    Disturbance.sha256 / basename / t_end / source_id

source_id：从数据包 wfit_meta.json 的 data_id 读（用于 UI 显示“P05R01”）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import config
import provenance

COVERAGE_TOL_S: float = 1.0e-9      # 覆盖比对容差（s）


class DisturbanceError(ValueError):
    """w(t) 文件不满足正式约束。"""


class Disturbance:
    def __init__(self, path: Path, t: np.ndarray, f: np.ndarray, sha256: str,
                 source_id: str | None = None):
        self.path = Path(path)
        self.t = np.asarray(t, dtype=float)
        self.f = np.asarray(f, dtype=float)
        self.sha256 = sha256
        self.source_id = source_id

    @property
    def basename(self) -> str:
        return self.path.name

    @property
    def short_sha(self) -> str:
        return provenance.short_hash(self.sha256)

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def n_samples(self) -> int:
        return int(self.t.size)

    def w(self, tt: float) -> float:
        """时间 tt(s) 处的 Y 向力(N)；覆盖已由 load 保证，窗外夹紧到端点。"""
        return float(np.interp(tt, self.t, self.f))

    def describe(self) -> str:
        return (f"w(t) 文件 {self.basename}"
                + (f"（源 {self.source_id}，" if self.source_id else "（")
                + f"sha12={self.short_sha}，{self.n_samples} 点，"
                + f"t∈[{self.t[0]:.3f},{self.t[-1]:.3f}]s）")


def _source_id_from_meta(packet_dir: Path) -> str | None:
    meta = packet_dir / "wfit_meta.json"
    if meta.is_file():
        try:
            return str(json.loads(meta.read_text(encoding="utf-8")).get("data_id", "")) or None
        except (OSError, ValueError):
            return None
    return None


def load_fixed(params: dict[str, Any], required_t_end: float | None = None) -> Disturbance:
    """
    加载并严格校验冻结 w(t) 文件。

    required_t_end：本次运行会查询到的最大时刻(s)。若文件末时刻 < 该值-COVERAGE_TOL 则报错。
    路径解析：相对项目根 / 绝对路径均可（config.resolve_path）。
    """
    raw = str(params.get("disturbance_file") or "").strip()
    if not raw:
        raise DisturbanceError("disturbance_file 为空：正式扰动必须指向冻结 w(t) 文件。")
    path = config.resolve_path(raw)
    if not path.is_file():
        raise DisturbanceError(f"扰动文件不存在：{path}")

    data = np.genfromtxt(str(path), delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[1] < 2 or data.shape[0] < 2:
        raise DisturbanceError(f"扰动文件至少要有两列且 ≥2 行数据：{path}")
    t = data[:, 0].astype(float)
    f = data[:, 1].astype(float)

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        raise DisturbanceError(f"扰动文件必须全为有限数：{path}")
    if not np.all(np.diff(t) > 0):
        raise DisturbanceError(f"扰动文件时间列必须严格递增（不允许等步长重排/重复）：{path}")

    sha256 = provenance.sha256_file(path)
    if required_t_end is not None:
        need = float(required_t_end) - COVERAGE_TOL_S
        if float(t[-1]) < need:
            raise DisturbanceError(
                f"扰动文件覆盖不足：文件到 t={t[-1]:.3f}s，但本次 schedule 到 "
                f"{float(required_t_end):.3f}s。禁止循环/补零/替换，请换覆盖更长的冻结 w(t)。")

    # 源数据 ID（随包 meta 提供，仅作展示）
    source_id = _source_id_from_meta(path.parent)
    return Disturbance(path, t, f, sha256, source_id=source_id)


def make_disturbance(params: dict[str, Any]) -> Disturbance:
    """兼容旧调用名：直接 load_fixed（无合成分支）。"""
    return load_fixed(params)


def describe(params: dict[str, Any]) -> str:
    """一行中文说明（run 摘要用）。不重复做覆盖校验，仅在文件存在时打印。"""
    raw = str(params.get("disturbance_file") or "").strip()
    if not raw:
        return "扰动：未指定文件"
    return f"扰动：文件 {Path(raw).name}（仅文件源，SHA 见 manifest）"
