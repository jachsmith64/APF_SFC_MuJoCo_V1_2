"""
实验数据记录器（V1.2）：在控制节拍上写一帧文本 CSV，控制台只回显摘要。

采样约定：只记录“控制节拍”上的数据（帧率=control_hz）。新增：
- dt_actual：本次控制节拍相对上次的真实时间间隔(s)，132 Hz 下 7/8 ms 交替如实记录；
- sfc_v_internal / sfc_v_out / sfc_shear_force：SFC 内部日志量（基线组记 0）。

列含义（单位 SI；首行以 "# " 注释，analysis 按列名解析，不依赖固定下标）：
    t, x_ref, y_ref, z_ref, x_cmd, y_cmd, x_act, y_act, z_act,
    dy, F_apf, w_force, dt_actual,
    sfc_v_internal, sfc_v_out, sfc_shear_force
保存后生成 params.json（config.save_parameters）、run_fingerprint.json（溯源）与 run_summary.txt。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

import config

_COLS = [
    "t", "x_ref", "y_ref", "z_ref", "x_cmd", "y_cmd",
    "x_act", "y_act", "z_act", "dy", "F_apf", "w_force",
    "dt_actual", "sfc_v_internal", "sfc_v_out", "sfc_shear_force",
]


class Recorder:
    def __init__(self, out_dir: Path, params: dict[str, Any], mode: str):
        self.out_dir = out_dir
        self.params = dict(params)
        self.mode = mode
        self.csv_path: Path | None = None
        self._file: TextIO | None = None
        self.n_rows = 0
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._row_sums: dict[str, float] = {}     # 供摘要均值统计
        self._last_actual_dt: float | None = None  # 最新 dt_actual

    # ------------------------------------------------------------- 打开/关闭
    def open(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "trajectory.csv"
        self._file = self.csv_path.open("w", encoding="utf-8", newline="\n")
        self._file.write("# " + ",".join(_COLS) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # ------------------------------------------------------------- 写帧
    def record(self, t: float, f: dict[str, float]) -> None:
        row = [str(t)] + [str(f.get(c, "")) for c in _COLS[1:]]
        self._file.write(",".join(row) + "\n")
        self.n_rows += 1
        if self._first_t is None:
            self._first_t = t
        self._last_t = t
        if "dt_actual" in f:
            self._last_actual_dt = float(f["dt_actual"])
        for key in ("x_act", "y_act", "z_act", "y_cmd", "dy", "F_apf", "w_force",
                    "dt_actual"):
            val = f.get(key)
            if isinstance(val, (int, float)):
                prev = self._row_sums.get(key, 0.0)
                self._row_sums[key] = prev + val

    # ------------------------------------------------------------- 摘要/指纹落盘
    def write_summary(self, wall_seconds: float, note: str = "") -> None:
        txt = self.out_dir / "run_summary.txt"
        lines = [
            "# APF-SFC MuJoCo 单次运行摘要",
            f"mode: {self.mode}",
            f"rows(control ticks): {self.n_rows}",
            f"first_t: {self._first_t:.4f}  last_t: {self._last_t:.4f}",
            f"wall_s: {wall_seconds:.2f}",
            "params:",
        ]
        for k, v in sorted(self.params.items()):
            lines.append(f"  {k} = {v}")
        if note:
            lines.append(f"note: {note}")
        txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        config.save_parameters(self.params, self.out_dir / "params.json")

    def write_fingerprint(self, fp: dict[str, Any]) -> None:
        """把 run 溯源指纹写进 run_fingerprint.json（analysis/pair 用它做 A/B 门控）。"""
        (self.out_dir / "run_fingerprint.json").write_text(
            json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
