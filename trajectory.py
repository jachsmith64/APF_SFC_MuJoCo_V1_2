"""
名义参考轨迹：机械臂沿世界 X 匀速平移（或按真实 schedule 往返），Y/Z/姿态锁起始值。

约定：起点 = INIT_Q 下 attachment_site 实测位姿 (x0,y0,z0,R0)，run 开始时从 env 读一次。
- 无 replay_schedule：x_ref(t)=x0+sign·vx·t；y_ref/z_ref 恒起点；姿态恒 R0。
- 有 replay_schedule（fit_w 生成的 [t(s),沿程mm]）：x_ref 按沿程回放，启停/换向与 w(t) 对齐。
  schedule 路径按 config.resolve_path 解析（相对项目根/绝对均可）。

V1.2：暴露 t_end() 供 disturbance 做“文件覆盖 ≥ schedule”校验。
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config


class ReferencePath:
    def __init__(self, params: dict[str, Any], pos0: np.ndarray, R0: np.ndarray):
        dv = config.derived(params)
        self.vx = dv["vx_m_s"] * dv["motion_sign"]       # X 速度（含方向），m/s
        self.x0 = float(pos0[0])
        self.y0 = float(pos0[1])                          # = y_ref(t)，恒定
        self.z0 = float(pos0[2])
        self.R0 = R0.copy()
        self.duration = float(params["duration_s"])
        self._sched_t: np.ndarray | None = None
        self._sched_along_mm: np.ndarray | None = None
        raw = str(params.get("replay_schedule") or "").strip()
        if raw:
            sched = config.resolve_path(raw)
            if not sched.is_file():
                raise ValueError(f"schedule 文件不存在：{sched}")
            a = np.loadtxt(str(sched), delimiter=",", comments="#", ndmin=2)
            if a.ndim == 1 or a.shape[1] < 2:
                a = a.reshape(-1, 2)
            st = a[:, 0]
            if not (np.diff(st) > 0).all():
                raise ValueError(f"schedule 时间必须严格递增：{sched}")
            if not np.all(np.isfinite(a[:, 1])):
                raise ValueError(f"schedule 沿程必须全为有限数：{sched}")
            self._sched_t = st
            self._sched_along_mm = a[:, 1]
            if abs(self._sched_along_mm[0]) > 1e-3:
                raise ValueError(
                    f"schedule 起点沿程应≈0，实际 {self._sched_along_mm[0]:.3f} mm：{sched}")

    def _replaying(self) -> bool:
        return self._sched_t is not None

    def t_end(self) -> float:
        """本次运行会查询到的最晚时刻(s)：回放=schedule 末时刻，否则=duration。"""
        return float(self._sched_t[-1]) if self._replaying() else self.duration

    def x_ref(self, t: float) -> float:
        if not self._replaying():
            return self.x0 + self.vx * t
        along = float(np.interp(t, self._sched_t, self._sched_along_mm,
                                left=self._sched_along_mm[0], right=self._sched_along_mm[-1]))
        return self.x0 + along * 1e-3

    def y_ref(self, t: float) -> float:
        return self.y0

    def z_ref(self, t: float) -> float:
        return self.z0

    def pose(self, t: float, y_cmd: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([self.x_ref(t), y_cmd, self.z0], dtype=float)
        return p, self.R0

    def describe(self) -> str:
        if not self._replaying():
            return (f"轨迹：沿 X {'+' if self.vx >= 0 else '-'}"
                    f"{abs(self.vx) * 1000:.1f} mm/s，Y/Z/姿态锁起始值")
        stroke = float(self._sched_along_mm.max())
        t_end = float(self._sched_t[-1])
        return (f"轨迹：回放真实往返 schedule（0→{stroke:.1f}mm→0，{t_end:.3f}s），"
                f"Y/Z/姿态锁起始值")
