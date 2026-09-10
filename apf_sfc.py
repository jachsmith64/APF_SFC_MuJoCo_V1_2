"""
APF-SFC 抑振控制器（V1.3：视觉加速度—虚拟力映射 + SFC 论文核心）。

链路（V1.3，取代 V1.2 的 F_apf = -k_a·e_y）：
    e_y 历史
      → a_y_est          （CausalAccelForceMapper：尾部窗口因果二次最小二乘）
      → F_vir = +force_map_mass_kg · a_y_est          （N；正号，见下）
      → m·v̇_s + μ|v_s|^(n-1)·v_s = F_vir             （SFC 剪切增稠虚拟动力学）
      → v_s = v_s + a_s·dt_actual
      → v_sfc_out = g·v_s                             （论文输出增益）
      → 运行层积分 y_sfc_offset += v_sfc_out·dt_actual （在 run.py 完成，非 SFC 核心）
      → y_cmd = y_ref + y_sfc_offset

符号约定（必须遵守）：
- F_vir = +force_map_mass_kg · a_y_est 为**正号**：正向加速度 → 正向虚拟力，
  负向加速度 → 负向虚拟力。禁止写成 -force_map_mass_kg·a_y_est。
- SFC 输出端**不加**人为固定负号；剪切增稠阻力已由方程内部的 -μ|v_s|^(n-1)·v_s 体现。

参数范围（V1.3）：正式 SFC 核心只保留 m、μ、n、g。
旧版 B0(b_eps)/K_v 不属于 Chen 标准 SFC 参数，已从正式控制计算中移除。

积分与约束：
- 用实际控制间隔 dt_actual = t_k − t_{k-1} 推进（132 Hz 下是 7/8 ms 交替，不得恒用 1/132）。
- 每步检查 dt_actual>0、输入/状态/输出有限；异常即抛 ApfSfcNumericalError，
  由 run 层把该 run 标为失败。
- n 正式范围 1<n≤5。
- 离散约束：显式欧拉对“平衡点 v_ss 处线性化阻尼”的稳定上界 dt_max = 2/(n·μ·v_ss^(n-1)/m)，
  由 sfc_dt_max() 计算，输入为**最大虚拟力** f_abs_max_N（不再由 k_a·e_max 推）。
"""

from __future__ import annotations

import numpy as np


class ApfSfcNumericalError(RuntimeError):
    """SFC 状态/输入非有限或 dt 非法：必须把 run 标为失败。"""


class CausalAccelForceMapper:
    """
    因果加速度—虚拟力映射器。

    输入：真实时间戳 t(s) 与该时刻的 e_y(m)（按控制节拍依次喂入）。
    输出：a_y_est(m/s²)、F_vir(N)、mapper_ready(bool)。

    方法：保存最近 accel_window_points 个 (t, e_y)，用尾部窗口对
        e_y(τ) = c0 + c1·τ + c2·τ² ,  τ_i = t_i − t_current
    做二次最小二乘拟合，则 a_y_est = 2·c2。
    τ 以当前时刻为零点，减少时间数值误差；必须使用真实时间戳（132 Hz 下 7/8 ms 交替）。
    样本不足（窗口未满）时 ready=False、a_y_est=0、F_vir=0，
    绝不用不完整差分在启动瞬间产生巨大虚拟力。
    """

    def __init__(self, force_map_mass_kg: float, accel_window_points: int = 5):
        self.force_map_mass_kg = float(force_map_mass_kg)
        n_pts = int(accel_window_points)
        if n_pts < 3 or n_pts % 2 == 0:
            raise ValueError(f"accel_window_points 必须为不小于 3 的奇数：{accel_window_points}")
        self.accel_window_points = n_pts
        self.reset()

    def reset(self) -> None:
        self._t: list[float] = []
        self._e: list[float] = []
        self.a_y_est = 0.0
        self.F_vir = 0.0
        self.ready = False

    def step(self, t: float, e_y: float) -> tuple[float, float, bool]:
        """喂入一个控制节拍样本，返回 (a_y_est, F_vir, mapper_ready)。"""
        if not (t is not None and np.isfinite(t)):
            raise ApfSfcNumericalError(f"映射器 t 非法：{t}")
        if not np.isfinite(e_y):
            raise ApfSfcNumericalError(f"映射器 e_y 非有限：{e_y}")
        tt = float(t)
        if self._t and tt <= self._t[-1]:
            raise ApfSfcNumericalError(
                f"映射器要求时间戳严格递增：{tt} <= {self._t[-1]}")

        self._t.append(tt)
        self._e.append(float(e_y))
        w = self.accel_window_points
        while len(self._t) > w:
            self._t.pop(0)
            self._e.pop(0)

        if len(self._t) < w:
            self.ready = False
            self.a_y_est = 0.0
            self.F_vir = 0.0
            return self.a_y_est, self.F_vir, self.ready

        t_arr = np.asarray(self._t, dtype=float)
        e_arr = np.asarray(self._e, dtype=float)
        tau = t_arr - t_arr[-1]                      # 以当前时刻为零点
        design = np.column_stack([np.ones_like(tau), tau, tau * tau])
        coeff, *_ = np.linalg.lstsq(design, e_arr, rcond=None)
        a_est = 2.0 * float(coeff[2])
        if not np.isfinite(a_est):
            raise ApfSfcNumericalError(f"映射器 a_y_est 非有限：{a_est}")
        f_vir = self.force_map_mass_kg * a_est       # 正号：F_vir = +m_map · a_y_est
        if not np.isfinite(f_vir):
            raise ApfSfcNumericalError(f"映射器 F_vir 非有限：{f_vir}")

        self.a_y_est = a_est
        self.F_vir = f_vir
        self.ready = True
        return self.a_y_est, self.F_vir, self.ready


class ApfSfc:
    """SFC 论文核心：m·v̇_s + μ|v_s|^(n-1)·v_s = F_vir；v_sfc_out = g·v_s。"""

    def __init__(self, m: float, mu: float, n: float, g: float):
        # 单位：m kg, μ 视 n（N·sⁿ/mⁿ）, n 无量纲, g 无量纲。
        self.m = float(m)
        self.mu = float(mu)
        self.n = float(n)
        self.g = float(g)
        if not (1.0 <= self.n <= 5.0):
            raise ValueError(f"SFC n 必须在 1..5：{self.n}")
        self.reset()

    # ------------------------------------------------------------- 状态
    def reset(self) -> None:
        """每次 run 开始清零全部状态。"""
        self.v = 0.0            # 内部虚拟速度 v_s（m/s）
        self.shear = 0.0        # 剪切增稠力 μ·sign(v_s)·|v_s|^n（N）
        self.a = 0.0            # 内部虚拟加速度 a_s（m/s²）
        self.v_out = 0.0        # 输出速度 g·v_s（m/s），即 v_sfc_out
        self.F_vir = 0.0        # 本步输入虚拟力（N）
        self.dt_last = 0.0      # 本步 dt_actual（s）

    @property
    def is_formal(self) -> bool:
        """正式 SFC 结果要求 1<n≤5（V1.3 无 B0/K_v 扩展项）。"""
        return 1.0 < self.n <= 5.0

    # ------------------------------------------------------------- 步进
    def step(self, dt_actual: float, F_vir: float) -> float:
        """
        推进一个控制节拍，返回 v_sfc_out = g·v_s（m/s）。

        输入：dt_actual 本次真实控制间隔(s, 必须>0)、F_vir 虚拟力(N, 由映射器给出)。
        输出：v_sfc_out(m/s)。累计位移 y_sfc_offset 由 run.py 在运行层积分，本类不返回 dy。
        """
        if not (dt_actual is not None and np.isfinite(dt_actual)) or dt_actual <= 0.0:
            raise ApfSfcNumericalError(f"SFC dt_actual 非法：{dt_actual}")
        if not np.isfinite(F_vir):
            raise ApfSfcNumericalError(f"SFC F_vir 非有限：{F_vir}")

        self.F_vir = float(F_vir)
        self.dt_last = float(dt_actual)
        v = self.v
        self.shear = self.mu * float(np.sign(v)) * abs(v) ** self.n      # μ|v|^(n-1)·v
        a = (self.F_vir - self.shear) / self.m                           # m·v̇ + shear = F_vir
        self.a = float(a)
        self.v = v + a * dt_actual
        self.v_out = self.g * self.v

        for name, val in (("v_s", self.v), ("shear", self.shear), ("a_s", self.a),
                          ("v_out", self.v_out), ("F_vir", self.F_vir)):
            if not np.isfinite(val):
                raise ApfSfcNumericalError(
                    f"SFC 状态 {name} 发散（{val}）于 dt={dt_actual:.6g} F_vir={F_vir:.4g}。")
        return self.v_out

    # ------------------------------------------------------------- 日志
    def logs(self) -> dict[str, float]:
        """SFC 内部日志量（recorder 列）。"""
        return {
            "sfc_a_internal": float(self.a),
            "sfc_v_internal": float(self.v),
            "sfc_v_out": float(self.v_out),
            "sfc_shear_force": float(self.shear),
            "F_vir": float(self.F_vir),
        }

    def params_dict(self) -> dict:
        return {"m": self.m, "mu": self.mu, "n": self.n, "g": self.g}


def sfc_dt_max(m: float, mu: float, n: float, f_abs_max_N: float) -> float:
    """
    论文离散约束的显式近似：在“平衡内部速度 v_ss”处把剪切阻尼线性化为
    a≈n·μ·v_ss^(n-1)/m，显式欧拉稳定要求 dt ≤ 2/a。

    V1.3：输入改为**最大虚拟力** f_abs_max_N（N），不再由 k_a·e_max 计算。
    v_ss 由 f_max 的稳态给出：v_ss = (f_max/μ)^(1/n)。
    """
    if mu <= 0.0:
        return float("inf")        # μ=0 无剪切（内部诊断）；另由有限性守卫保护
    f_max = abs(float(f_abs_max_N))
    if f_max <= 0.0:
        return float("inf")
    v_ss = (f_max / float(mu)) ** (1.0 / float(n))
    a_lin = float(n) * float(mu) * abs(v_ss) ** (float(n) - 1.0) / float(m)
    if a_lin <= 0.0:
        return float("inf")
    return 2.0 / a_lin
