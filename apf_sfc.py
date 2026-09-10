"""
APF-SFC 抑振控制器（V1.2 论文一致核心）。

链路（SFC 原论文 +《DS 修改指南》§3.1）：
    e_y   = y_actual − y_ref                     （Y 向跟踪误差，m）
    F_apf = −k_a · e_y                           （APF 只含吸引弹簧项，N）
    m·v̇ + μ|v|^(n-1)·v = F_apf                  （SFC 剪切增稠虚拟动力学）
    v_out = g · v                                （论文输出增益）
    dẏ   = v_out ⇒  dy += g·v·dt_actual
    y_cmd = y_ref + Δy

第一轮正式配置：B0(b_eps)=0、K_v=0（论文 Algorithm 1 整定不输出这两个项）。
扩展项保留实现、默认关闭，启用时必须由上层在 manifest/报告里单独标记。

积分与约束：
- 用实际控制间隔 dt_actual = t_k − t_{k-1} 推进（132 Hz 下是 7/8 ms 交替，不得恒用 1/132）。
- 每一步检查 dt_actual>0、输入/状态/输出有限；异常即抛 ApfSfcNumericalError，
  由 run 层把该 run 标为失败。
- n 正式范围 1<n≤5；n=1 只作内部诊断，不作为 SFC 正式结果（reset 后经 is_formal 检查）。
- 离散约束：显式欧拉对“平衡点 v_ss 处线性化阻尼”的稳定上界 dt_max = 2/(n·μ·v_ss^(n-1)/m)，
  由 sfc_dt_max() 计算；run 启动时用 e_des 峰复算并中止超限（≈0.039 s >> 1/132，仅作保险）。
"""

from __future__ import annotations

import math

import numpy as np


class ApfSfcNumericalError(RuntimeError):
    """SFC 状态/输入非有限或 dt 非法：必须把 run 标为失败。"""


class ApfSfc:
    def __init__(self, k_a: float, m: float, mu: float, n: float, g: float,
                 b_eps: float = 0.0, K_v: float = 0.0):
        # 单位：k_a N/m, m kg, μ 视 n（N·sⁿ/mⁿ）, n 无量纲, g 无量纲,
        # b_eps N·s/m（非论文扩展,默认0）, K_v N/m（非论文扩展,默认0）。
        self.k_a = float(k_a)
        self.m = float(m)
        self.mu = float(mu)
        self.n = float(n)
        self.g = float(g)
        self.b_eps = float(b_eps)
        self.K_v = float(K_v)
        if not (1.0 <= self.n <= 5.0):
            raise ValueError(f"SFC n 必须在 1..5：{self.n}")
        self.reset()

    # ------------------------------------------------------------- 状态
    def reset(self) -> None:
        """每次 run 开始清零全部状态。"""
        self.v = 0.0            # 内部虚拟速度 v（m/s）
        self.dy = 0.0           # Δy 虚拟偏移（m）
        self.F_apf = 0.0        # 上个节拍 APF 力（N）
        self.shear = 0.0        # 剪切增稠力 μ|v|^(n-1)·v（N）
        self.a = 0.0            # 虚拟加速度（m/s²）
        self.v_out = 0.0        # 输出补偿速度 g·v（m/s）
        self._checked_formal = False

    @property
    def is_formal(self) -> bool:
        """正式 SFC 结果要求 1<n≤5 且未启用非论文扩展项。"""
        return 1.0 < self.n <= 5.0 and self.b_eps == 0.0 and self.K_v == 0.0

    # ------------------------------------------------------------- 步进
    def step(self, dt_actual: float, e_y: float) -> float:
        """
        推进一个控制节拍，返回 Δy（y_cmd = y_ref + Δy 中的 Δy）。

        输入：dt_actual 本次真实控制间隔(s, 必须>0)、e_y 实测 Y 误差(m)。
        输出：Δy(m)。内部状态 v/dy/F_apf/shear/a/v_out 同步更新。
        """
        if not (dt_actual is not None and np.isfinite(dt_actual)) or dt_actual <= 0.0:
            raise ApfSfcNumericalError(f"SFC dt_actual 非法：{dt_actual}")
        if not np.isfinite(e_y):
            raise ApfSfcNumericalError(f"SFC e_y 非有限：{e_y}")

        self.F_apf = -self.k_a * float(e_y)
        v = self.v
        self.shear = self.mu * float(np.sign(v)) * abs(v) ** self.n   # μ|v|^(n-1)·v
        # 核心：m·v̇ + μ|v|^(n-1)v = F_apf；非论文扩展项（默认关）并入阻尼/刚度。
        a = (self.F_apf - self.b_eps * v - self.shear - self.K_v * self.dy) / self.m
        self.a = float(a)
        self.v = v + a * dt_actual
        self.v_out = self.g * self.v
        self.dy = self.dy + self.v_out * dt_actual

        for name, val in (("v", self.v), ("dy", self.dy), ("F_apf", self.F_apf),
                          ("shear", self.shear), ("a", self.a), ("v_out", self.v_out)):
            if not np.isfinite(val):
                raise ApfSfcNumericalError(f"SFC 状态 {name} 发散（{val}）于 dt={dt_actual:.6g} e_y={e_y:.4g}。")
        return self.dy

    # ------------------------------------------------------------- 日志
    def logs(self) -> dict[str, float]:
        """SFC 内部日志量（recorder 新增列）。"""
        return {
            "sfc_v_internal": float(self.v),
            "sfc_v_out": float(self.v_out),
            "sfc_shear_force": float(self.shear),
        }

    def params_dict(self) -> dict:
        return {"k_a": self.k_a, "m": self.m, "mu": self.mu, "n": self.n,
                "g": self.g, "b_eps": self.b_eps, "K_v": self.K_v}


def sfc_dt_max(k_a: float, m: float, mu: float, n: float, e_abs_max_m: float) -> float:
    """
    论文离散约束的显式近似：在“平衡内部速度 v_ss”处把剪切阻尼线性化为 a≈n·μ·v_ss^(n-1)/m，
    显式欧拉稳定要求 dt ≤ 2/a。v_ss 由 f_max=k_a·e_max 的稳态给出。

    对 P05R01：e_max=125.38µm → f_max=0.188N，m=1,n=2.828,μ=7.615e4 下 dt_max≈0.039 s，
    同时覆盖 1000Hz(1ms) 与 132Hz(~7.6ms)。此处给出推导一致的保守值，不硬编码。
    """
    if mu <= 0.0:
        return float("inf")        # μ=0 无剪切（内部诊断）；另由有限性守卫保护
    f_max = float(k_a) * float(e_abs_max_m)
    v_ss = (abs(f_max) / float(mu)) ** (1.0 / float(n))
    a_lin = float(n) * float(mu) * abs(v_ss) ** (float(n) - 1.0) / float(m)
    if a_lin <= 0.0:
        return float("inf")
    return 2.0 / a_lin
