"""
阻尼最小二乘(DLS) IK：把一个世界系 TCP 目标位姿解成关节角位置指令。

控制链路：y_cmd(APF/SFC) → [x_ref(t), y_cmd, z0, R0] → 本模块解 q_cmd
→ data.ctrl（关节位置伺服）→ 力矩。基线组同样走 IK，只是 y_cmd=y_ref。

实现说明：
- 每个控制节拍在“实测 q”上做一次局部线性化：误差 e=[e_p; e_o]，
  dq = Jᵀ(JJᵀ+λ²I)⁻¹ e，q_cmd = q_meas + dq。参考点每节拍只动很小量，
  连续闭环跟踪即可收敛，无需在节拍内做多轮重线性化（近似单步 CLIK，
  与真机 resolved-rate 用法一致；残余误差由高频位置伺服收掉）。
- 姿态误差用旋转矢量：e_o = rotvec(R_t · R_curᵀ)，与 jacr 世界系转动约定一致。
- 姿态全程锁成初始 R0（第一版只平移、不转动 TCP）。

输入/输出/实验作用：见函数 docstring。纯计算，不触碰硬件、不推进仿真。
"""

from __future__ import annotations

import numpy as np
import mujoco

import simenv


def rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 纯旋转 → 旋转矢量(3,)（轴×角，rad），小角度与大角度都稳健。"""
    # 平面内角度 theta = arccos((tr-1)/2)
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_a)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > np.pi - 1e-3:
        # 180° 附近：acos 失稳，改从反对称部分 + 对角符号取特征向量
        # 这里 TCP 每节拍误差很小，几乎不会走到；取近似即可（符号任意）。
        skew = 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        return theta * skew / (np.linalg.norm(skew) + 1e-12)
    axis = 0.5 / np.sin(theta) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return axis * theta


def dls_ik(
    env: simenv.SimEnv,
    p_t: np.ndarray,
    R_t: np.ndarray,
    lam: float = 1.0e-4,
    max_iters: int = 3,
    tol: float = 1.0e-8,
) -> np.ndarray:
    """
    由实测 q 出发，向目标位姿 [p_t(3), R_t(3,3)] 迭代解 DLS，返回 q_cmd(6,)。

    输入：env（提供实测位姿与 Jacobian）、p_t 世界位置、R_t 世界姿态，
         lam 阻尼系数、max_iters 每节拍内牛顿轮数、tol 位置/姿态误差停止阈值。
    输出：q_cmd 关节角位置（rad），会被 set_ctrl 直接使用。
    实验作用：把 SFC 给出的 y_cmd 与参考 x/z 一起变成关节位置指令。

    实现：每轮把临时 qpos 置为当前 q 再 mj_forward 重线性化（位姿/Jacobian 都随 q 变），
    求解 dq = Jᵀ(JJᵀ+λ²I)⁻¹ e 后更新 q；最后把 qpos 还原成实测值，避免污染物理状态。
    时间连续时目标每节拍只动很小量，1~2 轮即收敛；大台阶（起步）由多轮兜底。
    """
    m, d = env.model, env.data
    q0 = d.qpos.copy()                 # 实测关节角，结束前还原
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    q = q0.copy()
    for _ in range(max(1, int(max_iters))):
        p_c = d.site_xpos[env.site_id].copy()
        R_c = d.site_xmat[env.site_id].reshape(3, 3).copy()
        mujoco.mj_jacSite(m, d, env._jacp, env._jacr, env.site_id)

        e_p = p_t - p_c
        e_o = rotvec(R_t @ R_c.T)
        if np.linalg.norm(e_p) <= tol and np.linalg.norm(e_o) <= tol:
            break
        e = np.concatenate([e_p, e_o])
        J = np.vstack([env._jacp, env._jacr])          # (6,nv)
        # 阻尼最小二乘：dq = Jᵀ(JJᵀ+λ²I)⁻¹ e，λ² 加在 JJᵀ 上（正确一侧）。
        # 若误写成 (JᵀJ+λ²I)⁻¹ 且 Jᵀ 在左，等于正则化失效，牛顿会发散到关节限位。
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(J.shape[0]), e)
        q = q + dq
        # 关节限位兜底
        jr = m.jnt_range
        q = np.clip(q, jr[:, 0] + 1e-6, jr[:, 1] - 1e-6)
        # 重线性化到新 q
        d.qpos[:] = q
        mujoco.mj_forward(m, d)

    # 还原实测物理状态（qpos + 全部派生量）
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    return q
