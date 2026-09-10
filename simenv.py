"""
MuJoCo 环境薄封装：负责模型加载、复位、按物理步长推进、外部力施加与位姿/Jacobian 读取。

输入  - params：一次实验的参数 dict（config.parameter_defaults() 或 JSON 载入）。
        用到 physics_hz（物理步长）、MODEL_XML_PATH 等常量从 config 取。
输出  - 一个 SimEnv 实例，run.py 在物理循环里调用其方法；不包含任何控制逻辑。

关键约束（与代码规范一致）：
- 不能直接改 qpos 当作执行器；控制一律走 data.ctrl（内置关节位置伺服 gainprm=5000）。
- 等效 Y 向扰动 w(t) 通过 xfrc_applied 加在 wrist_3_link body（世界系 Y 力），与 SFC 解耦。
- 复位时把 ctrl 置为 INIT_Q，让位置伺服从静止开始就已托住重力，避免起步冲击。
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

import config


class SimEnv:
    def __init__(self, params: dict[str, Any], xml_path=None):
        self.params = params
        xml_path = str(xml_path) if xml_path is not None else str(config.MODEL_XML_PATH)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 物理步长由参数决定（默认 1000 Hz → 1 ms；模型自带 0.002 会被覆盖）。
        self.model.opt.timestep = 1.0 / float(params["physics_hz"])

        # 用名称解析关键 id；解析失败立刻报错，避免运行时才发现拼写问题。
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, config.TCP_SITE_NAME)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, config.FORCE_BODY_NAME)
        if self.site_id < 0:
            raise ValueError(f"site 不存在：{config.TCP_SITE_NAME}")
        if self.body_id < 0:
            raise ValueError(f"body 不存在：{config.FORCE_BODY_NAME}")

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------ 状态
    def reset(self, qpos: list[float]) -> None:
        """把机器人放到指定关节角并静止；ctrl 同步为同一目标，让伺服预紧。"""
        d = self.data
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.ctrl[:] = qpos
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def step(self) -> None:
        """推进一个物理步（已施加的外力保留到下次被改写/清零）。"""
        mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ 位姿 / Jacobian
    def tcp_pos(self) -> np.ndarray:
        """attachment_site 在世界系的位置 (3,)，单位 m。"""
        return self.data.site_xpos[self.site_id].copy()

    def tcp_rot(self) -> np.ndarray:
        """attachment_site 在世界系的姿态矩阵 (3,3)。"""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def jac_site(self) -> tuple[np.ndarray, np.ndarray]:
        """当前关节角下 site 的平动 (3,nv)/转动 (3,nv) Jacobian，均为世界系。"""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return self._jacp.copy(), self._jacr.copy()

    # ------------------------------------------------------------------ 输入
    def set_ctrl(self, q_cmd: np.ndarray) -> None:
        """关节位置目标（rad）。内置位置伺服会把它变成力矩，见 config 注释。"""
        self.data.ctrl[:] = np.asarray(q_cmd, dtype=float)

    def clear_external_force(self) -> None:
        self.data.xfrc_applied[:] = 0.0

    def apply_force_world(self, force_xyz: np.ndarray) -> None:
        """在 FORCE_BODY 上施加世界系平动力（N）。xfrc_applied 形如 (nbody,6)，前 3 列=世界系力。"""
        row = self.data.xfrc_applied[self.body_id]
        row[0:3] = np.asarray(force_xyz, dtype=float)

    # ------------------------------------------------------------------ 模型信息
    @property
    def nv(self) -> int:
        return self.model.nv

    def joint_range(self) -> np.ndarray:
        return self.model.jnt_range.copy()
