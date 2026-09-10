"""
单次无头/观赛仿真入口（V1.3）：跑一遍 baseline(A) 或 apf_sfc(B)。

V1.3 控制链（每个控制节拍，A/B 同序执行）：
    y_act → e_y = y_act − y_ref
      → CausalAccelForceMapper.step(t, e_y) → (a_y_est, F_vir, mapper_ready)
          · A、B 两组都调用，以便记录**同一口径**的诊断列；
          · A 组只记录、**绝不**把估计量送入机器人控制（v_sfc_out=0、y_sfc_offset=0）。
      → B 组：v_sfc_out = ctrl.step(dt_actual, F_vir)        # SFC 核心 m·v̇_s+μ|v_s|^(n-1)·v_s=F_vir
              y_sfc_offset += v_sfc_out·dt_actual            # 运行层积分（非 SFC 核心）
              y_cmd = y_ref + y_sfc_offset
      → 既有绝对位姿 DLS-IK → MuJoCo 关节位置伺服（模型 XML 内置 kp/kv）。

    运行层积分说明：y_sfc_offset 的积分是“让 SFC 输出速度接到现有 MuJoCo 关节位置伺服”
    的**接口适配**，不属于 SFC 核心方程；SFC 核心只输出 v_sfc_out=g·v_s（m/s）。
    本工程不使用直接力矩控制，也不修改 actuator XML。

    w(t) 是施加到 MuJoCo body 上的**真实扰动**（env.apply_force_world([0, w, 0])）；
    F_vir 是控制器由视觉等效测量构造出的**虚拟力输入**，两者严格分离：
    控制器绝不读取 w_force 作为 SFC 输入，run 也绝不把 w.w(t) 传给 ctrl.step()。

用法（项目根目录、带 mujoco 的 venv）：
    python run.py --params <replay_params.json> --mode baseline
    python run.py --params <replay_params.json> --mode apf_sfc
    python run.py --params <replay_params.json> --mode apf_sfc --show
        # 严格观赛：MuJoCo 3D + 实时 e_y 图 都必须就绪，任一失败在 t=0 前报错退出（零落盘）
    python run.py --params <replay_params.json> --mode apf_sfc --show-best-effort
        # 尽力观赛：两窗任一失败则降级无头继续

观赛链（liveview.py）与“显示不改变主仿真”：
- LiveView 用一份从同一 XML 克隆的 model/data 喂 passive viewer；主仿真的 model/data 从不交给
  GUI → 拖动/暂停/选项只影响克隆，主仿真数值序列不受显示影响。
- pacing 的墙钟零点在窗口就绪后记录，无开局快进；pacing 只影响墙钟，不影响物理步数与落盘。
- 本次不把“trajectory.csv 与无 --show 逐字节一致”当作既成声明；改为在 run_fingerprint.json 写
  execution_integrity 计数 + visualization 诊断，并可用两次运行做 T-VIS-02 数值列复核。

A/B 公平性约定（《DS 修改指南》§7.1，不变）：
- A、B 共用同一个不可变 params 快照、同一 INIT_Q、同一冻结 w(t)（文件逐字节一致），唯一差别是 mode。
- 每次 run 在 out_dir 写 trajectory.csv / params.json / run_summary.txt / run_fingerprint.json；
  指纹新增 execution_integrity（恒在）与 visualization（仅观赛时非空），二者只读诊断、
  不进 A/B 门控比较。

安全（R-018/R-019，不变）：控制节拍用真实 dt_actual；SFC/扰动/位姿非有限即中止；IK 饱和记失败；
dt_max 启动复算，超限中止。
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
import disturbance
import provenance
from apf_sfc import ApfSfc, CausalAccelForceMapper, sfc_dt_max
from kinematics import dls_ik
from liveview import LiveView, LiveViewError
from recorder import Recorder
from simenv import SimEnv
from trajectory import ReferencePath


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _effective_params(params: dict[str, Any]) -> dict[str, Any]:
    """把路径字段归一化成项目相对串，便于指纹/A-B 跨机一致。"""
    out = dict(params)
    if out.get("disturbance_file"):
        out["disturbance_file"] = config.project_relative_str(out["disturbance_file"])
    if out.get("replay_schedule"):
        out["replay_schedule"] = config.project_relative_str(out["replay_schedule"])
    return out


def _model_saturation(q_cmd: np.ndarray, jr: np.ndarray, margin: float = 1e-4) -> bool:
    """关节目标是否已贴到限位（DLS 输出饱和），返回 True 视为 IK 饱和。"""
    if q_cmd is None:
        return True
    lo = q_cmd - (jr[:, 0] + margin)
    hi = (jr[:, 1] - margin) - q_cmd
    return bool(np.any(lo < 0.0) or np.any(hi < 0.0))


def _packet_f_max(dir_of_w: Path) -> float | None:
    """
    读 w_force 同目录 sfc_tuning.json 的 f_max_N（V1.3 最大虚拟力，N），作为 dt_max 复算输入。

    V1.3 起 dt_max 由**最大虚拟力**复算（不再用 k_a·e_des）。
    文件不存在 → None；存在但为 V1.2 旧格式/缺字段 → sfc_tune.load_tuning 抛 ValueError，
    由调用方转成 run 失败（旧整定文件在新架构下不静默兼容）。
    """
    from sfc_tune import load_tuning
    p = dir_of_w / "sfc_tuning.json"
    if not p.is_file():
        return None
    return float(load_tuning(p)["f_max_N"])


def _resolve_y_limit(dir_of_w: Path) -> tuple[float, str]:
    """观赛纵轴半宽：优先冻结包 e_des_um.csv（ceil(1.2×max|e_des| µm)），缺则默认并注明来源。"""
    from liveview import DEFAULT_Y_LIMIT_UM, derive_y_limit_um
    v, s = derive_y_limit_um(dir_of_w)
    if v is None:
        return DEFAULT_Y_LIMIT_UM, f"default(missing e_des_um.csv):{s}"
    return float(v), s


def _read_trajectory_t(csv_path: Path | None) -> np.ndarray:
    """回读 trajectory.csv 的 t 列（首行 # 注释被跳过）。文件异常返回空数组。"""
    if csv_path is None or not Path(csv_path).is_file():
        return np.array([], dtype=float)
    try:
        arr = np.loadtxt(str(csv_path), delimiter=",", comments="#")
        if arr.ndim == 0:
            return np.array([], dtype=float)
        if arr.ndim == 1:               # 单列（理论不会）也按 t 处理
            return np.asarray(arr, dtype=float)
        return np.asarray(arr[:, 0], dtype=float)
    except (OSError, ValueError):
        return np.array([], dtype=float)


def check_execution_integrity(
    *,
    duration_s: float,
    physics_hz: int,
    control_hz: float,
    physics_steps: int,
    control_ticks: int,
    record_rows: int,
    show: bool,
    live_samples_pushed: int | None,
    max_control_dt_s: float | None,
    t_series,
) -> dict[str, Any]:
    """run 结束自动一致性检查（纯函数，tests/test_viz.py 单测）。

    判据（任务书 §4.1）：physics_steps==ceil((duration-1e-9)·hz)；control_ticks==record_rows；
    show 时 live_samples_pushed==control_ticks；trajectory.t 严格递增；最大控制时间间隔
    ≤ 1/control_hz + 1/physics_hz + 1e-9。
    physics_steps_ok 的语义是“跑满全程才应为真”（提前停止/中止时如实为 False）。
    """
    expected = math.ceil((float(duration_s) - 1e-9) * float(physics_hz))
    arr = np.asarray(t_series, dtype=float).reshape(-1)
    if arr.size >= 2:
        t_incr = bool(np.all(np.diff(arr) > 0.0))
    else:
        t_incr = bool(arr.size == 0 or (arr.size == 1 and float(arr[0]) == 0.0))
    dt_allow = 1.0 / float(control_hz) + 1.0 / float(physics_hz) + 1e-9
    if max_control_dt_s is None:
        dt_ok = None
    else:
        dt_ok = bool(max_control_dt_s <= dt_allow)
    pushed = int(live_samples_pushed) if live_samples_pushed is not None else None
    return {
        "expected_physics_steps": int(expected),
        "physics_steps": int(physics_steps),
        "physics_steps_ok": bool(physics_steps == expected),
        "control_ticks": int(control_ticks),
        "record_rows": int(record_rows),
        "control_ticks_record_rows_ok": bool(control_ticks == record_rows),
        "live_samples_pushed": pushed,
        "live_samples_pushed_ok": (bool(pushed == control_ticks) if show else None),
        "trajectory_t_strictly_increasing": t_incr,
        "max_control_dt_actual_s": (float(max_control_dt_s)
                                    if max_control_dt_s is not None else None),
        "max_control_dt_ok": dt_ok,
    }


def run_single(
    params: dict[str, Any],
    mode: str,
    out_dir: Path,
    stop_path: Path | None = None,
    verbose: bool = True,
    show: bool = False,
    pace_rate: float = 1.0,
    show_best_effort: bool = False,
    window_s: float = 10.0,
    y_limit_um: float | None = None,
    y_limit_source: str | None = None,
    show_label: str | None = None,
) -> dict[str, Any]:
    """
    跑完一次实验并落盘。返回结果 dict（含指纹信息与 completed）。

    任何异常/停止：写 run_fingerprint.json completed=false 后抛出/返回。
    严格观赛（show 且非 best-effort）：LiveView.begin() 失败在 t=0 前抛 LiveViewError，
    此时 Recorder 尚未创建 → out_dir 零落盘，调用方可干净中止。
    """
    if mode not in config.VALID_RUN_MODES:
        raise ValueError(f"未知 mode：{mode}，可选 {config.VALID_RUN_MODES}")
    _utf8_stdio()          # 可能被 import 后以非 UTF-8 stdout 调（pair.py/测试），先保打印不炸
    out_dir = Path(out_dir)
    wall_start = time.perf_counter()
    failed: list[str] = []

    env = SimEnv(params)
    env.reset(config.INIT_Q)

    pos0 = env.tcp_pos()
    R0 = env.tcp_rot()
    ref = ReferencePath(params, pos0, R0)

    # ---- 扰动：文件唯一来源，加载即校验 + 覆盖不短于 schedule ----
    try:
        w = disturbance.load_fixed(params, required_t_end=ref.t_end())
    except disturbance.DisturbanceError as exc:
        raise RuntimeError(f"[run] w(t) 校验失败：{exc}") from exc
    schedule_path = config.resolve_path(params.get("replay_schedule") or "")
    schedule_sha = provenance.sha256_file(schedule_path) if schedule_path.is_file() else None

    # ---- 视觉加速度→虚拟力映射器（A/B 共用；A 组只记录诊断，不进控制）----
    mapper = CausalAccelForceMapper(
        force_map_mass_kg=float(params["force_map_mass_kg"]),
        accel_window_points=int(params["accel_window_points"]))

    # ---- SFC 控制器（B 组） / dt_max 启动复算（输入改为最大虚拟力 f_max_N）----
    ctrl = None
    sfc_par = {"m": float(params["sfc_m"]), "mu": float(params["sfc_mu"]),
               "n": float(params["sfc_n"]), "g": float(params["sfc_g"])}
    dt_max_s: float | None = None
    f_vir_max_n: float | None = None
    # 整定文件在 A/B 都读（A/B 指纹里的映射/虚拟力幅值同口径；旧格式在两组都显式报错），
    # 但只有 B 组要求它存在（A 不用 SFC，缺文件不阻塞基线）。
    try:
        f_vir_max_n = _packet_f_max(Path(w.path).parent)
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"[run] SFC 整定文件不可用：{exc}") from exc
    if mode == config.RUN_MODE_APF_SFC:
        ctrl = ApfSfc(**sfc_par)
        if f_vir_max_n is None:
            raise RuntimeError(
                f"[run] 缺少 V1.3 整定文件 sfc_tuning.json（{Path(w.path).parent}），"
                f"无法复算 dt_max；请先运行 sfc_tune.py 生成后再跑 B 组。")
        dt_max_s = sfc_dt_max(sfc_par["m"], sfc_par["mu"], sfc_par["n"], f_vir_max_n)

    force_mapping_note = {
        "method": "a_y_est_to_F_vir_positive",
        "force_map_mass_kg": float(params["force_map_mass_kg"]),
        "accel_window_points": int(params["accel_window_points"]),
        "f_vir_expected_max_N": f_vir_max_n,
        "note": ("F_vir=+force_map_mass_kg·a_y_est（正号）；A 组只记录、不送入控制；"
                 "SFC 输出端无固定负号，剪切增稠阻力在方程内部。"),
    }

    # ---- 可选实时观赛（strict 默认）。begin 失败在 t=0 前抛，零落盘 ----
    live: LiveView | None = None
    if show:
        if y_limit_um is None:
            y_limit_um, y_limit_source = _resolve_y_limit(Path(w.path).parent)
        live = LiveView(
            xml_path=str(config.MODEL_XML_PATH),
            init_q=np.asarray(config.INIT_Q, dtype=float),
            tcp0=pos0,
            title=f"[run] {show_label or mode} · {out_dir.name}",
            window_s=float(window_s),
            y_limit_um=float(y_limit_um),
            y_limit_source=str(y_limit_source or "default"),
            pace_rate=float(pace_rate),
            strict=not show_best_effort,
            overlay_label=show_label or mode,
        )
        live.begin()      # 失败即抛（strict）；best-effort 则内部已降级
        if verbose:
            print(f"[run] 可视化：{live.summary()}  纵轴 ±{live.y_limit_um:g}µm（{live.y_limit_source}）",
                  flush=True)

    # ---- 记录器：只在窗口就绪后创建/打开（strict 失败不留 out_dir）----
    rec: Recorder | None = None
    try:
        rec = Recorder(out_dir, params, mode)
        rec.open()
    except BaseException:
        if live is not None:
            live.close()
        raise

    physics_hz = int(params["physics_hz"])
    control_hz = float(params["control_hz"])
    physics_dt = 1.0 / float(physics_hz)
    control_dt = 1.0 / float(control_hz)
    duration = float(params["duration_s"])
    lam = float(params["ik_lambda"])
    ik_iters = int(params["ik_max_iters"])
    ik_tol = float(params["ik_tol_m"])

    if verbose:
        print(f"[run] mode={mode}  ctrl={'ApfSfc(paper)' if ctrl else 'off'}", flush=True)
        print(f"[run] {ref.describe()}", flush=True)
        print(f"[run] {w.describe()}", flush=True)
        print(f"[run] physics={physics_hz}Hz control={control_hz}Hz "
              f"dur={duration:.3f}s out={out_dir}", flush=True)
        print(f"[run] 映射 F_vir=+{force_mapping_note['force_map_mass_kg']:g}·a_y_est "
              f"(窗口 {force_mapping_note['accel_window_points']} 点，A/B 均记录；A 不入控制)",
              flush=True)
        if ctrl:
            print(f"[run] SFC m={sfc_par['m']} mu={sfc_par['mu']:.4g} n={sfc_par['n']:.4f} "
                  f"g={sfc_par['g']:.6g} formal={ctrl.is_formal} "
                  f"f_vir_max≈{f_vir_max_n:.6g}N dt_max≈{dt_max_s}", flush=True)

    next_ctrl = 0.0
    t = 0.0
    prev_ctrl_t: float | None = None
    dt_actual = control_dt
    step_count = 0
    control_ticks = 0
    live_pushed = 0
    max_dt_actual: float | None = None
    stopped_early = False
    ik_saturated = False
    completed = False
    reason = ""
    last_y_act = pos0[1]
    last_y_cmd = pos0[1]
    y_sfc_offset = 0.0        # 运行层累计位置偏移（B 组非零；A 组恒 0）
    last_F_vir = 0.0

    log_every = max(1.0, duration / 10.0)
    next_log = log_every

    try:
        while t < duration - 1e-9:
            if t + 1e-12 >= next_ctrl:
                # ---- 控制节拍：测量 → 加速度估计 → (B)SFC → IK → ctrl ----
                site = env.data.site_xpos[env.site_id]
                y_act = float(site[1])
                y_ref = ref.y_ref(t)
                e_y = y_act - y_ref
                if not np.isfinite(e_y):
                    raise RuntimeError(f"[run] e_y 非有限（t={t:.4f}）")
                if live is not None:
                    live.push(t, e_y * 1e6)
                    live_pushed += 1
                dt_actual = (t - prev_ctrl_t) if prev_ctrl_t is not None else control_dt
                if max_dt_actual is None or dt_actual > max_dt_actual:
                    max_dt_actual = dt_actual

                # 因果二次拟合估计 a_y_est → F_vir=+force_map_mass_kg·a_y_est。
                # A、B 都调用，保证两组记录到同口径诊断列；A 组只记录、不进控制。
                a_y_est, F_vir, mapper_ready = mapper.step(t, e_y)
                last_F_vir = F_vir

                v_sfc_out = 0.0
                if ctrl is not None:
                    if dt_max_s is not None and dt_actual > dt_max_s:
                        raise RuntimeError(
                            f"[run] dt_actual={dt_actual:.6f}s 超过论文离散上界 dt_max≈{dt_max_s:.4f}s，中止。")
                    # SFC 核心：m·v̇_s+μ|v_s|^(n-1)·v_s=F_vir → v_sfc_out=g·v_s（m/s）
                    v_sfc_out = ctrl.step(dt_actual, F_vir)
                    # 运行层积分（MuJoCo 关节位置伺服的接口适配，非 SFC 核心）：
                    y_sfc_offset += v_sfc_out * dt_actual
                prev_ctrl_t = t
                last_y_cmd = y_ref + y_sfc_offset

                p_t, R_t = ref.pose(t, last_y_cmd)
                q_cmd = dls_ik(env, p_t, R_t, lam=lam, max_iters=ik_iters, tol=ik_tol)
                if not np.all(np.isfinite(q_cmd)):
                    raise RuntimeError(f"[run] IK 输出非有限（t={t:.4f}）")
                if _model_saturation(q_cmd, env.model.jnt_range):
                    ik_saturated = True
                env.set_ctrl(q_cmd)

                # 记录一帧
                f = {
                    "x_ref": ref.x_ref(t), "y_ref": y_ref, "z_ref": ref.z_ref(t),
                    "x_cmd": ref.x_ref(t), "y_cmd": last_y_cmd,
                    "x_act": float(site[0]), "y_act": y_act, "z_act": float(site[2]),
                    "e_y": e_y,
                    # mapper_ready 记为数值 1/0（analysis 用 np.loadtxt 整表读列）
                    "a_y_est": a_y_est, "mapper_ready": (1.0 if mapper_ready else 0.0),
                    "F_vir": F_vir,
                    "w_force": float(w.w(t)),
                    "y_sfc_offset": y_sfc_offset,
                    "dt_actual": dt_actual,
                }
                if ctrl is not None:
                    f.update(ctrl.logs())
                else:
                    f.update({"sfc_a_internal": 0.0, "sfc_v_internal": 0.0,
                              "sfc_v_out": 0.0, "sfc_shear_force": 0.0})
                rec.record(t, f)
                control_ticks += 1
                next_ctrl += control_dt
                last_y_act = y_act

            # ---- 物理步：施加当前扰动力再积分 ----
            wf = w.w(t)
            if not np.isfinite(wf):
                raise RuntimeError(f"[run] 扰动力非有限（t={t:.4f}）")
            env.apply_force_world(np.array([0.0, wf, 0.0]))
            env.step()
            t += physics_dt
            step_count += 1
            if live is not None:
                live.after_step(t, env.data.qpos, env.data.qvel)

            if verbose and t >= next_log:
                print(f"[run] t={t:7.2f}s  y_act={last_y_act * 1e3:10.4f} mm  "
                      f"y_cmd={last_y_cmd * 1e3:10.4f} mm  "
                      f"y_sfc_offset={y_sfc_offset * 1e6:8.2f} µm  "
                      f"F_vir={last_F_vir:9.4g} N", flush=True)
                next_log += log_every

            if step_count % 500 == 0 and stop_path is not None and stop_path.exists():
                print(f"[run] 检测到停止信号，提前结束于 t={t:.2f}s", flush=True)
                stopped_early = True
                reason = "用户停止"
                break

        if ik_saturated and not stopped_early:
            reason = "IK 关节目标饱和"
        completed = (not stopped_early) and (not ik_saturated) and not failed

    except Exception as exc:      # 数值发散/校验失败等 → run 失败
        failed.append(str(exc))
        raise

    finally:
        if live is not None:
            live.close()
        viz_diag = live.diagnostics() if live is not None else None
        if rec is not None:
            rec.close()
        t_series = _read_trajectory_t(rec.csv_path if rec is not None else None)
        integrity = check_execution_integrity(
            duration_s=duration, physics_hz=physics_hz, control_hz=control_hz,
            physics_steps=step_count, control_ticks=control_ticks,
            record_rows=(rec.n_rows if rec is not None else 0),
            show=show, live_samples_pushed=(live_pushed if show else None),
            max_control_dt_s=max_dt_actual, t_series=t_series)
        viz = _visualization_payload(show, viz_diag)
        wall = time.perf_counter() - wall_start
        fp = _fingerprint(params, mode, out_dir, rec, step_count, wall,
                          completed=(not failed) and not stopped_early and not ik_saturated,
                          reason=reason or ("; ".join(failed) if failed else ""),
                          w=w, schedule_path=schedule_path, schedule_sha=schedule_sha,
                          sfc_par=sfc_par, sfc_formal=(ctrl.is_formal if ctrl else None),
                          dt_max_s=dt_max_s, force_mapping=force_mapping_note,
                          stopped_early=stopped_early,
                          ik_saturated=ik_saturated, physics_hz=physics_hz,
                          control_hz=control_hz, duration=duration,
                          integrity=integrity, visualization=viz)
        if rec is not None:
            rec.write_fingerprint(fp)
            note = w.describe()
            rec.write_summary(wall, note=note)
        if verbose:
            print(f"[run] 完成" if completed else "[run] 失败/中止",
                  f"：{rec.n_rows if rec else 0} 帧，物理步 {step_count}，"
                  f"结束 t={(rec._last_t if rec and rec._last_t is not None else t):.3f}s，"
                  f"耗时 {wall:.1f}s，{w.describe()}", flush=True)
            print(f"[run] execution_integrity: 物理 {integrity['physics_steps']}/"
                  f"{integrity['expected_physics_steps']}({'OK' if integrity['physics_steps_ok'] else '截断/中止'}) "
                  f"ticks==rows {integrity['control_ticks_record_rows_ok']} "
                  f"push==ticks {integrity['live_samples_pushed_ok']} "
                  f"t 严格递增 {integrity['trajectory_t_strictly_increasing']} "
                  f"dt≤上界 {integrity['max_control_dt_ok']}", flush=True)

    return {
        "mode": mode, "out_dir": str(out_dir),
        "rows": (rec.n_rows if rec is not None else 0),
        "physics_steps": step_count,
        "t_end": (rec._last_t if rec is not None and rec._last_t is not None else t),
        "wall_s": wall, "stopped_early": stopped_early,
        "ik_saturated": ik_saturated, "completed": completed,
        "w_sha256": w.sha256, "schedule_sha256": schedule_sha,
        "dt_max_s": dt_max_s,
    }


def _visualization_payload(show: bool, diag: dict | None) -> dict[str, Any]:
    if not show:
        return {"requested": False}
    return dict(diag) if diag else {"requested": True}


def _fingerprint(params, mode, out_dir, rec, step_count, wall, *, completed, reason,
                 w, schedule_path, schedule_sha, sfc_par, sfc_formal, dt_max_s,
                 force_mapping, stopped_early, ik_saturated, physics_hz, control_hz,
                 duration, integrity, visualization) -> dict:
    effective = _effective_params(params)
    return {
        "version": "v1.3",
        "mode": mode,
        "completed": bool(completed),
        "reason": reason,
        "stopped_early": bool(stopped_early),
        "ik_saturated": bool(ik_saturated),
        "out_dir": str(out_dir),
        "rows": int(rec.n_rows) if rec is not None else 0,
        "physics_steps": int(step_count),
        "wall_s": round(wall, 3),
        "params_fingerprint": provenance.json_fingerprint(effective),
        "params": effective,
        "disturbance": {"file": config.project_relative_str(w.path), "sha256": w.sha256},
        "schedule": {"file": (config.project_relative_str(schedule_path)
                              if schedule_sha is not None else None), "sha256": schedule_sha},
        "model": provenance.model_fingerprint(),
        "init_q": [round(float(x), 12) for x in config.INIT_Q],
        "physics_hz": int(physics_hz), "control_hz": float(control_hz),
        "duration_s": float(duration),
        "force_mapping": dict(force_mapping) if force_mapping else None,
        "sfc": dict(sfc_par) if sfc_par else None,
        "sfc_paper_consistent": sfc_formal,
        "dt_max_s": dt_max_s,
        "execution_integrity": integrity,
        "visualization": visualization,
        "control_chain": ("e_y -> a_y_est -> F_vir=+force_map_mass_kg*a_y_est -> "
                          "SFC(m,mu,n,g) -> v_sfc_out=g*v_s -> runtime y_sfc_offset -> "
                          "y_cmd=y_ref+y_sfc_offset -> DLS-IK -> MuJoCo joint position servo"),
        "position_servo_adaptation": (
            "y_sfc_offset 的运行层积分是“把 SFC 输出速度接到现有 MuJoCo 关节位置伺服”的"
            "接口适配，不是 SFC 核心；SFC 核心只输出 v_sfc_out=g·v_s(m/s)。"
            "本工程不做直接力矩控制，也不修改 actuator XML。"),
        "servo_note": config.SERVO_NOTE,
    }


def _pos_pace(text: str) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--pace 必须是数值") from None
    if not (v > 0.0):
        raise argparse.ArgumentTypeError("--pace 必须 > 0（不可为 0/负 = 不限速）")
    return v


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="APF-SFC MuJoCo 单次运行")
    p.add_argument("--params", default="", help="参数 JSON 路径；缺省用默认参数")
    p.add_argument("--mode", required=True, choices=sorted(config.VALID_RUN_MODES),
                   help="baseline=A组(无附加控制) / apf_sfc=B组")
    p.add_argument("--out", default="", help="输出目录（缺省自动生成 outputs/run_*）")
    p.add_argument("--stop-path", default="", help="存在即提前停止的信号文件")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true",
                   help="严格观赛：3D+实时 e_y 图都必须就绪，任一失败在 t=0 前报错退出（零落盘）")
    g.add_argument("--show-best-effort", action="store_true",
                   help="尽力观赛：任一窗口失败则降级无头继续（不报错）")
    p.add_argument("--pace", type=_pos_pace, default=1.0,
                   help="真实时倍率（1=真实时；>1 越接近尽快；仅观赛时生效）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    if args.params:
        params = config.load_parameters(Path(args.params))
    else:
        params = config.parameter_defaults()

    errors = config.validate_params(params)
    if errors:
        for e in errors:
            print(f"[config] {e}", file=sys.stderr)
        return 2

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = config.OUTPUT_ROOT / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    stop_path = Path(args.stop_path) if args.stop_path else None
    show = bool(args.show or args.show_best_effort)
    try:
        run_single(params, args.mode, out_dir, stop_path=stop_path,
                   show=show, show_best_effort=args.show_best_effort,
                   pace_rate=args.pace)
    except LiveViewError as exc:
        print(f"[run] 可视化启动失败（t=0 前中止，未产生输出）：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[run] 失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
