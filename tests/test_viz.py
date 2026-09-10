"""
V1.2 观赛链无头自检（对应《运行时可视化 DS 任务书》T-VIS 可自动化的部分）。

可自动化项（本文件，需带 mujoco 的 venv —— 克隆隔离那几项要真的 load XML）：
- T-VIS-05 滚动窗横轴表 + 图内“恰一条 e_y 线” + 纵轴只扩不缩 + 抽稀 + 缓冲裁剪；
- T-VIS-02 的 run 结束自动检查 `check_execution_integrity` 纯函数单测
  （含 expected=ceil((62.853−1e-9)·1000)=62853、等式不变量、dt 上界、单调向量）；
- 部分 T-VIS-07 `GUI_FORCED_OFF`：strict begin 抛 LiveViewError / best-effort 不开窗返回；
- 克隆 observer 隔离：`_load_clone()` 的 model/data 与主仿真不同对象。

需真窗人工确认的项（本文件不碰，避免弹窗/点击）：
T-VIS-01（窗口就绪后才 t=0、无开局快进）、T-VIS-03（人为拖慢图表）、T-VIS-04（关窗不中止）、
T-VIS-06（正式 pair --show 视觉公平）、T-VIS-07 真缺组件路径 —— 这些归 #35 冒烟/人工，
RESULTS 里会以 PASSED-BY-HAND/PENDING-USER 另行记录。

用法：<venv>/python.exe tests/test_viz.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import liveview
from liveview import (  # noqa: E402
    DEFAULT_Y_LIMIT_UM,
    LiveView,
    LiveViewError,
    ceil_decimate,
    derive_y_limit_um,
    make_e_y_figure,
    rolling_window_bounds,
    symmetric_y_limit,
    trim_history,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = ""):
    RESULTS.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


# ---------------------------------------------------------------- T-VIS-05 滚动窗
def t_rolling():
    # 任务书 §5 / §6 T-VIS-05 表：0/5/10 s → [0,10]，15 s → [5,15]
    for tmax, expect in ((0.0, (0.0, 10.0)), (5.0, (0.0, 10.0)),
                         (10.0, (0.0, 10.0)), (15.0, (5.0, 15.0))):
        got = rolling_window_bounds(tmax, 10.0)
        check(f"rolling: tmax={tmax} → {expect}", got == expect, f"got {got}")
    check("rolling: 长窗 12s@10", rolling_window_bounds(22.0, 10.0) == (12.0, 22.0))


def t_decimate_trim():
    x = np.arange(1000, dtype=float)
    y = x * 2.0
    dx, dy = ceil_decimate(x, y, 100)
    check("decimate: ≤max_pts", dx.size <= 100, f"{dx.size} 点")
    check("decimate: 含最新样本(末点)", dx[-1] == x[-1] and dy[-1] == y[-1])
    check("decimate: 含首点且有序", dx[0] == 0.0 and np.all(np.diff(dx) > 0))
    xs, ys = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    dx2, dy2 = ceil_decimate(xs, ys, 10)
    check("decimate: 不抽稀原样返回", dx2 is xs or np.array_equal(dx2, xs))
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    ey = [float(v) for v in ts]
    trim_history(ts, ey, 2.5)
    check("trim: 保留 ≥keep_from", ts == [3.0, 4.0], f"{ts}")
    trim_history(ts, ey, 99.0)
    check("trim: 全清不报错", ts == [] and ey == [])
    ts2 = [0.0, 1.0, 2.0]
    ey2 = list(ts2)
    trim_history(ts2, ey2, -1.0)
    check("trim: keep_from 在起点前不误删", ts2 == [0.0, 1.0, 2.0])


def t_axis_growth():
    check("axis: 越界只扩(120>100)", symmetric_y_limit(100.0, 120.0) == 120.0)
    check("axis: 未越界不缩(50)", symmetric_y_limit(100.0, 50.0) == 100.0)
    check("axis: 相等保持", symmetric_y_limit(100.0, 100.0) == 100.0)
    check("axis: 顺序无关 max", symmetric_y_limit(80.0, 250.0) == 250.0)


def t_chart_agg():
    # T-VIS-05：图内只有一条 e_y 数据线（无 w/高通/PSD/第二条线）
    import matplotlib
    matplotlib.use("Agg")            # 无头；必须在 pyplot 前
    fig, ax, line = make_e_y_figure("t-unit", y_limit_um=200.0, window_s=10.0)
    try:
        check("chart: 恰一条数据线", len(ax.lines) == 1)
        check("chart: 线初始为空集", line.get_xdata().size == 0 and line.get_ydata().size == 0)
        lo, hi = ax.get_xlim()
        check("chart: xlim 初值 [0,10]", math.isclose(lo, 0.0) and math.isclose(hi, 10.0),
              f"[{lo:g},{hi:g}]")
        yl, yh = ax.get_ylim()
        check("chart: ylim 对称 ±200", math.isclose(yl, -200.0) and math.isclose(yh, 200.0))
        check("chart: ylabel 含 µm", "µm" in ax.get_ylabel())
        check("chart: xlabel 为 t / s", "t / s" in ax.get_xlabel())
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def t_derive_limit():
    # 正式包 e_des_um.csv → ceil(1.2×max|e_des|)µm（同一正式 pair A/B 共用一次）
    v, src = derive_y_limit_um(config.OUTPUT_ROOT / "wfit_P05R01")
    check("derive: 从冻结包得 y_limit", v is not None and v == 151.0, f"v={v}")
    check("derive: 来源非 missing", v is not None and src.startswith("e_des_um.csv"), src)
    with tempfile.TemporaryDirectory() as td:
        v2, s2 = derive_y_limit_um(td)
        check("derive: 缺文件→(None,missing)", v2 is None and s2 == "missing")
    check("derive: 默认常量>0", DEFAULT_Y_LIMIT_UM == 200.0)


# ---------------------------------------------------------------- T-VIS-02 执行完整性
def t_exec_integrity():
    import run as run_mod
    f = run_mod.check_execution_integrity
    # expected = ceil((62.853 − 1e-9)·1000) = 62853
    base = dict(duration_s=62.853, physics_hz=1000, control_hz=1000.0,
                physics_steps=62853, control_ticks=62853, record_rows=62853,
                show=False, live_samples_pushed=None,
                max_control_dt_s=0.0010000000000000009,
                t_series=np.arange(62853) / 1000.0)
    r = f(**base)
    check("int: expected=62853", r["expected_physics_steps"] == 62853)
    check("int: physics_steps_ok(跑满)", r["physics_steps_ok"] is True)
    check("int: ticks==rows", r["control_ticks_record_rows_ok"] is True)
    check("int: push 非 show 为 None", r["live_samples_pushed_ok"] is None)
    check("int: t 严格递增", r["trajectory_t_strictly_increasing"] is True)
    check("int: dt≤上界", r["max_control_dt_ok"] is True)
    # 提前停止/中止 → 如实 False（不假装“跑满”）
    cut = dict(base); cut["physics_steps"] = 1000
    r2 = f(**cut)
    check("int: 提前停 physics_steps_ok=False",
          r2["physics_steps_ok"] is False and r2["physics_steps"] == 1000)
    # show 时 pushed==ticks 才 OK；pushed 少 → False
    sh = dict(base); sh["show"] = True; sh["live_samples_pushed"] = 62853
    r3 = f(**sh)
    check("int: show pushed==ticks", r3["live_samples_pushed_ok"] is True and
          r3["live_samples_pushed"] == 62853)
    sh2 = dict(sh); sh2["live_samples_pushed"] = 60000
    r4 = f(**sh2)
    check("int: pushed≠ticks→False", r4["live_samples_pushed_ok"] is False)
    # t 不严格递增（重复时间戳）→ False
    dup = dict(base); dup["t_series"] = np.array([0.0, 0.0, 0.001])
    r5 = f(**dup)
    check("int: t 重复→非严格递增", r5["trajectory_t_strictly_increasing"] is False)
    down = dict(base); down["t_series"] = np.array([1.0, 0.5, 0.0])
    r6 = f(**down)
    check("int: t 下降→False", r6["trajectory_t_strictly_increasing"] is False)
    # dt 超上界 → False；None（无控制节拍样本）→ None
    allow = 1.0 / 1000.0 + 1.0 / 1000.0 + 1e-9
    over = dict(base); over["max_control_dt_s"] = allow + 1e-3
    check("int: dt 超上界→False", f(**over)["max_control_dt_ok"] is False)
    none = dict(base); none["max_control_dt_s"] = None
    check("int: dt=None→None", f(**none)["max_control_dt_ok"] is None)


# ---------------------------------------------------------------- T-VIS-07(部分) GUI 强制无头
def t_gui_forced_off():
    saved = liveview.GUI_FORCED_OFF
    liveview.GUI_FORCED_OFF = True
    try:
        lv = LiveView(xml_path=str(config.MODEL_XML_PATH), init_q=config.INIT_Q,
                      tcp0=np.zeros(3), title="t-gui-off", strict=True)
        try:
            lv.begin()
            check("gui_off: strict 应抛 LiveViewError", False)
        except LiveViewError as exc:
            check("gui_off: strict 抛 LiveViewError", True, f"{exc}")
            check("gui_off: component=viewer", exc.component == "viewer")
        lv2 = LiveView(xml_path=str(config.MODEL_XML_PATH), init_q=config.INIT_Q,
                       tcp0=np.zeros(3), title="t-gui-off-best", strict=False)
        lv2.begin()                       # best-effort：不抛
        check("gui_off: best-effort 不抛", True)
        check("gui_off: best-effort 未开任何窗", not lv2.has_any())
        diag = lv2.diagnostics()
        kinds = [k for (_c, k, _m) in [tuple(e) for e in diag["errors"]]]
        check("gui_off: 错误记录了 APFSFC_NOGUI", "APFSFC_NOGUI" in kinds)
        check("gui_off: errors 含两组件", len(diag["errors"]) == 2)
    finally:
        liveview.GUI_FORCED_OFF = saved


# ---------------------------------------------------------------- 克隆 observer 隔离
def t_clone_isolation():
    import mujoco
    xml = str(config.MODEL_XML_PATH)
    main_m = mujoco.MjModel.from_xml_path(xml)
    main_d = mujoco.MjData(main_m)
    lv = LiveView(xml_path=xml, init_q=np.asarray(config.INIT_Q, dtype=float),
                  tcp0=np.zeros(3), title="t-clone")
    lv._load_clone()
    try:
        check("clone: model 与主仿真非同一对象", lv._clone_m is not main_m)
        check("clone: data 与主仿真非同一对象", lv._clone_d is not main_d)
        check("clone: 与独立重载也非同一对象",
              lv._clone_m is not mujoco.MjModel.from_xml_path(xml))
        nq = min(lv._clone_d.qpos.size, main_m.nq)
        check("clone: qpos 已写 init_q",
              np.allclose(np.asarray(lv._clone_d.qpos)[:nq],
                          np.asarray(config.INIT_Q)[:nq]))
        check("clone: mj_forward 后位姿有限",
              np.all(np.isfinite(lv._clone_d.qpos)))
        # 内存独立性：写 clone.qpos 后 main_d.qpos 不变（GUI 只能动 clone，回不去主仿真）
        before = np.array(main_d.qpos, copy=True)
        if lv._clone_m.nu:
            lv._clone_d.ctrl[:] = 1.0
        lv._clone_d.qpos[:] = np.linspace(0.0, 0.1, lv._clone_d.qpos.size)
        check("clone: 写 clone qpos 后 main qpos 未变", np.array_equal(main_d.qpos, before))
    finally:
        # 不 launch 窗口，无需 close；清理引用即可
        lv._viewer = None
        lv._fig = None


def main() -> int:
    for fn in (t_rolling, t_decimate_trim, t_axis_growth, t_chart_agg,
               t_derive_limit, t_exec_integrity, t_gui_forced_off, t_clone_isolation):
        try:
            fn()
        except Exception as exc:
            RESULTS.append((fn.__name__, False, f"异常：{exc!r}"))
            print(f"[FAIL] {fn.__name__} 抛异常：{exc!r}")
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n共 {len(RESULTS)} 项：PASS {len(RESULTS) - len(failed)}，FAIL {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        for s in (sys.stdout, sys.stderr):
            s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
