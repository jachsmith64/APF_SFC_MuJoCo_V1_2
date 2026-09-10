"""
运行时可视化（V1.2 改造版）：MuJoCo 3D 窗口 + 实时 e_y 图 + 真实时 pacing（克隆 observer）。

改造要点（对应《运行时可视化 DS 任务书》§1–§8）：
1. 只读隔离：本模块**不再持有主仿真的 model/data**。它在 begin() 时从同一 XML 重新加载一份
   clone MjModel/MjData，唯一把这份 clone 交给 mujoco.viewer.launch_passive。GUI 的任何
   扰动/暂停/选项修改都只落在 clone 上；主仿真（步进 + 记录）的 model/data 从不出现在 viewer，
   因此开启/关闭显示不改变主仿真的数值序列（该声明由 T-VIS-02 数值列复核支撑，非设计自证）。
2. pacing 零点（_t0_wall）只在两个窗口都就绪并 warm 后记录——t=0 不再开局快进。
3. 图表显式非阻塞显示：plt.ion() + plt.show(block=False) + canvas.draw/flush_events，
   不用 plt.pause()。横轴前 window_s 秒固定 [0, window_s]，之后滚动最近 window_s 秒；
   纵轴以 0 为对称中心的固定 y_limit_um，越界只向外扩、不缩回；A/B 由调用方传同一 y_limit_um。
4. 严格模式：默认 --show 期望两个窗口；任一失败在 t=0 前抛 LiveViewError（调用方据此干净中止、
   零落盘）。--show-best-effort 才允许降级继续。
5. 证据计数：viewer_syncs / chart_redraws / samples_received 供 run.py 写
   run_fingerprint.json 的 visualization 区（只读诊断，不进 A/B 公平性比较）。
6. GUI 异常全部记成 (component, kind, message)，绝不裸 except: pass。

线程模型（不变约束）：
- mujoco passive viewer（run_physics_thread=False）自己不 mj_step，后台 GLFW 线程渲染 clone；
- matplotlib/TkAgg 图只由主线程驱动（physics 主循环里节流 draw_idle+flush_events）；
- 真实时 pacing 用 ~30Hz 整体追赶式 sleep，避免 Windows 1ms 粒度 overshoot；
- 关闭任一只窗只停止该画面，仿真继续并正常落盘。

本模块顶层 import 零副作用（不 import mujoco/matplotlib），APFSFC_NOGUI=1 强制无头。
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import Any

import numpy as np

GUI_FORCED_OFF: bool = os.environ.get("APFSFC_NOGUI") == "1"

_VIEW_FPS = 50.0          # 3D 同步上限
_CHART_FPS = 15.0         # 实时图刷新上限（任务书建议 10–15 fps）
_PACE_HZ = 30.0           # 真实时追赶检查频率
_VIEWER_OPEN_TIMEOUT_S = 10.0   # 3D 窗口就绪超时（防 GLFW 失败挂死）
_WARM_S = 0.15            # 就绪后、pacing 记零前的首次绘制安定时间
_CLOSE_SETTLE_S = 0.2     # close 后让 GLFW/Tk 线程先收尾（A→B 同进程重开用）
_CHART_MARGIN_S = 2.0     # 历史缓冲额外余量
DEFAULT_Y_LIMIT_UM = 200.0

_MAX_PTS = 4000           # 显示抽稀上限


def _note(msg: str) -> None:
    print(f"[liveview] {msg}", flush=True)


class LiveViewError(Exception):
    """严格观赛启动失败。携带 (component, kind, message) 供上层明确报错。"""

    def __init__(self, component: str, kind: str, message: str):
        super().__init__(f"{component}: {kind}: {message}")
        self.component = component
        self.kind = kind
        self.message = message


# ---------------------------------------------------------------------------
# 纯函数（供图表 / run.py 复用，并可无头单测 —— tests/test_viz.py）
# ---------------------------------------------------------------------------

def rolling_window_bounds(tmax: float, window_s: float) -> tuple[float, float]:
    """前 window_s 秒固定 [0, window_s]，之后滚动最近 window_s 秒。"""
    w = float(window_s)
    if float(tmax) <= w:
        return 0.0, w
    return float(tmax) - w, float(tmax)


def ceil_decimate(x: np.ndarray, y: np.ndarray, max_pts: int):
    """抽稀到 ≤ max_pts 点，且必含首点与末点（最新样本）——显示点数严格不超过上限。"""
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if n <= max_pts:
        return x, y
    m = int(max_pts)
    idx = np.unique(np.round(np.linspace(0.0, n - 1, num=m)).astype(int))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    idx = np.unique(idx)
    return x[idx], y[idx]


def trim_history(ts: list, ey: list, keep_from: float) -> None:
    """原地裁掉 keep_from 之前的历史点。"""
    i = 0
    n = len(ts)
    while i < n and ts[i] < keep_from:
        i += 1
    if i:
        del ts[:i]
        del ey[:i]


def symmetric_y_limit(base_um: float, peak_abs: float) -> float:
    """以 0 为中心的对称纵轴半宽：只扩不缩。"""
    return max(float(base_um), float(peak_abs))


def make_e_y_figure(title: str, y_limit_um: float, window_s: float):
    """建 e_y 图（仅一条数据线，无 w/高通/PSD/在线指标）。调用方需先选好后端。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=100)
    (line,) = ax.plot([], [], lw=1.0, color="#c0392b")
    ax.set_xlabel("t / s")
    ax.set_ylabel("e_y = y_act − y_ref  /  µm")
    ax.set_title(str(title))
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0.0, float(window_s))
    L = float(abs(y_limit_um))
    if L <= 0.0:
        L = 1.0
    ax.set_ylim(-L, L)
    fig.tight_layout()
    return fig, ax, line


def derive_y_limit_um(dir_of_w) -> tuple[float | None, str]:
    """
    按任务书建议：y_limit_um = ceil(1.2 × max|e_des|)（e_des_um.csv 值列单位 µm）。
    返回 (值, 来源串)；无该文件/列异常 → (None, "missing")。
    """
    import math as _m
    p = str(dir_of_w)
    from pathlib import Path
    cand = Path(p) / "e_des_um.csv"
    if not cand.is_file():
        return None, "missing"
    try:
        arr = np.loadtxt(str(cand), delimiter=",", comments="#")
        if arr.ndim == 1 or arr.shape[1] < 2:
            return None, "missing-shape"
        vals = arr[:, 1]
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            return None, "missing-nonfinite"
        peak = float(np.max(np.abs(vals)))
    except (OSError, ValueError):
        return None, "missing-read"
    val = int(_m.ceil(1.2 * peak))
    return val, f"e_des_um.csv ceil(1.2×max|e_des|)={val}µm"


# ---------------------------------------------------------------------------
# LiveView
# ---------------------------------------------------------------------------

class LiveView:
    """一场 run 的实时观赛组件。构造后可 push()/after_step()；begin() 开启窗口。"""

    def __init__(
        self,
        *,
        xml_path: str,
        init_q,
        tcp0: np.ndarray,
        title: str,
        window_s: float = 10.0,
        y_limit_um: float = DEFAULT_Y_LIMIT_UM,
        y_limit_source: str = "default",
        max_pts: int = _MAX_PTS,
        pace_rate: float = 1.0,
        strict: bool = True,
        overlay_label: str = "",
    ):
        self.xml_path = str(xml_path)
        self.init_q = np.asarray(init_q, dtype=float)
        self.tcp0 = np.asarray(tcp0, dtype=float)
        self.title = str(title)
        self.window_s = float(window_s)
        self.y_limit_um = float(y_limit_um)
        self.y_limit_source = str(y_limit_source)
        self.max_pts = int(max_pts)
        self.pace_rate = float(pace_rate) if float(pace_rate) > 0.0 else 0.0
        self.strict = bool(strict)
        self.overlay_label = str(overlay_label)

        # 显示对象（只在 begin() 里建立）
        self._clone_m: Any | None = None
        self._clone_d: Any | None = None
        self._viewer = None
        self._fig = None
        self._ax = None
        self._line = None
        self._canvas = None

        # 曲线缓冲与纵轴
        self._ts: list[float] = []
        self._ey: list[float] = []
        self._y_lim_um = float(y_limit_um)

        # 诊断计数（close() 不清空，供 run.py 在 close 后读取写指纹）
        self._viewer_syncs = 0
        self._chart_redraws = 0
        self._samples_received = 0
        self._errors: list[tuple[str, str, str]] = []
        self._opened = {"viewer": False, "chart": False}
        self._closed = False

        # pacing / 节流
        self._t0_wall: float | None = None
        self._last_view = 0.0
        self._last_chart = 0.0
        self._last_pace = 0.0
        self._pace_active = False
        # 图刷新自适应：记录上一次整帧重绘耗时，慢画布自动降低刷新率（显示可丢帧，主循环不被饿死）
        self._last_draw_s = 0.0

    # ------------------------------------------------------------------ 状态
    def has_any(self) -> bool:
        return self._viewer is not None or self._fig is not None

    def component_status(self) -> dict:
        def st(comp, obj) -> str:
            if obj is not None:
                return "on"
            msgs = [f"{k}: {m}" for (c, k, m) in self._errors if c == comp]
            if self._opened.get(comp):
                return f"closed({msgs[0]})" if msgs else "closed"
            return f"off({msgs[0]})" if msgs else "off"
        return {"viewer": st("viewer", self._viewer),
                "chart": st("chart", self._fig)}

    def summary(self) -> str:
        s = self.component_status()
        return f"viewer={s['viewer']} chart={s['chart']}"

    def diagnostics(self) -> dict:
        s = self.component_status()
        return {
            "requested": True,
            "strict": self.strict,
            "pace_rate": self.pace_rate,
            "window_s": self.window_s,
            "y_limit_um": self.y_limit_um,
            "y_limit_source": self.y_limit_source,
            "viewer_started": bool(self._opened["viewer"]),
            "chart_started": bool(self._opened["chart"]),
            "viewer": s["viewer"],
            "chart": s["chart"],
            "viewer_syncs": self._viewer_syncs,
            "chart_redraws": self._chart_redraws,
            "samples_received": self._samples_received,
            "errors": [list(e) for e in self._errors],
        }

    # ------------------------------------------------------------------ 生命周期
    def begin(self) -> None:
        """
        打开 3D 与图窗，安定后记录 pacing 零点。strict 失败会先关已开窗再抛 LiveViewError
        （此时尚未有任何物理步/落盘 → 调用方可干净中止）。
        """
        self._closed = False
        if GUI_FORCED_OFF:
            msg = "APFSFC_NOGUI=1 强制无头"
            if self.strict:
                raise LiveViewError("viewer", "APFSFC_NOGUI", msg)
            self._errors.append(("viewer", "APFSFC_NOGUI", msg))
            self._errors.append(("chart", "APFSFC_NOGUI", msg))
            self._pace_active = False
            self._t0_wall = None
            return
        try:
            self._load_clone()
            self._open_viewer()
            self._open_chart()
        except Exception as exc:
            self._close_windows()
            if self.strict:
                raise LiveViewError("general", type(exc).__name__, str(exc)) from exc
            self._errors.append(("general", type(exc).__name__, str(exc)))
            return
        if self.strict:
            missing = []
            if self._viewer is None:
                missing.append("3D viewer")
            if self._fig is None:
                missing.append("chart")
            if missing:
                self._close_windows()
                detail = "; ".join(f"{k}: {m}" for (c, k, m) in self._errors) or "unknown"
                raise LiveViewError("begin", "NotReady",
                                    f"窗口未就绪：{', '.join(missing)}（{detail}）")
        self._warm()
        self._pace_active = (self.pace_rate > 0.0 and self.has_any())
        # P0：pacing 零点只在此刻记录（两窗已就绪并 warm），消除开局快进。
        self._t0_wall = time.perf_counter()

    def _load_clone(self) -> None:
        """从同一 XML 加载独立 model/data（与主仿真完全不同的对象）。"""
        import mujoco
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        d = mujoco.MjData(m)
        nq = min(d.qpos.size, self.init_q.size)
        d.qpos[:nq] = self.init_q[:nq]
        d.qvel[:] = 0.0
        if m.nu:                    # nu=执行器数（mujoco 3.8）；data.ctrl 长度=nu
            d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        self._clone_m = m
        self._clone_d = d

    def _open_viewer(self):
        """把 clone 交给 passive viewer。用包装线程 + 队列 + 超时防 GLFW 失败挂死。"""
        if self._clone_m is None:
            return
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                import mujoco.viewer
                h = mujoco.viewer.launch_passive(
                    self._clone_m, self._clone_d,
                    show_left_ui=False, show_right_ui=False)
                q.put(("ok", h))
            except BaseException as exc:  # noqa: BLE001 含 FatalError
                q.put(("err", exc))

        threading.Thread(target=worker, daemon=True).start()
        try:
            status, payload = q.get(timeout=_VIEWER_OPEN_TIMEOUT_S)
        except queue.Empty:
            self._errors.append(("viewer", "Timeout",
                                 f"{_VIEWER_OPEN_TIMEOUT_S:.0f}s 内未就绪（无显示/GLFW 挂起）"))
            return
        if status == "err":
            exc = payload
            self._errors.append(("viewer", type(exc).__name__, str(exc)))
            return
        handle = payload
        # 相机 best-effort（须在 lock 内改）
        try:
            with handle.lock():
                cam = handle.cam
                cam.lookat[:] = self.tcp0
                cam.distance = 1.8
                cam.azimuth = 120.0
                cam.elevation = -25.0
        except Exception as exc:  # noqa: BLE001
            self._errors.append(("viewer", "Camera", f"{type(exc).__name__}: {exc}"))
        if self.overlay_label:
            try:
                import mujoco
                handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,
                                   mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   self.overlay_label, None)])
            except Exception as exc:  # noqa: BLE001
                self._errors.append(("viewer", "OverlayText",
                                     f"{type(exc).__name__}: {exc}"))
        self._viewer = handle
        self._opened["viewer"] = True

    def _open_chart(self):
        """显式、非阻塞地开 e_y 图；图窗 best-effort 放屏幕右侧。"""
        try:
            import matplotlib
            if not str(matplotlib.get_backend()).lower().endswith("tkagg"):
                matplotlib.use("TkAgg")          # 必须在 pyplot 前；二次 begin 同进程 no-op
            import tkinter as tk                 # noqa: F401 提前触发 TclError
        except Exception as exc:  # noqa: BLE001
            self._errors.append(("chart", "Backend", f"{type(exc).__name__}: {exc}"))
            return
        try:
            fig, ax, line = make_e_y_figure(self.title, self.y_limit_um, self.window_s)
            import matplotlib.pyplot as plt
            plt.ion()
            plt.show(block=False)
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception as exc:  # noqa: BLE001
            self._errors.append(("chart", "Open", f"{type(exc).__name__}: {exc}"))
            return
        self._place_chart_right(fig)
        self._fig = fig
        self._ax = ax
        self._line = line
        self._canvas = fig.canvas
        self._opened["chart"] = True

    @staticmethod
    def _place_chart_right(fig) -> None:
        try:
            mgr = fig.canvas.manager
            win = mgr.window
            if win is None:
                return
            win.update_idletasks()
            try:
                screen_w = win.winfo_screenwidth()
            except Exception:  # noqa: BLE001
                return
            w_px = int(fig.get_figwidth() * fig.dpi)
            x = max(0, int(screen_w) - w_px - 90)
            try:
                win.geometry(f"+{x}+60")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001  best-effort
            pass

    def _warm(self) -> None:
        """首个可显示帧：viewer 同步一次、图 draw/flush 一次，短暂安定。"""
        if self._viewer is not None:
            try:
                self._push_clone_to_viewer(0.0, self.init_q, np.zeros_like(self.init_q))
            except Exception as exc:  # noqa: BLE001
                self._errors.append(("viewer", "Warm", f"{type(exc).__name__}: {exc}"))
            time.sleep(_WARM_S)
        if self._fig is not None and self._canvas is not None:
            try:
                self._canvas.draw()
                self._canvas.flush_events()
            except Exception as exc:  # noqa: BLE001
                self._drop_chart(f"Warm: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_windows()

    def _close_windows(self) -> None:
        v = self._viewer
        if v is not None:
            try:
                v.close()
            except Exception as exc:  # noqa: BLE001
                self._errors.append(("viewer", "Close", f"{type(exc).__name__}: {exc}"))
            self._viewer = None
            time.sleep(_CLOSE_SETTLE_S)   # 让 GLFW 线程收尾，便于同进程再开（pair B）
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(("chart", "Close", f"{type(exc).__name__}: {exc}"))
            self._fig = None
            self._ax = None
            self._line = None
            self._canvas = None

    # ------------------------------------------------------------------ 数据
    def push(self, t: float, e_y_um: float) -> None:
        """控制节拍调用：追加一个 e_y 样本（µm），缓冲只留最近 window_s+margin。"""
        self._ts.append(float(t))
        self._ey.append(float(e_y_um))
        self._samples_received += 1
        self._y_lim_um = symmetric_y_limit(self._y_lim_um, abs(float(e_y_um)))
        trim_history(self._ts, self._ey, float(t) - (self.window_s + _CHART_MARGIN_S))

    # ------------------------------------------------------------------ 每物理步
    def after_step(self, t: float, qpos, qvel) -> None:
        now = time.perf_counter()
        if self._viewer is not None and now - self._last_view >= 1.0 / _VIEW_FPS:
            self._last_view = now
            try:
                self._push_clone_to_viewer(float(t), qpos, qvel)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(("viewer", "Sync", f"{type(exc).__name__}: {exc}"))
                self._drop_viewer("Sync 异常，停用 3D 画面（仿真继续）")
        if self._fig is not None:
            # 刷新间隔 = max(1/15s, 3×上次整帧重绘耗时)：健康机器照常 15fps；
            # 慢/远程桌面画布（重绘可达几十 ms）自动降频，保证绘制只占小部分墙钟。
            min_gap = max(1.0 / _CHART_FPS, 3.0 * self._last_draw_s)
            if now - self._last_chart >= min_gap:
                self._last_chart = now
                _t0 = time.perf_counter()
                self._redraw_chart(float(t))
                self._last_draw_s = time.perf_counter() - _t0
        if self._pace_active and now - self._last_pace >= 1.0 / _PACE_HZ:
            self._last_pace = now
            self._pace(float(t))

    def _push_clone_to_viewer(self, t: float, qpos, qvel) -> None:
        import mujoco
        handle = self._viewer
        if handle is None:
            return
        if not handle.is_running():
            self._drop_viewer("3D 窗口已关闭（仅该画面停止，仿真继续）")
            return
        d = self._clone_d
        d.time = float(t)
        d.qpos[:] = np.asarray(qpos, dtype=d.qpos.dtype)[: d.qpos.size]
        d.qvel[:] = np.asarray(qvel, dtype=d.qvel.dtype)[: d.qvel.size]
        mujoco.mj_forward(self._clone_m, d)
        handle.sync(state_only=True)
        self._viewer_syncs += 1

    def _drop_viewer(self, note: str) -> None:
        if self._viewer is None:
            return
        self._viewer = None
        self._errors.append(("viewer", "Closed", note))
        _note(note + "（产物不受影响）")

    def _redraw_chart(self, tmax: float) -> None:
        if not self._ts:
            return
        lo, hi = rolling_window_bounds(tmax, self.window_s)
        try:
            ts = np.asarray(self._ts, dtype=float)
            ey = np.asarray(self._ey, dtype=float)
            m = ts >= lo
            x = ts[m]
            y = ey[m]
            if x.size == 0:
                return
            x, y = ceil_decimate(x, y, self.max_pts)
            self._ax.set_xlim(lo, hi)
            self._ax.set_ylim(-self._y_lim_um, self._y_lim_um)
            try:
                self._ax.set_title(f"{self.title}  t={tmax:.2f} s")
            except Exception:  # noqa: BLE001
                pass
            self._line.set_data(x, y)
            self._canvas.draw_idle()
            self._canvas.flush_events()
            self._chart_redraws += 1
        except Exception as exc:  # noqa: BLE001
            self._drop_chart(f"{type(exc).__name__}: {exc}")

    def _drop_chart(self, note: str) -> None:
        if self._fig is None:
            return
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        except Exception:  # noqa: BLE001
            pass
        self._fig = None
        self._ax = None
        self._line = None
        self._canvas = None
        self._errors.append(("chart", "Closed", note))
        _note("实时图窗口关闭/异常（仅该画面停止，仿真继续，产物不受影响）。")

    # ------------------------------------------------------------------ pacing
    def _pace(self, t: float) -> None:
        """真实时追赶：一次整体 sleep（受 _PACE_HZ 节流），不逐物理步长睡。"""
        if self._t0_wall is None or self.pace_rate <= 0.0:
            return
        target = self._t0_wall + float(t) / self.pace_rate
        delay = target - time.perf_counter()
        if delay > 0.002:
            try:
                time.sleep(min(delay, 0.05))
            except OSError:
                pass
