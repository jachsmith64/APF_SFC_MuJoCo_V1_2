"""
APF-SFC MuJoCo V1.2 启动器（Tk 参数窗）。

V1.2 分工：
- 左侧参数按 config.PARAM_GROUPS 驱动；SFC 论文参数（mu/n/g/B0/K_v）只读显示
  （readonly=True 置灰），来源是 sfc_tune.py 生成的 sfc_tuning.json，不在 UI 手调。
- 右侧按钮：
    · 试跑 A / 试跑 B → run.py 单组，输出 outputs/run_<时间戳>_<mode>（诊断，非正式）
    · 正式 A+B → pair.py 子进程，同一 params 快照跑成唯一 outputs/pair_*/（run_A/run_B
      + analysis/ + manifest.json），gate 不过不进入正式分析；子进程输出流式进日志
    · 停止 → 写 UI_STOP 信号文件，run/pair 优雅结束
    · 查看最近正式结果 → 打开最近 provisional=false 且含 analysis 的 pair 目录
- 高级字段默认折叠；A/B 必须共用同一份 params/w(t)（正式由 pair 保证）。

运行（带 mujoco 的 venv）：python launcher_ui.py
参数存档：outputs/last_params.json
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import BooleanVar, Entry, StringVar, Tk, filedialog, messagebox, ttk

import config

PROJ = config.PROJECT_DIR
LAST_PARAMS = PROJ / "outputs" / "last_params.json"
UI_STOP = PROJ / "outputs" / "ui_stop.request"
_BUSY_END = "__BUSY_END__\n"

# 勾选“运行可视化”后，除 mujoco+numpy 外还须能 import 观赛链组件（viewer/Tk/TkAgg）。
_PROBE_VIZ = ("import mujoco, mujoco.viewer, numpy, tkinter; "
              "import matplotlib.backends.backend_tkagg")

# 本机跑实验的默认解释器（含 mujoco/numpy 的 venv）。改机器/venv 时更新此处，
# 或设环境变量 APFSFC_PYTHON 指向带 mujoco 的 python.exe/pythonw.exe 覆盖它。
RUN_PY_DEFAULT = r"C:\Users\PC\Desktop\code\code\Myproject-1\venv\Scripts\python.exe"

_run_py_cache: dict[str, str | None] = {"path": None}


def _resolve_run_python() -> str | None:
    """
    探测能跑仿真的解释器（能 import mujoco+numpy），返回后缓存。

    候选顺序（去重后逐个探测，第一个成功者即用）：
    1. 环境变量 APFSFC_PYTHON（显式覆盖）；
    2. 启动本 UI 的解释器 sys.executable 本身 —— 若 UI 用 venv 的 pythonw/python 起，
       它自带 mujoco，直接用（保留 pythonw“不弹黑窗”）；
    3. 与 sys.executable 同级的另一形态（pythonw↔python）；
    4. 项目默认 venv（pythonw 优先防黑窗，再 python.exe）。

    关键点：UI 不再依赖“我是被谁启动的”，因此哪怕 UI 被 C:\\Python314 这类无 mujoco 的
    裸解释器拉起，也会自动改用默认 venv，把 ModuleNotFoundError 变成可执行的探测。
    """
    if _run_py_cache["path"] is not None:
        return _run_py_cache["path"]

    cands: list[str] = []

    def add(p: str) -> None:
        p = (p or "").strip()
        if p and Path(p).is_file():
            cands.append(p)

    add(os.environ.get("APFSFC_PYTHON", ""))
    cur = str(sys.executable)
    add(cur)
    cur_p = Path(cur)
    if cur_p.name.lower() == "pythonw.exe":
        add(str(cur_p.with_name("python.exe")))      # pythonw 的同级 python.exe
    else:
        add(str(cur_p.with_name("pythonw.exe")))     # python.exe 的同级 pythonw.exe
    add(str(Path(RUN_PY_DEFAULT).with_name("pythonw.exe")))   # 默认 venv：pythonw 优先
    add(RUN_PY_DEFAULT)

    seen: set[str] = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        try:
            r = subprocess.run([c, "-c", "import mujoco, numpy"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=30)
        except Exception:
            continue
        if r.returncode == 0:
            _run_py_cache["path"] = c
            return c
    return None


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _probe_viz(py: str) -> tuple[bool, str]:
    """
    对已解析的解释器再探测观赛链：能 import mujoco.viewer / tkinter / backend_tkagg。
    返回 (是否就绪, 失败细节)。GUI 环境下该探测可真实打开/失败（import 不弹窗）。
    """
    try:
        r = subprocess.run([py, "-c", _PROBE_VIZ],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=45)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if r.returncode == 0:
        return True, ""
    tail = (r.stderr or r.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit={r.returncode}"
    return False, detail


class _StreamReader(threading.Thread):
    def __init__(self, stream, box_q: "queue.Queue[str]"):
        super().__init__(daemon=True)
        self._s = stream
        self._q = box_q

    def run(self):
        for line in iter(self._s.readline, ""):
            self._q.put(line)


class Launcher:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("APF-SFC MuJoCo 抑振实验 V1.2（UR10e · 正式 w(t) 回放）")
        self.vars: dict[str, StringVar] = {}
        self._param_widgets: dict[str, object] = {}
        self._advanced_rows: list = []
        self._proc: subprocess.Popen | None = None
        self._box_q: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._saved_states: list | None = None   # busy 进入时快照的控件原始 state
        self._last_rc = 0                        # 最近子进程返回码（_BUSY_END 用）
        self._stop_requested_busy = False        # 本次 busy 期间是否请求过停止
        self._cb_show = None
        self._cb_adv = None

        self._build_toolbar()
        self._build_scroll_params()
        self._build_actions()
        self._build_log()
        self._load_defaults()
        self._toggle_advanced()
        self._poll_queue()

    # ------------------------------------------------------------ 顶部工具条
    def _build_toolbar(self):
        import tkinter as tk
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="默认参数", command=self._load_defaults).pack(side="left")
        ttk.Button(bar, text="载入配置…", command=self._load_dialog).pack(side="left", padx=4)
        ttk.Button(bar, text="保存配置…", command=self._save_dialog).pack(side="left")
        self.var_advanced = BooleanVar(value=False)
        self._cb_adv = ttk.Checkbutton(bar, text="高级参数", variable=self.var_advanced,
                                       command=self._toggle_advanced)
        self._cb_adv.pack(side="left", padx=12)
        self.var_show = BooleanVar(value=False)
        self._cb_show = ttk.Checkbutton(bar, text="运行可视化（MuJoCo 窗口 + 实时 e_y 图）",
                                        variable=self.var_show,
                                        command=self._on_show_toggle)
        self._cb_show.pack(side="left", padx=(12, 0))
        note = (f"模型 {config.MODEL_XML_PATH.name} · 只读 SFC 参数来自 sfc_tuning.json · "
                f"扰动唯一来源冻结 w(t)")
        ttk.Label(bar, text=note, foreground="#555").pack(side="left")

    # ------------------------------------------------------------ 参数区
    def _build_scroll_params(self):
        import tkinter as tk
        outer = ttk.Frame(self.root)
        outer.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._params = ttk.Frame(canvas)
        self._params.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._params, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"))
        row = 0
        for group in config.PARAM_GROUPS:
            grp = ttk.LabelFrame(self._params, text=group["group"], padding=6)
            grp.grid(row=row, column=0, sticky="ew", padx=2, pady=4)
            self._params.columnconfigure(0, weight=1)
            for gi, field in enumerate(group["fields"]):
                self._add_field(grp, gi, field)
            row += 1

    def _add_field(self, parent, gi, field):
        key = field["key"]
        label = field["label"] + ("" if field.get("unit") == "" else f"  ({field['unit']})")
        star = " ★" if field.get("advanced") else ""
        ttk.Label(parent, text=label + star).grid(row=gi, column=0, sticky="w", padx=(0, 6))
        var = StringVar()
        self.vars[key] = var
        readonly = bool(field.get("readonly"))
        if field["type"] == "choice":
            cb = ttk.Combobox(parent, textvariable=var, state="readonly",
                              values=field.get("choices", []), width=12)
            cb.grid(row=gi, column=1, sticky="w")
            self._param_widgets[key] = cb
        elif field["type"] == "file":
            box = ttk.Frame(parent)
            ent = Entry(box, textvariable=var, width=34,
                        state="disabled" if readonly else "normal",
                        bg="#ececec" if readonly else "white")
            ent.pack(side="left")
            if not readonly:
                ttk.Button(box, text="浏览…", width=6,
                           command=lambda: var.set(filedialog.askopenfilename())).pack(
                    side="left", padx=2)
            box.grid(row=gi, column=1, sticky="w")
            self._param_widgets[key] = ent
        else:  # float/int
            ent = Entry(parent, textvariable=var, width=14,
                        state="disabled" if readonly else "normal",
                        bg="#ececec" if readonly else "white")
            ent.grid(row=gi, column=1, sticky="w")
            self._param_widgets[key] = ent
        if readonly:
            ttk.Label(parent, text="（论文整定，只读）", foreground="#888").grid(
                row=gi, column=2, sticky="w", padx=(2, 0))
        presets = field.get("presets")
        if presets and not readonly:
            import tkinter as tk
            pv = StringVar()
            om = tk.OptionMenu(parent, pv, "预设", *[p[0] for p in presets],
                               command=lambda _n, f=field, v=var, ps=presets:
                               self._apply_preset(f, v, ps, pv))
            om.config(width=5)
            om.grid(row=gi, column=3, sticky="w", padx=(2, 0))
        self._advanced_rows.append((parent, gi, bool(field.get("advanced"))))

    @staticmethod
    def _apply_preset(field, var, presets, pv):
        for label, val in presets:
            if label == pv.get():
                var.set(str(val))

    def _on_show_toggle(self):
        if self.var_show.get():
            self._log_write("[launcher] 可视化开启：子进程将带 --show（MuJoCo 窗口 + 实时 e_y 图，"
                            "真实时观赛）。子进程会自动探测带 mujoco 的解释器；若探测失败可设"
                            " APFSFC_PYTHON 显式指定。\n")

    def _toggle_advanced(self):
        show = self.var_advanced.get()
        for parent, gi, adv in self._advanced_rows:
            for w in parent.grid_slaves(row=gi):
                if adv and not show:
                    w.grid_remove()
                elif adv and show:
                    w.grid()

    # ------------------------------------------------------------ 按钮
    def _build_actions(self):
        row1 = ttk.Frame(self.root, padding=(6, 2))
        row1.pack(side="top", fill="x")
        self._bt = {}
        # 注意：Tk 的 command= 回调不带参。之前用 `lambda a=arg, c=cmd: c(a)`
        # 给每个处理器硬塞一个参数，导致 _stop(self) 被调成 _stop(None) 抛 TypeError，
        # 停止信号根本没写成；零参 lambda（打开 outputs）同样会被多塞 None 崩掉。
        # 现在每个按钮各自包成真正的零参闭包（试跑/打开 outputs 用 lambda 绑定参数，
        # 其余直接绑无需参的方法），杜绝该类错误。
        btns = [
            ("跑 A（试跑）", lambda: self._run_trial("baseline")),
            ("跑 B（试跑）", lambda: self._run_trial("apf_sfc")),
            ("★ 正式 A+B pair", self._run_pair),
            ("停止", self._stop),
            ("查看最近正式结果", self._open_latest_pair),
            ("打开 outputs", lambda: self._open_dir(config.OUTPUT_ROOT)),
        ]
        for text, cmd in btns:
            b = ttk.Button(row1, text=text, command=cmd)
            b.pack(side="left", padx=3)
            self._bt[text] = b
        self._status = ttk.Label(row1, text="空闲", foreground="#0a0")
        self._status.pack(side="right")

    # ------------------------------------------------------------ 日志区
    def _build_log(self):
        import tkinter as tk
        box = ttk.Frame(self.root)
        box.pack(side="bottom", fill="both", expand=False, padx=6, pady=4)
        self._log = tk.Text(box, height=13, state="disabled", wrap="word",
                            font=("Consolas", 9))
        sb = ttk.Scrollbar(box, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ------------------------------------------------------------ 参数装卸
    def _load_defaults(self):
        self._apply_params(config.parameter_defaults())
        self._log_write("已载入默认参数（SFC 只读=论文整定默认）。\n")

    def _load_dialog(self):
        p = filedialog.askopenfilename(initialdir=PROJ, filetypes=[("JSON", "*.json")])
        if p:
            self._apply_params(config.load_parameters(Path(p)))
            self._log_write(f"已载入 {Path(p).name}\n")

    def _save_dialog(self):
        params, errs = self._collect()
        if errs:
            messagebox.showerror("参数错误", "\n".join(errs))
            return
        p = filedialog.asksaveasfilename(initialdir=PROJ / "outputs",
                                         defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            config.save_parameters(params, Path(p))
            self._log_write(f"已保存 {Path(p)}\n")

    def _apply_params(self, params: dict):
        for key, var in self.vars.items():
            var.set(str(params.get(key, "")))

    def _collect(self) -> tuple[dict, list]:
        """从控件取参并做数值解析/校验。返回 (params, errs)。"""
        params = config.parameter_defaults()
        for group in config.PARAM_GROUPS:
            for field in group["fields"]:
                key = field["key"]
                if key not in self.vars:
                    continue
                raw = self.vars[key].get().strip()
                if field["type"] in ("float", "int") and raw == "":
                    continue
                params[key] = raw
        parsed: dict = {}
        for key, val in params.items():
            f = config.find_field(key)
            if f and f["type"] in ("float", "int"):
                try:
                    parsed[key] = float(val) if f["type"] == "float" else int(float(val))
                except (TypeError, ValueError):
                    parsed[key] = val
            else:
                parsed[key] = val
        errs = config.validate_params(parsed)
        return parsed, errs

    def _save_collected(self) -> dict | None:
        params, errs = self._collect()
        if errs:
            messagebox.showerror("参数错误", "\n".join(errs))
            return None
        LAST_PARAMS.parent.mkdir(parents=True, exist_ok=True)
        config.save_parameters(params, LAST_PARAMS)
        return params

    # ------------------------------------------------------------ 运行调度
    def _set_busy(self, busy: bool):
        """
        busy 进入时快照每个参数控件（含可视化/高级勾选框）的原始 state；退出时按原值恢复。

        修复两类历史 bug：1) busy 期可编辑 Entry 被 disabled，退出时因“当前已是 disabled”而
        不再恢复成 normal；2) readonly Combobox（只读但可下拉）在退出时被错判成 normal。
        现在一律按进入前的快照回写，不猜。
        """
        self._busy = busy
        for text, b in self._bt.items():
            if text == "停止":
                b.configure(state="normal" if busy else "disabled")
            else:
                b.configure(state="disabled" if busy else "normal")
        self._status.configure(text="运行中…" if busy else "空闲",
                               foreground="#c60" if busy else "#0a0")

        guarded = list(self._param_widgets.values())
        if self._cb_show is not None:
            guarded.append(self._cb_show)
        if self._cb_adv is not None:
            guarded.append(self._cb_adv)
        if busy:
            if self._saved_states is None:
                self._saved_states = []
                for w in guarded:
                    try:
                        self._saved_states.append((w, str(w.cget("state"))))
                    except Exception:
                        pass
            for w, _orig in (self._saved_states or []):
                try:
                    w.configure(state="disabled")
                except Exception:
                    pass
            self._stop_requested_busy = False
        else:
            saved = self._saved_states
            self._saved_states = None
            if saved:
                for w, orig in saved:
                    try:
                        w.configure(state=orig)   # 原样恢复：normal/disabled/readonly 各归各位
                    except Exception:
                        pass
            # 无快照的兜底只恢复按钮（_spawn 恒先 busy=True，正常不至此）

    def _run_interpreter(self) -> str | None:
        """返回可跑仿真的解释器；找不到时弹窗说明并返回 None。"""
        py = _resolve_run_python()
        if py is not None:
            return py
        messagebox.showerror(
            "找不到可跑仿真的 Python",
            "launcher 探测不到能 import mujoco+numpy 的解释器。\n\n"
            f"已试：\n  {RUN_PY_DEFAULT}\n  {sys.executable}\n"
            "解决办法：设环境变量 APFSFC_PYTHON 指向带 mujoco 的 python.exe\n"
            "（例如 venv\\Scripts\\python.exe），或用 start_ui.bat 启动本 UI。")
        return None

    def _viz_probe_or_abort(self, py: str) -> bool:
        """勾选了可视化：先探测观赛链组件，失败则日志+弹窗并中止启动（不静默无头）。"""
        if not self.var_show.get():
            return True
        self._log_write(f"[launcher] 校验观赛组件（mujoco.viewer / tkinter / TkAgg）于 {py} …\n")
        ok, why = _probe_viz(py)
        if ok:
            self._log_write("[launcher] 观赛组件就绪。\n")
            return True
        self._log_write("[launcher] 观赛组件探测失败：\n  " + why + "\n")
        messagebox.showerror(
            "可视化环境未就绪",
            "已勾选“运行可视化”，但子进程解释器无法导入观赛组件（mujoco.viewer / "
            "tkinter / matplotlib TkAgg）。\n\n失败：\n  " + why + "\n\n"
            "严格 --show 要求 MuJoCo 3D 与实时 e_y 图都就绪，任一失败会在 t=0 前报错退出"
            "（不是自动转无头）。本次不启动；请修复后重试。")
        return False

    def _run_trial(self, mode: str):
        params = self._save_collected()
        if params is None:
            return
        py = self._run_interpreter()
        if py is None:
            return
        if not self._viz_probe_or_abort(py):
            return
        self._log_write(f"[launcher] 使用解释器：{py}\n")
        self._unlink_stop()
        cmd = [py, str(PROJ / "run.py"),
               "--params", str(LAST_PARAMS), "--mode", mode,
               "--stop-path", str(UI_STOP)]
        if self.var_show.get():
            cmd.append("--show")
        self._spawn(cmd, f"[launcher] 试跑 {mode}（输出 outputs/run_<时间>_{mode}）\n")

    def _run_pair(self, _arg=None):
        params = self._save_collected()
        if params is None:
            return
        py = self._run_interpreter()
        if py is None:
            return
        if not self._viz_probe_or_abort(py):
            return
        self._log_write(f"[launcher] 使用解释器：{py}\n")
        self._unlink_stop()
        cmd = [py, str(PROJ / "pair.py"),
               "--params", str(LAST_PARAMS),
               "--stop-path", str(UI_STOP)]
        if self.var_show.get():
            cmd.append("--show")
        self._spawn(cmd, "[launcher] 正式 A+B：同一 params 快照 → 唯一 pair 目录\n")

    def _spawn(self, cmd: list[str], note: str):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        self._log_write("\n" + note + "$ " + " ".join(cmd) + "\n")
        self._last_rc = 0
        self._stop_requested_busy = False
        self._set_busy(True)
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True,
                                          encoding="utf-8", env=env)
        except Exception as exc:
            self._log_write(f"[launcher] 启动失败：{exc}\n")
            self._set_busy(False)
            return
        _StreamReader(self._proc.stdout, self._box_q).start()
        threading.Thread(target=self._wait_proc, args=(self._proc,), daemon=True).start()

    def _wait_proc(self, proc: subprocess.Popen):
        proc.wait()
        self._last_rc = proc.returncode
        self._proc = None
        self._box_q.put(_BUSY_END)

    def _stop(self):
        self._stop_requested_busy = True
        try:
            UI_STOP.write_text("stop", encoding="utf-8")
            self._log_write("[launcher] 已写停止信号，等待当前 run/pair 优雅结束…\n")
        except OSError as exc:
            self._log_write(f"[launcher] 写停止信号失败：{exc}\n")

    @staticmethod
    def _unlink_stop():
        try:
            UI_STOP.unlink(missing_ok=True)
        except OSError:
            pass

    def _open_latest_pair(self, *_):
        latest = self._latest_pair()
        if latest is None:
            self._log_write("[launcher] 没找到 provisional=false 且含 analysis 的正式 pair。\n")
            return
        self._log_write(f"[launcher] 打开 {latest}\n")
        self._open_dir(latest)

    def _latest_pair(self) -> Path | None:
        hits = []
        for m in sorted(config.OUTPUT_ROOT.glob("pair_*/manifest.json"), reverse=True):
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("provisional"):
                continue
            an = d.get("analysis") or {}
            if not an.get("done"):
                continue
            p = Path(an.get("dir") or m.parent / "analysis")
            if p.joinpath("analysis_metrics.json").is_file():
                hits.append(m.parent)
        return hits[0] if hits else None

    # ------------------------------------------------------------ 日志泵
    def _poll_queue(self):
        while True:
            try:
                line = self._box_q.get_nowait()
            except queue.Empty:
                break
            if line == _BUSY_END:
                rc = self._last_rc
                self._set_busy(False)
                self._log_write(f"[launcher] 子进程结束（返回码 {rc}）。\n")
                if rc != 0 and self.var_show.get() and not self._stop_requested_busy:
                    self._log_write(
                        "[launcher] 已勾选可视化且子进程非零退出：严格 --show 下任一窗口未就绪"
                        "会在 t=0 前报错（不是自动转无头）。请查上方 [liveview]/[run] 行。\n")
                    messagebox.showwarning(
                        "观赛启动未就绪（子进程非零退出）",
                        f"子进程以返回码 {rc} 结束，且本次未请求停止。\n\n"
                        "你勾选了“运行可视化”（严格 --show）：要求 MuJoCo 3D 与实时 e_y 图都"
                        "就绪，任一失败会在 t=0 前报错退出。\n\n"
                        "日志中 [run]/[liveview] 的报错行会指出具体失败组件与原因；"
                        "若确要“开不出窗也能跑”，请命令行改用 --show-best-effort。")
            else:
                self._log_insert(line)
        self.root.after(120, self._poll_queue)

    def _log_write(self, text: str):
        self._log_insert(text)

    def _log_insert(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    @staticmethod
    def _open_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            pass


def main():
    _utf8_stdio()
    import tkinter as tk
    from tkinter import ttk as _  # noqa: F401
    root = tk.Tk()
    root.geometry("1020x860")
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
