# CHANGELOG — V1.2 最小可信版本

对照《DS 修改指南》§1–§10 与《代码审查报告》逐项落地。每条给出落点文件与验收线索。

## 数据冻结与去“合成”

- R-001…R-003 / 指南 §1「合成信号」：`config.py` 删除 `disturbance_mode / w_amp_n / w_seed /
  sfc_M_v`；`disturbance.py` 删除 synthetic/off/scale/fade，只留 `load_fixed`（严格递增、
  有限、覆盖 ≥ schedule 时长、不足抛 `DisturbanceError`）。
- `fit_w.py finalize_packet` 保证冻结文件 `w_force/schedule/e_des_um` 只写一次、只算哈希
  不碰字节；meta/params/summary 可随时安全刷新。
- replay_params 改用**项目相对路径**（R-011），不再存 C:/D: 绝对路径。

## SFC 论文一致化（E-002/003/005 + §4.2）

- `apf_sfc.py`：`m·v̇+μ|v|^(n-1)v=F_apf`，`v_out=g·v`，`dy+=v_out·dt_actual`；
  `B0(b_eps)/K_v` 作为非论文扩展保留但**正式=0**；`is_formal` 要求 1<n≤5 且 B0=Kv=0。
- `sfc_tune.py`：Algorithm 1 纯函数 `tuning_from_e_des`，输出整条 trace + `dt_max_s`；
  `sfc_tuning.json` 记录 e_des SHA（audit）。
- E-009 已核：`n=2.8279765, μ=76150.4348, g=0.0241538` 与报告逐位一致。

## 离散与时间戳（R-008 / §3 时序）

- `run.py` 控制器用真实 `dt_actual=t_k−t_{k-1}`；recorder 记 `dt_actual` 列。
- 复算论文离散上界 `sfc_dt_max`（≈0.0391 s，欧拉对 v_ss 线性化阻尼的稳定界，非硬编码），
  dt_actual 超限即中止。132 Hz(≈7.6ms) 与 1000 Hz(1ms) 均低于上界，均可正式跑。
- `spectral.py` 均匀重采样按**平均步长**（不做 median→125Hz 误判），1000 Hz 已均匀则 no-op。

## 谱 profile（§6 / R-014 / R-015 / R-017）

- `make_profile.py` 读原始 CSV（未当 1 kHz）→ `spectral_profile.json`：慢成分区间、振动带、
  慢瓣主峰、振动主导峰（单段交叉核对）。+6dB/1 分辨率合并判据全记录在 json。
- `analysis.py` 一律用冻结 profile 频带；**慢主导频率与振动主导频率分列**；A/B 门控
  （同 w/SHA、模型指纹、INIT_Q、物理/控制/时长、params 指纹、模式 baseline/apf_sfc），
  不满足即报错；profile/w/指纹 SHA 进结果。

## 公平性与溯源（§7 / R-011 / R-017）

- `run.py` 每次落 `run_fingerprint.json`（version v1.2、params 指纹、disturbance/schedule SHA、
  model 指纹、INIT_Q、物理/控制/时长、sfc_paper_consistent、dt_max 复算、completed/reason）。
- `pair.py`：正式 A/B 用同一 params 快照 → **唯一** `outputs/pair_*/`，manifest `provisional`
  至全部完成+门控通过+分析写完才转 false；绝不覆盖旧结果。
- `launcher_ui.py`：SFC 参数 readonly（来源 sfc_tuning），试跑/正式 pair/停止/查最近正式结果。

## 其它审查项

- R-005/E-010：SFC 全链路 finite 检查，异常即 abort 标 failed。
- R-018：IK 关节目标贴限位 → `ik_saturated`，run 不算成功。
- R-021：`run_fingerprint` 记录控制/物理频率与真实 dt 统计，分析按真实时间戳重采样。
- 历史遗留：删除 V1.1 合成扰动 demo 输出目录；`README.md` 同步 V1.2。

## 运行时观赛链改造（DS《运行时可视化》任务书版，对应 T-VIS-01~07）

- **撤回旧声明**：不再把“`--show` 与无 `--show` 的 trajectory.csv 逐字节一致、run_fingerprint
  完全相同（不新增字段）”写成既成事实。改为 T-VIS-02 数值列复核口径：同参数 headless 与开窗
  各跑一次，比较 trajectory.csv 数值列逐元素 + 两份 `execution_integrity`。原隐患
  （`viewer.sync()` 把 GUI 操作同步回共享 model/data）由下述克隆 observer 隔离解决。
- `liveview.py`（重写，不再持有主仿真 model/data）：
  - P0 pacing 零点：`begin()` 先克隆 + 开 3D + 显式开图（`plt.ion()/show(block=False)`/
    draw/flush，不用 `plt.pause()`），warm 后**最后**才记 `_t0_wall` → t=0 无开局快进。
  - P0 只读隔离：自持从同一 XML 重载的 clone MjModel/MjData，只把 clone 交给
    `launch_passive`；主仿真 model/data 从不出现在 viewer → GUI 操作不写回正式状态。
    `launch_passive` 放包装线程 + 队列 + 超时防 GLFW 失败挂死。
  - strict 默认：3D 与实时图任一未就绪 → `LiveViewError`（t=0 前抛、Recorder 未建=零落盘）；
    另设 `--show-best-effort` 才降级无头。所有 GUI 异常记 (component, kind, message)。
  - 图仅一条 `e_y(µm)`；横轴前 10s 固定 `[0,10]`、之后滚动最近 10s；纵轴 0-对称固定
    `y_limit_um` 越界只扩不缩；A/B 同值同窗（`ceil(1.2×max|e_des_um|)`，包缺则默认+记来源）。
  - 图刷新自适应：目标 ~15fps，实测整帧重绘耗时 > 刷新间隔/3 时自动降频（慢画布/远程桌面）。
    修复点：原固定 15fps 每帧重绘在慢 Tk 环境可达 ~60ms/次，把主循环拖慢到「显示反而饿死
    计算」；自适应后绘制只占小部分墙钟，物理/控制节拍不再被图拖累。
- `run.py`：`--show` 与 `--show-best-effort` 互斥；`--pace` 必须 >0（≤0 argparse 报错）。
  观赛 LiveView 移到 Recorder 创建之前（strict 失败零落盘）。新增 execution_integrity 自动检查
  （physics_steps==ceil((duration−1e-9)·hz)、ticks==rows、show 时 pushed==ticks、t 严格递增、
  dt≤1/ctrl+1/phys+1e-9）与 visualization 诊断（viewer_syncs/chart_redraws/samples_received/
  窗口状态/errors），写进 run_fingerprint.json 两个新顶层键；二者只读，不进 A/B 门控。
- `pair.py`：A/B 用同一 window_s/y_limit_um，标题 `A baseline`/`B APF-SFC`，A 结束、B 开窗前
  段间提示；默认无头不变；同款互斥组 + `--pace>0` 校验。
- `launcher_ui.py`：勾“运行可视化”时对已解析解释器先探测
  `import mujoco, mujoco.viewer, numpy, tkinter; import matplotlib.backends.backend_tkagg`，
  失败明确弹窗指明组件并中止（不再静默无头）；`_set_busy` 改为 busy 进入时**快照控件原始 state**、
  退出按原值恢复（修两类 bug：可编辑 Entry busy 期变 disabled 后不再恢复、readonly Combobox 被
  误恢复 normal）；子进程非零退出且已勾可视化且未请求停止 → popup 提示查 [liveview]/[run] 报错行。
- 验收：`tests/test_viz.py`（T-VIS-02/05 自动化 + `GUI_FORCED_OFF` strict/best-effort +
  克隆内存隔离断言；全无头可跑）。真窗项 T-VIS-01/03/04/06 与部分 07 归交互冒烟/人工，
  以交付汇总记录。

## 运行解释器自愈（用户上报）

- 现象：即使 start_ui.bat（venv pythonw）启动，run/pair 子进程仍可能落到无 mujoco 的裸解释器
  （如 C:\Python314），`import mujoco` 崩溃。
- 修法（`launcher_ui.py`）：不再用 `sys.executable` 硬起子进程，改为 `_resolve_run_python()`
  探测“能 import mujoco+numpy”的解释器——候选依次为 APFSFC_PYTHON → 当前解释器 →
  同级 pythonw/python → 默认 venv（pythonw 优先防黑窗），找到即缓存。
  找不到时弹窗给出 APFSFC_PYTHON 指引，不再让 ModuleNotFoundError 半路崩掉 run。
- 已自测两场景：UI 在 venv 内 → 用自身；UI 模拟在 C:\Python314 → 自动回落默认 venv。

## 停止按钮修复（用户上报）

- 现象：点“停止”不停止反而报 `TypeError: Launcher._stop() takes 1 positional argument but 2
  were given`（Tk 回调内异常），`outputs/ui_stop.request` 从未写入 → run/pair 继续跑。
- 根因：`_build_actions` 曾用 `command=(lambda a=arg, c=cmd: c(a))` 给每个按钮统一“塞一个参数”，
  而 `_stop(self)` 不接受参数；同构的 `打开 outputs` 零参 lambda 也被多塞 `None`，同属潜在 bug。
- 修法（`launcher_ui.py`）：按钮一律改绑**真正的零参闭包**——试跑/打开 outputs 用
  `lambda: self._run_xxx(...)` 绑参，停止/正式 pair/查结果直接绑零参方法，符合 Tk
  `command=` 无参契约。点“停止”→ 写 `ui_stop.request`，run/pair 下次 500 步轮询即优雅结束。
- 已自测：`_run_pair/_stop/_open_latest_pair` 均可被零参调用；`_stop()` 写入信号文件成功；
  模块全量 `py_compile` 通过。**需重启 launcher 生效**（旧的报错窗口跑的是修复前代码）。

## 已知边界（如实记录）

- 在刚性 menagerie UR10e（强伺服）上，SFC 经参考偏移生效：主要压慢/低频
  （时域 RMS −29%、慢成分 −28%），冻结振动带 RMS 基本不变。结论只适用于本定性模型。
- `outputs/plant_cache.npz` 无 `fp` 指纹 → 首次跑 `fit_w`/`measure_plant` 会自动重标定（~60 s）。
