# CHANGELOG

## V1.3 — 控制链改为“视觉加速度 → 虚拟力 → SFC”（本节为最新）

取代 V1.2 的 `F_apf = −k_a·e_y` 输入链。**V1.2 各节原样保留在下方，仅作历史记录。**

### 新控制链（唯一正式链路）

    e_y 历史 → a_y_est → F_vir = +force_map_mass_kg·a_y_est
             → m·v̇_s + μ|v_s|^(n−1)·v_s = F_vir
             → v_s += a_s·dt_actual → v_sfc_out = g·v_s
             → 运行层 y_sfc_offset += v_sfc_out·dt_actual
             → y_cmd = y_ref + y_sfc_offset
             → 既有绝对位姿 DLS-IK → MuJoCo 关节位置伺服（产生关节力矩）

- **正号硬约束**：`F_vir = +force_map_mass_kg·a_y_est`；禁止 `−force_map_mass_kg·a_y_est`。
  SFC 输出端**不加**固定负号——剪切增稠阻力已在方程内部的 `−μ|v_s|^(n−1)·v_s` 里。
- **两类力分离**：`w(t)`（`w_force.csv`，`xfrc_applied` 于 `wrist_3_link`）是**真实物理扰动**；
  `F_vir` 是控制器由视觉等效测量构造的**虚拟力输入**。`ctrl.step()` 绝不收 `w.w(t)`，
  控制器也绝不读 `w_force` 作为 SFC 输入。
- **位置伺服链是接口适配**：`y_sfc_offset` 的运行层积分只是把 SFC 输出速度接到现有 MuJoCo
  位置伺服上；SFC 核心只输出 `v_sfc_out=g·v_s`(m/s)。本工程不做直接力矩控制、不改 actuator
  XML，不表述为“SFC 直接输出关节力矩”。

### 逐文件

- `apf_sfc.py`（重写）：
  - 新增 `CausalAccelForceMapper(force_map_mass_kg, accel_window_points)`：保存最近 N 个
    **真实时间戳**与 e_y，以 `τ_i = t_i − t_current` 对 `e_y(τ)=c0+c1τ+c2τ²` 做尾部窗口
    最小二乘，`a_y_est = 2·c2`；样本不足 `mapper_ready=False, a_y_est=0, F_vir=0`
    （杜绝启动瞬间不完整差分产生巨大虚拟力）；窗口必须为 ≥3 的奇数（3/5/7/9）。
    只用真实时间戳，不做定点 5–40 Hz 带通，也不对原始位置直接二阶差分。
  - `ApfSfc(m, μ, n, g)`：删除 `k_a`、`B0(b_eps)`、`K_v`；`step(dt_actual, F_vir) -> v_sfc_out(m/s)`
    内部严格为 `shear = μ·sign(v_s)·|v_s|^n`、`a_s = (F_vir − shear)/m`、
    `v_s = v_s + a_s·dt_actual`、`v_sfc_out = g·v_s`；累计位移**移出**本类（由 run.py 积分）。
    `logs()` 至少给出 `sfc_a_internal/sfc_v_internal/sfc_v_out/sfc_shear_force/F_vir`。
  - `sfc_dt_max(m, μ, n, f_abs_max_N)`：输入改为**最大虚拟力**（N），不再由 `k_a·e_max` 推。
- `config.py`：删除 `k_a` / `sfc_B0` / `sfc_K_v`；新增参数组「视觉加速度—虚拟力映射」含
  `force_map_mass_kg`（默认 1.0 kg，help 注明“归一化初值，需依据新虚拟力幅值重新整定；与
  `sfc_m` 不是同一个变量”）与 `accel_window_points`（默认 5，presets 3/5/7/9，校验 ≥3 奇数）。
  SFC 组 help 改为新方程与 `v_sfc_out=g·v_s`。`sfc_mu` 上限 1e8→1e12（V1.3 整定 μ 量级变大），
  默认值同步为随包整定值。版本串 `v1.3`。
- `run.py`：每个控制节拍先 `e_y = y_act − y_ref`，**A/B 都调用**映射器（记录同口径诊断）；
  A 组 `v_sfc_out=0, y_sfc_offset=0`（估计量**不进入**机器人控制）；B 组
  `v_sfc_out = ctrl.step(dt_actual, F_vir)`、`y_sfc_offset += v_sfc_out·dt_actual`、
  `y_cmd = y_ref + y_sfc_offset`。`dt_max` 由整定文件的 `f_max_N` 复算。列名
  `F_apf→F_vir`、`dy→y_sfc_offset`，新增 `e_y/a_y_est/mapper_ready` 与 `sfc_a_internal`。
  指纹 `version=v1.3`，新增 `force_mapping`（method/force_map_mass_kg/accel_window_points/
  f_vir_expected_max_N）、`control_chain`、`position_servo_adaptation`。
  `_packet_e_max` → `_packet_f_max`（读 `sfc_tuning.json` 的 `f_max_N`，旧格式显式报错）。
- `recorder.py` / `analysis.py`：列名同步为 20 列（含 `e_y/a_y_est/mapper_ready/F_vir/
  y_sfc_offset/sfc_a_internal`，删除 `F_apf/dy`）；摘要新增 3 组诊断（a_y_est 峰+RMS、
  F_vir 峰+RMS、B 组 sfc_v_out 峰+RMS）。既有指标不变：A/B 仍以 `e_y=y_act−y_ref` 计，
  频带仍来自冻结 `spectral_profile.json`，时域 RMS/ptp/慢成分/振动带 RMS/门控不动。
- `sfc_tune.py`（重写整定口径）：读 e_des 真实 t 列 + 位移列（µm→m），用**与正式控制相同**的
  因果二次拟合生成 `a_y_est` 序列，跳过未 ready 项，`F_vir_series = force_map_mass_kg·a_y_est`，
  `f_ease=P50`、`f_interf=P99`、`f_max=max`，后续沿用论文 Algorithm 1 求 n/μ/g。
  新增 `REQUIRED_TUNING_KEYS` + `load_tuning()`：**旧格式（含 `k_a_N_per_m` 或缺新字段）
  显式抛错**并提示重跑，不静默读取。
- `fit_w.py`：`_write_replay_params` 经 `load_tuning` 读 `force_map_mass_kg`、
  `accel_window_points`、`sfc_m/mu/n/g`，不再写 `k_a/sfc_B0/sfc_K_v`，旧格式不再被
  `try/except: pass` 静默吞掉。
- `launcher_ui.py`：仅同步文档串（SFC 只读字段为 m/mu/n/g）；UI 仍由 `config.PARAM_GROUPS` 驱动。
- 测试/文档：`tests/test_v12.py` → `tests/test_v13.py`（新增映射器符号/恒零/窗口合法性与
  旧格式拒绝用例）；`tests/acceptance.py` 改查 `force_map_mass_kg`/`accel_window_points` 与
  新 `method`，删 k_a/B0/K_v 检查；README 控制链、整定来源、观赛/结果口径同步 V1.3。

### 冻结与未改动项

- `w_force.csv`/`schedule.csv`/`e_des_um.csv` 字节未变（`tests/acceptance.py` 复核 SHA）；
  仅刷新 `sfc_tuning.json`（V1.3 格式）与 `replay_params.json`（新字段）。
- 未改：MuJoCo XML / UR10e 模型 / 真实扰动反演算法（`wfit.py` 的 plant inversion）/
  实时观赛隔离机制（liveview 克隆 observer）/ A/B 实验组织方式（pair 的 A→B 同参快照）/
  `disturbance.py` / `make_profile.py` / `spectral.py` / `kinematics.py` / `simenv.py`。

### 随包 P05R01 的 V1.3 整定结果（`accel_window_points=9, velocity_ratio=1.5, force_map_mass_kg=1.0`）

`f_ease=0.0610958 N, f_interf=0.403809 N, f_max=0.837816 N, a_rms=0.12905 m/s²,
a_peak=0.837816 m/s², n=4.6576149, μ=253358215.174, g=0.01561887, dt_max=0.0077432 s`。
本 e_des 上窗口 5/7 分别得 n=11.23/5.97（>5，超正式范围，`sfc_tune.py` 会报错），故取 9。

---

## V1.2 最小可信版本（历史）

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
