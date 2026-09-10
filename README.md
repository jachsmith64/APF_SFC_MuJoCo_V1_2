# APF-SFC MuJoCo 抑振仿真（V1.3 最小可信版本）

在 MuJoCo 里回放真实预实验等效外扰 `w(t)`（文件唯一来源），对 **A=基线 / B=APF+SFC**
做**公平 A/B**：同一 params 快照、同一 `INIT_Q`、同一冻结 `w(t)`/`schedule.csv`（逐字节同
SHA-256），唯一差别是是否实例化 `ApfSfc`。评价频带来自预实验冻结的
`spectral_profile.json`，B 不参与选带。

实现依据：`_DS修改指南.md` §1–§10 与 `_代码审查报告.md` R-001…R-021、E-001…E-010；
逐条对账见 `CHANGELOG.md`。正式跑的 A/B 门控不过不进入分析（`pair.py` + `run_fingerprint.json`）。

## 控制链与模型（V1.3：视觉加速度 → 虚拟力 → SFC）

    e_y     = y_act − y_ref                          （y_act 来自 MuJoCo，m）
    e_y 历史 → a_y_est                               （因果二次最小二乘，真实时间戳）
    F_vir   = +force_map_mass_kg · a_y_est           （正号！N）
    m·v̇_s + μ|v_s|^(n−1)·v_s = F_vir                 （SFC 论文剪切增稠虚拟 ODE）
    v_s     = v_s + a_s·dt_actual
    v_sfc_out = g·v_s                                （m/s，SFC 核心的最终输出）
    y_sfc_offset += v_sfc_out·dt_actual              （运行层积分，非 SFC 核心）
    y_cmd   = y_ref + y_sfc_offset
            → p_cmd=[x_ref,y_cmd,z_ref],R=R₀ → DLS-IK → MuJoCo 关节位置伺服（产生关节力矩）

**符号约定（必须遵守）**：`F_vir = +force_map_mass_kg·a_y_est` 是**正号**——正向加速度给正向
虚拟力；禁止写成 `−force_map_mass_kg·a_y_est`。SFC 输出端**不加**人为固定负号，剪切增稠阻力
已经由方程内部的 `−μ|v_s|^(n−1)·v_s` 体现。`force_map_mass_kg`（kg）与 SFC 内部虚拟质量
`sfc_m` **不是同一个变量**。

**位置伺服链是接口适配**：`y_sfc_offset` 的运行层积分是把“SFC 输出的速度”接到本工程已有
MuJoCo 关节位置伺服上的适配层；SFC 核心只输出 `v_sfc_out = g·v_s`（m/s）。本工程不使用直接
力矩控制，也不修改 actuator XML——关节力矩由 MuJoCo 位置伺服（kp=5000/kv=500）内部产生，
**不能表述为“SFC 直接输出关节力矩”**。

**两类力严格分离（不可混用）**：
- `w(t)`（`w_force.csv`）＝施加在 MuJoCo body 上的**真实物理扰动**（`xfrc_applied` 于
  `wrist_3_link`），A/B 逐字节同文件；
- `F_vir` ＝控制器由**视觉等效测量**构造出的**虚拟力输入**（`run.py` 绝不把 `w.w(t)` 传给
  `ctrl.step()`，控制器也绝不读取 `w_force` 作为 SFC 输入）。

- 等效 Y 外扰 `w(t)` 作为世界系 Y 力加在 `wrist_3_link`（`xfrc_applied`），与 SFC 解耦。
- 模型 `assets_ur10e/` = mujoco_menagerie UR10e，**加载即用不改 XML**（自带关节位置伺服
  kp=5000/kv=500，见 `SERVO_NOTE`，哈希溯源进指纹）。
- 控制器用**真实 dt_actual**（132 Hz 的 7/8 ms 交替如实记录）；SFC 离散上界 dt_max 启动时
  复算，超限即中止。

## 目录

| 文件 | 作用 |
|---|---|
| `config.py` | 参数 schema/校验/路径解析（SFC 论文参数 readonly；扰动仅文件来源） |
| `provenance.py` | SHA-256 / JSON 指纹 / 模型指纹（XML+网格） |
| `simenv.py` | MuJoCo 加载/复位/物理步/外力/位姿/Jacobian |
| `trajectory.py` | 名义路径（回放 `schedule.csv` 或匀速直线） |
| `apf_sfc.py` | `CausalAccelForceMapper`（e_y 历史 → a_y_est → F_vir=+force_map_mass_kg·a_y_est）+ SFC 论文核心 `ApfSfc(m,μ,n,g).step(dt_actual,F_vir)→v_sfc_out` + `sfc_dt_max(m,μ,n,f_abs_max_N)` |
| `kinematics.py` | DLS IK（含姿态） |
| `disturbance.py` | 冻结 `w(t)` 加载校验（严格递增/覆盖/有限/SHA） |
| `recorder.py` | 控制节拍落盘 trajectory.csv + params.json + run_fingerprint.json |
| `run.py` | 无头单组 CLI（`--mode baseline|apf_sfc`） |
| `analysis.py` | V1.3 离线分析（冻结 profile 频带 + 慢/振动分离 + 分段 + A/B 门控 + 4 图 + 控制链诊断：a_y_est/F_vir/sfc_v_out 峰与 RMS） |
| `pair.py` | 正式 A+B：唯一 pair 目录 + manifest + 门控 + 自动分析 |
| `launcher_ui.py` | Tk 参数窗（SFC 只读；试跑 / ★正式 A+B / 停止 / 查最近结果） |
| `liveview.py` | 运行时观赛：克隆 observer 的 MuJoCo 3D 窗口 + 实时 e_y(µm) 图 + pacing（strict/best-effort；同步/重绘计数进指纹） |
| `sfc_tune.py` | V1.3 整定：e_des → 同款因果二次拟合 → |F_vir| 分位(P50/P99/max) + 论文 Algorithm 1 → `sfc_tuning.json`；旧格式用 `load_tuning()` 显式拒绝 |
| `make_profile.py` | 预实验原始数据 → 冻结 `spectral_profile.json`（慢/振动带） |
| `wfit.py` / `fit_w.py` | 真实预实验 → 冻结 w/e/s + `finalize_packet` 溯源 |
| `spectral.py` | 均匀重采样 / Welch / 频带 RMS（公共口径） |
| `tests/` | 自检 + 数据包验收 |

## 快速开始

统一用带 mujoco 的 venv：`C:\Users\PC\Desktop\code\code\Myproject-1\venv\Scripts\python.exe`

```bat
:: ① 图形界面（推荐）
start_ui.bat
::    → “载入配置…” 选 outputs\wfit_P05R01\replay_params.json
::    → “★ 正式 A+B pair” → 自动跑成唯一 outputs\pair_*/（run_A+run_B+analysis+manifest）

:: ② 无头正式 A/B（等价于 UI 那颗按钮）
<venv>\python.exe pair.py --params outputs\wfit_P05R01\replay_params.json
<venv>\python.exe pair.py --params outputs\wfit_P05R01\replay_params.json --control-hz 132

:: ③ 只看某组（诊断）
<venv>\python.exe run.py --params outputs\wfit_P05R01\replay_params.json --mode baseline

:: ④ 自检 / 数据包验收
<venv>\python.exe tests\test_v13.py      :: 纯配置/纯函数层（含映射器/SFC 符号与恒零自检）
<venv>\python.exe tests\test_viz.py      :: 观赛链无头自动化（T-VIS-02/05 + 克隆隔离 + 强制无头）
<venv>\python.exe tests\acceptance.py    :: 数据包/指纹验收

:: ⑤ 观赛（克隆 observer：MuJoCo 3D + 实时 e_y(µm) 图，真实时 pacing；正式 pair 默认无头）
<venv>\python.exe run.py --params outputs\wfit_P05R01\replay_params.json --mode apf_sfc --show
<venv>\python.exe run.py --params outputs\wfit_P05R01\replay_params.json --mode apf_sfc --show-best-effort
<venv>\python.exe pair.py --params outputs\wfit_P05R01\replay_params.json --show
::    严格 --show：3D 与 e_y 图都必须就绪，任一失败在 t=0 前报错退出（零落盘）；
::    确要“开不出窗也能跑”才用 --show-best-effort（降级无头，不报错）。
::    UI 勾“运行可视化”等效严格 --show，启动前先探测 mujoco.viewer/tkinter/TkAgg，
::    失败会明确弹窗说明组件；launcher 自动解析带 mujoco 的解释器（APFSFC_PYTHON 可覆盖）。
```

### 观赛链（DS 任务书版：strict 默认 / 克隆 observer / 计数证据）

- **只读隔离**：`LiveView` 在 `begin()` 从同一 XML 重新加载一份 clone MjModel/MjData，
  只把这份 clone 交给 passive viewer；主仿真（步进+记录）的 model/data 从不进 GUI → 用户
  拖动/暂停/改选项只落在 clone，回不到正式仿真。`tests/test_viz.py` 有“写 clone qpos 后
  主 data 不变”的内存独立性断言。
- **pacing 零点**在 3D 与图窗都就绪并 warm 后才记录——`t=0` 不再开局快进。
- **图只画一条 `e_y(µm)`**：前 10 s 横轴固定 `[0,10]`，之后滚动最近 10 s；纵轴是以 0 为
  对称中心的固定半宽 `y_limit_um`，越界只向外扩不缩回。正式 pair 只从冻结扰动包算一次
  `ceil(1.2×max|e_des_um|)`（缺文件则用默认 200 并记来源），A/B 同值同窗，标题标
  `A baseline` / `B APF-SFC`。图刷新率自适应：目标 ~15 fps，实测整帧重绘耗时超过刷新间隔的
  1/3 时自动降频——慢画布/远程桌面上绘制只占小部分墙钟，主循环不会被图拖到跟不上物理步
  （显示可丢帧，计算不丢步）。
- **计数证据**：每次 run 落 `execution_integrity`（physics_steps/control_ticks/record_rows/
  pushed 与自动核对）+ `visualization`（viewer_syncs/chart_redraws/samples_received/窗口状态/
  errors），两个顶层键只读诊断、不进 A/B 门控。
- **一致性口径**：不把“与无 `--show` 逐字节一致”写成既成事实；改为 T-VIS-02 数值列复核
  （同参数 headless 与开窗各跑一次，比较 trajectory.csv 数值列逐元素 + 两份
  `execution_integrity`）。显示可以丢帧，仿真计算不允许丢步。

正式结果落在 `outputs/pair_YYYYMMDD_HHMMSS/`：

| 产物 | 含义 |
|---|---|
| `run_A/` `run_B/` | 各带 trajectory.csv / params.json / run_fingerprint.json / run_summary.txt |
| `analysis/` | analysis_metrics.json + analysis_summary.txt + figures/(fig1…fig4) |
| `manifest.json` | `provisional=false` 才算正式；记录 params 指纹、w/schedule/SHA、模型指纹、门控、analysis 哈希 |

## 冻结频带怎么来（为什么不再写死 1–45 Hz）

`make_profile.py` 只读预实验原始 CSV（真实时间戳，未当 1 kHz），把 `pre_motion+post_stop`
当静态噪声、`X_outbound+X_return` 当运动；两者共用窗做 Welch 后，取“运动谱高于静态谱
+6 dB、间隙 < 1 分辨率自动合并、慢瓣与首振动簇之间谷”为判据，输出冻结
`spectral_profile.json`：
慢成分 `[0, slow_top]Hz`、振动带列表、慢瓣主峰、振动主导峰与**单段交叉核对**
（X_outbound/X_return 各自谱都能复现 3.9 Hz 基频 + ~7/12.5 Hz 谐波，证明非拼接假象）。
换数据集必须重跑并重冻结，旧 profile 不复用。

P05R01 冻结结果（`outputs/wfit_P05R01/spectral_profile.json`）：
慢成分 `[0, ~2.5]Hz`（慢瓣主峰 ~0.5 Hz），振动带主簇 `~2.5–14 Hz`
（主导峰 ~3.5 Hz + 谐波），另含数条高频结构模。

## SFC 参数从哪来（为什么只读）

`sfc_tune.py`（V1.3）读随包 `e_des_um.csv` 的**真实 t 列 + 位移列**，用与正式控制**完全相同**的
因果二次拟合（`apf_sfc.CausalAccelForceMapper`）生成 `a_y_est` 序列，跳过未 ready 的前几项，再取

    F_vir_series = force_map_mass_kg · a_y_est_series
    f_ease = P50(|F_vir|)   f_interf = P99(|F_vir|)   f_max = max(|F_vir|)

代入论文 Algorithm 1：`n = ln(f_interf/f_ease)/ln(v_c/v_d)`、`Ψ(n)`、`μ`、`g`。
产物 `sfc_tuning.json`（`method=acceleration_to_virtual_force`）含全部公式中间量与 e_des SHA；
`replay_params.json` 的 `force_map_mass_kg`/`accel_window_points`/`sfc_m,μ,n,g` 取自它。

**旧格式不会被静默读取**：V1.2 的 `sfc_tuning.json`（含 `k_a_N_per_m`，或缺新字段）由
`sfc_tune.load_tuning()` **显式抛错**并提示重新运行 `sfc_tune.py`；`run.py`（B 组）与
`fit_w._write_replay_params` 都走这条检查。

本包 P05R01 整定值（`force_map_mass_kg=1.0 kg, accel_window_points=9, m=1, fc_ease=1 Hz, 压缩比 1.5`）：

| 量 | 值 |
|---|---|
| `f_ease / f_interf / f_max` | 0.06110 / 0.40381 / 0.83782 N |
| `a_est_rms / a_est_peak` | 0.12905 / 0.83782 m/s² |
| `n / μ / g` | 4.6576149 / 2.533582e8 / 0.01561887 |
| `dt_max_s` | 0.0077432 s（>1/132≈7.576 ms，>1/1000 s） |

> 整定口径提醒：`P99/P50(|F_vir|)` 直接决定 `n`。窗口越小，二阶差分越尖，比值越大。
> 本 e_des 上 `accel_window_points=5/7` 得 `n=11.23/5.97`（超出正式范围 1<n≤5，`sfc_tune.py`
> 会明确报错并给出可选的 `velocity_ratio`），`=9` 才落在正式范围内，故本包取 9。
> 换 e_des / 改 `force_map_mass_kg` / 改压缩比都必须重跑 `sfc_tune.py`，不在 UI 手调。

## 结果口径与已知边界

- 指标：`e_y` 用平均步长均匀重采样后做 Welch；**慢成分 RMS**（`slow_component_range`）、
  **各冻结振动带 RMS**（`identified_vibration_bands`）与总振动 RMS、振动主导频与慢主导频**分列**
  （R-014）；另给时域 RMS/峰峰值与稳态往返/瞬态分段。
- 抑制比 = B/A（RMS）。正式 A/B 必须 `gate.ok`；门控项见 `analysis.gate_pair`。
- **V1.2 时代演示结果（控制链已换代，仅存档，不代表 V1.3）**：该组数字来自
  `F_apf=−k_a·e_y` 的旧 A/B（本模型 + 旧整定，1000 Hz 与 132 Hz 几乎一致）：B 主要压低慢/低频
  分量（时域 RMS 27.7→19.6 µm，慢成分 22.2→15.9 µm），**冻结振动带 RMS ~10.5 µm 基本不变**。
  V1.3 换了整条输入链（加速度→虚拟力）与整组 SFC 参数（n 2.83→4.66、μ 7.6e4→2.5e8），
  **必须重跑正式 A+B 才能给出 V1.3 的抑制数字**；在重跑完成前不要引用上面这组旧值。
  原因是这里用刚性 menagerie UR10e（强关节伺服）作定性模型，SFC 输出经参考偏移生效，
  无法直接对 3.9 Hz 本征振动做功；真实软性负载/力控负载上的抑制效果需换更柔性的 plant 验证。
  该结论如实记录，不作为“SFC 无效”的推广。

## 说明与约定

- 坐标系：MuJoCo 世界 x 前/y 左/z 上；home `shoulder_pan=0` 使臂沿 +X 展开；主运动 = X，
  抑振 = Y；TCP = `attachment_site`（m/rad/s/N）。
- `w_force.csv`/`schedule.csv`/`e_des_um.csv` 为冻结文件，只算哈希不改字节（改了破坏发布哈希）。
- 物理默认 1000 Hz；control 1000 / 132 两个预设都可正式跑（dt_actual 如实记录）。
- 依赖见 `requirements.txt`（mujoco==3.8.0 / numpy==2.4.4 / matplotlib；tkinter 为标准库）。
