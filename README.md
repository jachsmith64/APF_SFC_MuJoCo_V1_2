# APF-SFC MuJoCo 抑振仿真（V1.2 最小可信版本）

在 MuJoCo 里回放真实预实验等效外扰 `w(t)`（文件唯一来源），对 **A=基线 / B=APF+SFC**
做**公平 A/B**：同一 params 快照、同一 `INIT_Q`、同一冻结 `w(t)`/`schedule.csv`（逐字节同
SHA-256），唯一差别是是否实例化 `ApfSfc`。评价频带来自预实验冻结的
`spectral_profile.json`，B 不参与选带。

实现依据：`_DS修改指南.md` §1–§10 与 `_代码审查报告.md` R-001…R-021、E-001…E-010；
逐条对账见 `CHANGELOG.md`。正式跑的 A/B 门控不过不进入分析（`pair.py` + `run_fingerprint.json`）。

## 控制链与模型

    e_y    = y_actual − y_ref                       （y_act 来自 MuJoCo，m）
    F_apf  = −k_a·e_y
    m·v̇ + μ|v|^(n−1)·v = F_apf        （SFC 论文剪切增稠虚拟 ODE；B0/K_v 第一轮=0）
    v_out  = g·v ;   dy += v_out·dt_actual
    y_cmd  = y_ref + dy  →  p_cmd=[x_ref,y_cmd,z_ref],R=R₀ → DLS-IK → 关节伺服

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
| `apf_sfc.py` | SFC 论文 ODE 积分 + `sfc_dt_max`；`is_formal`（1<n≤5 且 B0=Kv=0） |
| `kinematics.py` | DLS IK（含姿态） |
| `disturbance.py` | 冻结 `w(t)` 加载校验（严格递增/覆盖/有限/SHA） |
| `recorder.py` | 控制节拍落盘 trajectory.csv + params.json + run_fingerprint.json |
| `run.py` | 无头单组 CLI（`--mode baseline|apf_sfc`） |
| `analysis.py` | V1.2 离线分析（冻结 profile 频带 + 慢/振动分离 + 分段 + A/B 门控 + 4 图） |
| `pair.py` | 正式 A+B：唯一 pair 目录 + manifest + 门控 + 自动分析 |
| `launcher_ui.py` | Tk 参数窗（SFC 只读；试跑 / ★正式 A+B / 停止 / 查最近结果） |
| `liveview.py` | 运行时观赛：克隆 observer 的 MuJoCo 3D 窗口 + 实时 e_y(µm) 图 + pacing（strict/best-effort；同步/重绘计数进指纹） |
| `sfc_tune.py` | SFC 论文 Algorithm 1 整定 → `sfc_tuning.json` |
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
<venv>\python.exe tests\test_v12.py      :: 纯配置/纯函数层
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

`sfc_tune.py` 按论文 Algorithm 1 + 随包 `e_des_um.csv` 的幅值分位统计整定（k_a=1500、
fc_ease=1 Hz、压缩比 1.5），写 `sfc_tuning.json`（含全部公式中间量与 e_des SHA）。config 默认与
`replay_params.json` 取自它：`m=1, n≈2.828, μ≈7.62e4, g≈0.02415, B0=K_v=0`
（`sfc_B0`/`sfc_K_v` 是非论文扩展项，正式为 0）。改 k_a/压缩比必须重跑 sfc_tune，不在 UI 手调。

## 结果口径与已知边界

- 指标：`e_y` 用平均步长均匀重采样后做 Welch；**慢成分 RMS**（`slow_component_range`）、
  **各冻结振动带 RMS**（`identified_vibration_bands`）与总振动 RMS、振动主导频与慢主导频**分列**
  （R-014）；另给时域 RMS/峰峰值与稳态往返/瞬态分段。
- 抑制比 = B/A（RMS）。正式 A/B 必须 `gate.ok`；门控项见 `analysis.gate_pair`。
- **V1.2 演示结果**（本模型 + 本整定，1000 Hz 与 132 Hz 几乎一致）：B 主要压低慢/低频
  分量（时域 RMS 27.7→19.6 µm，慢成分 22.2→15.9 µm），**冻结振动带 RMS ~10.5 µm 基本不变**。
  原因是这里用刚性 menagerie UR10e（强关节伺服）作定性模型，SFC 输出经参考偏移生效，
  无法直接对 3.9 Hz 本征振动做功；真实软性负载/力控负载上的抑制效果需换更柔性的 plant 验证。
  该结论如实记录，不作为“SFC 无效”的推广。

## 说明与约定

- 坐标系：MuJoCo 世界 x 前/y 左/z 上；home `shoulder_pan=0` 使臂沿 +X 展开；主运动 = X，
  抑振 = Y；TCP = `attachment_site`（m/rad/s/N）。
- `w_force.csv`/`schedule.csv`/`e_des_um.csv` 为冻结文件，只算哈希不改字节（改了破坏发布哈希）。
- 物理默认 1000 Hz；control 1000 / 132 两个预设都可正式跑（dt_actual 如实记录）。
- 依赖见 `requirements.txt`（mujoco==3.8.0 / numpy==2.4.4 / matplotlib；tkinter 为标准库）。
