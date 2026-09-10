"""
APF-SFC 机械臂抑振 MuJoCo V1.2：统一配置与参数 schema。

相对 V1.1 的正式改动（对应《DS 修改指南》§3/§4/§5/§7.2）：
- SFC 参数改为论文一致核心：m、μ、n、g；B0/K_v 第一轮正式配置 = 0。
- 扰动只允许“文件”来源：删除 disturbance_mode/scale、w_amp_n、w_seed 等合成分支。
- 加入项目相对路径解析（replay_params 不再存 C:\\ D:\\ 绝对路径）。
- UI 只读字段用 readonly=True 描述（mu/n/g/B0/K_v 显示、不可随意手调）。

设计原则不变：常量大写编号分节；所有会被 UI 修改的参数集中在 PARAM_GROUPS；
validate_params 只做纯内存检查。坐标约定见下。

坐标系约定（与 analysis/disturbance/trajectory 一致）：
- MuJoCo 世界系：x 前、y 左、z 上；base 与 world 对齐；
- 初始 home = shoulder_pan=0 使整臂沿 +X 展开；主运动 = 世界 X；被测/抑振方向 = 世界 Y；
- TCP = attachment_site；单位 m / rad / s / N。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final


# =============================================================================
# 0. 项目路径与模型资源
# =============================================================================

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
ASSET_DIR: Final[Path] = PROJECT_DIR / "assets_ur10e"
OUTPUT_ROOT: Final[Path] = PROJECT_DIR / "outputs"

# 加载 UR10e（menagerie，自带 kp=5000/kv=500 位置伺服 actuator）。
MODEL_XML_PATH: Final[Path] = ASSET_DIR / "scene.xml"
UR10E_XML_PATH: Final[Path] = ASSET_DIR / "ur10e.xml"

# 初始关节角（rad）。shoulder_pan=0 使整臂朝 +X 展开。
INIT_Q: Final[list[float]] = [0.0, -math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, 0.0]

# TCP / 力施加 body/site 名称。
TCP_SITE_NAME: Final[str] = "attachment_site"
FORCE_BODY_NAME: Final[str] = "wrist_3_link"

RUN_MODE_BASELINE: Final[str] = "baseline"      # A：无附加控制
RUN_MODE_APF_SFC: Final[str] = "apf_sfc"        # B：开启 APF+SFC
VALID_RUN_MODES: Final[set[str]] = {RUN_MODE_BASELINE, RUN_MODE_APF_SFC}

# 随包数据包（V1.2 首个固定 w(t) 数据包）默认位置（存参数时用项目相对路径）。
PACKET_DIR: Final[Path] = OUTPUT_ROOT / "wfit_P05R01"
PACKET_W_FILE: Final[Path] = PACKET_DIR / "w_force.csv"
PACKET_SCHEDULE_FILE: Final[Path] = PACKET_DIR / "schedule.csv"
PACKET_E_DES_FILE: Final[Path] = PACKET_DIR / "e_des_um.csv"
PACKET_META_FILE: Final[Path] = PACKET_DIR / "wfit_meta.json"
PACKET_TUNING_FILE: Final[Path] = PACKET_DIR / "sfc_tuning.json"
PACKET_PROFILE_FILE: Final[Path] = PACKET_DIR / "spectral_profile.json"

# 伺服参数来自模型 XML 内部 actuator（哈希溯源，见 provenance.model_fingerprint）。
SERVO_NOTE: Final[str] = "menagerie UR10e 内置关节位置伺服 kp=5000 kv=500（模型 xml 内）"


# =============================================================================
# 1. 参数 schema
# =============================================================================
# 字段键约定同 V1.1。新增 readonly: True 的字段仅用于“存档+校验+UI 只读显示”，
# 数值来源于 sfc_tune.py 生成文件，不在 UI 里编辑。
PARAM_GROUPS: Final[list[dict[str, Any]]] = [
    {
        "group": "轨迹（replay 时由 schedule 覆盖 X）",
        "fields": [
            {
                "key": "x_speed_mm_s", "label": "X 运动速度", "unit": "mm/s", "type": "float",
                "default": 5.0, "min": 0.5, "max": 20.0, "step": 0.5, "advanced": False,
                "help": "非回放匀速段默认 5 mm/s；file 回放下由 schedule.csv 覆盖。",
            },
            {
                "key": "duration_s", "label": "运动时长", "unit": "s", "type": "float",
                "default": 62.853, "min": 5.0, "max": 300.0, "step": 5.0, "advanced": False,
                "help": "file 回放时长 = 模板时长（replay_params 自动写入，勿手改）。",
            },
            {
                "key": "motion_direction", "label": "运动方向", "unit": "", "type": "choice",
                "default": "+X", "choices": ["+X", "-X"], "advanced": False,
                "help": "沿世界 X 正/负方向；X 主运动、Y 抑振方向。",
            },
        ],
    },
    {
        "group": "时序",
        "fields": [
            {
                "key": "physics_hz", "label": "物理仿真频率", "unit": "Hz", "type": "int",
                "default": 1000, "min": 250, "max": 2000, "step": 100, "advanced": False,
                "presets": [("1000 Hz", 1000)], "help": "m.opt.timestep=1/physics_hz。",
            },
            {
                "key": "control_hz", "label": "控制/采样频率", "unit": "Hz", "type": "float",
                "default": 1000.0, "min": 25.0, "max": 1000.0, "step": 1.0,
                "presets": [("1000", 1000.0), ("132", 132.0)], "advanced": False,
                "help": "APF+SFC+IK 频率，须 ≤ physics；非整数分频用时间累加器，控制器用真实 dt_actual。",
            },
        ],
    },
    {
        "group": "APF（只含吸引弹簧项）",
        "fields": [
            {
                "key": "k_a", "label": "APF 弹簧刚度 k_a", "unit": "N/m", "type": "float",
                "default": 1500.0, "min": 0.0, "max": 1.0e5, "step": 10.0, "advanced": False,
                "help": "F_apf=-k_a·e_y。第一轮冻结 1500；若改必须重跑 sfc_tune.py 重算 SFC 参数。",
            },
        ],
    },
    {
        "group": "SFC（论文核心，整定结果只读）",
        "fields": [
            {
                "key": "sfc_m", "label": "虚拟惯量 m", "unit": "kg", "type": "float",
                "default": 1.0, "min": 1.0e-3, "max": 1.0e3, "step": 0.1, "advanced": True,
                "readonly": True,
                "help": "m·v̇+μ|v|^(n-1)·v=F_apf；算法整定输入，第一轮取 1.0。",
            },
            {
                "key": "sfc_mu", "label": "剪切增稠系数 μ", "unit": "N·sⁿ/mⁿ", "type": "float",
                "default": 76150.4348, "min": 0.0, "max": 1.0e8, "step": 0.5, "advanced": True,
                "readonly": True,
                "help": "论文 Algorithm 1 输出；来自 sfc_tuning.json，勿手调（上限 1e8，科学计数）。",
            },
            {
                "key": "sfc_n", "label": "非线性指数 n", "unit": "", "type": "float",
                "default": 2.8279765, "min": 1.0, "max": 5.0, "step": 0.1, "advanced": True,
                "readonly": True,
                "help": "论文 Algorithm 1 输出；正式 1<n≤5，n=1 仅内部诊断。",
            },
            {
                "key": "sfc_g", "label": "输出增益 g", "unit": "", "type": "float",
                "default": 0.02415382, "min": 0.0, "max": 1.0, "step": 1e-4, "advanced": True,
                "readonly": True,
                "help": "论文输出增益：dẏ=g·v；论文整定结果，勿手调。",
            },
            {
                "key": "sfc_B0", "label": "线性阻尼 b_eps（非论文扩展）", "unit": "N·s/m",
                "type": "float", "default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01,
                "advanced": True, "readonly": True,
                "help": "非论文扩展，默认 0；启用须在报告中单独标记（上限 0.3，不回到 60 量级）。",
            },
            {
                "key": "sfc_K_v", "label": "回零刚度 K_v（非论文扩展）", "unit": "N/m",
                "type": "float", "default": 0.0, "min": 0.0, "max": 1.0e5, "step": 10.0,
                "advanced": True, "readonly": True,
                "help": "非论文扩展，默认 0；与 k_a 重复刚度，正式不用。",
            },
        ],
    },
    {
        "group": "等效扰动 w(t)（冻结文件，唯一来源）",
        "fields": [
            {
                "key": "disturbance_file", "label": "w(t) 文件（必填）", "unit": "", "type": "file",
                "default": "outputs/wfit_P05R01/w_force.csv", "advanced": False,
                "help": "CSV [t(s), Y向力(N)]，严格递增、有限、覆盖不短于 schedule；"
                       "加载即算 SHA-256。A/B 必须逐字节同文件。",
            },
        ],
    },
    {
        "group": "IK（阻尼最小二乘）",
        "fields": [
            {
                "key": "ik_lambda", "label": "阻尼系数 λ", "unit": "", "type": "float",
                "default": 1.0e-4, "min": 1.0e-7, "max": 1.0e-1, "step": 1.0e-5,
                "advanced": True,
                "help": "dq = Jᵀ(JJᵀ+λ²I)⁻¹·err。",
            },
            {
                "key": "ik_max_iters", "label": "IK 最大迭代", "unit": "", "type": "int",
                "default": 3, "min": 1, "max": 100, "step": 1, "advanced": True,
                "help": "每控制节拍内牛顿迭代上限。",
            },
            {
                "key": "ik_tol_m", "label": "IK 位置收敛容差", "unit": "m", "type": "float",
                "default": 1.0e-8, "min": 1.0e-12, "max": 1.0e-4, "step": 1.0e-9,
                "advanced": True,
                "help": "末端位置误差低于该值认为已收敛。",
            },
        ],
    },
    {
        "group": "高级：真实 w(t) 拟合回放",
        "fields": [
            {
                "key": "replay_schedule", "label": "机器人沿程 schedule", "unit": "", "type": "file",
                "default": "outputs/wfit_P05R01/schedule.csv", "advanced": True,
                "help": "fit_w 生成的 schedule.csv [t(s),沿程mm]，机器人按真实往返回放；"
                       "留空=沿 X 匀速直线（仅供试跑，正式用 schedule）。",
            },
        ],
    },
]


def parameter_defaults() -> dict[str, Any]:
    return {f["key"]: f["default"] for group in PARAM_GROUPS for f in group["fields"]}


def flatten_groups(groups: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    groups = groups if groups is not None else PARAM_GROUPS
    return [field for group in groups for field in group["fields"]]


def find_field(key: str) -> dict[str, Any] | None:
    for field in flatten_groups():
        if field["key"] == key:
            return field
    return None


def field_meta(groups: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    return {f["key"]: f for f in flatten_groups(groups)}


# --------------------------------------------------------------------- 路径解析
def resolve_path(value: str | Path) -> Path:
    """相对路径按项目根目录解析；绝对路径原样返回。"""
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p


def project_relative_str(path: str | Path) -> str:
    """转成相对项目根的正斜杠字符串；不在项目内的文件保留绝对路径。"""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(PROJECT_DIR.resolve())
    except ValueError:
        return str(p)
    return rel.as_posix()


# --------------------------------------------------------------------- 校验
def validate_params(params: dict[str, Any], groups: list[dict[str, Any]] | None = None) -> list[str]:
    """
    纯内存检查，返回中文错误列表（空=通过）。

    V1.2 规则：数值上下界、choice、control_hz≤physics_hz、
    disturbance_file 必填且存在、replay_schedule 若填必须存在、SFC n∈[1,5]。
    """
    groups = groups if groups is not None else PARAM_GROUPS
    errors: list[str] = []
    meta = field_meta(groups)

    for key, field in meta.items():
        if key not in params:
            errors.append(f"缺少参数 {key}（{field.get('label', key)}）。")
            continue
        value = params[key]
        label = field.get("label", key)
        ftype = field.get("type")

        if ftype in ("float", "int"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label}({key}) 必须是数字。")
                continue
            if not math.isfinite(number):
                errors.append(f"{label}({key}) 必须是有限数。")
                continue
            lo, hi = field.get("min"), field.get("max")
            if lo is not None and number < lo:
                errors.append(f"{label}({key}) 不能小于 {lo}。")
            if hi is not None and number > hi:
                errors.append(f"{label}({key}) 不能大于 {hi}。")
        elif ftype == "choice":
            if value not in field.get("choices", []):
                errors.append(f"{label}({key}) 只能取 {field.get('choices')}。")
        elif ftype == "file":
            if value:
                p = resolve_path(str(value))
                if not p.is_file():
                    errors.append(f"{label}({key}) 文件不存在：{value}")
            elif field.get("key") == "disturbance_file":
                errors.append(f"{label}({key}) 必填（正式扰动只有文件来源）。")

    if "control_hz" in params and "physics_hz" in params:
        if float(params["control_hz"]) > float(params["physics_hz"]):
            errors.append("控制/采样频率不能高于物理仿真频率。")
    if "sfc_n" in params:
        n = float(params["sfc_n"])
        if not (1.0 <= n <= 5.0):
            errors.append("SFC 指数 n 必须在 1..5（正式 1<n≤5）。")
    return errors


# =============================================================================
# 2. JSON 存档 / 读取
# =============================================================================

def save_parameters(params: dict[str, Any], path: Path) -> None:
    """扁平参数 + 版本/生成时间写成 JSON 快照（V1.2）。"""
    import json
    from datetime import datetime
    snapshot = {
        "version": "v1.2",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": dict(params),
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def load_parameters(path: Path) -> dict[str, Any]:
    """读回参数；缺字段用默认补齐、未知字段忽略（向前兼容）。"""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("parameters", data) if isinstance(data, dict) else {}
    merged = parameter_defaults()
    for key, value in raw.items():
        if key in merged:
            merged[key] = value
    return merged


# =============================================================================
# 3. 便捷派生值
# =============================================================================

def derived(params: dict[str, Any]) -> dict[str, Any]:
    speed_mm_s = float(params["x_speed_mm_s"])
    return {
        "vx_m_s": speed_mm_s / 1000.0,
        "motion_sign": 1.0 if params["motion_direction"] == "+X" else -1.0,
        "physics_dt_s": 1.0 / float(params["physics_hz"]),
        "control_dt_s": 1.0 / float(params["control_hz"]),
        "total_distance_m": speed_mm_s / 1000.0 * float(params["duration_s"]),
    }
