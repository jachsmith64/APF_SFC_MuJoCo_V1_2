"""
SFC 参数整定脚本（V1.3）：按 SFC 原论文 Algorithm 1 + P05R01 的**加速度**分位统计。

V1.3 相对 V1.2 的正式改动（旧版 f_ease=k_a·e_ease 在新架构下已失效）：
- 读 e_des_um.csv 的真实 t 列与位移列，位移 µm→m；
- 用与正式控制**相同**的因果二次拟合算法（apf_sfc.CausalAccelForceMapper）生成 a_y_est 序列；
- 跳过估计器未 ready 的前几项；
- F_vir_series = force_map_mass_kg · a_y_est_series；
- f_ease = P50(|F_vir_series|)、f_interf = P99(|F_vir_series|)、f_max = max(|F_vir_series|)；
- 保留 Chen Algorithm 1 由 (f_ease, f_interf, v_d, v_c, ω_c, m) 计算 (n, μ, g) 的后续公式：
      n    = ln(f_interf/f_ease) / ln(v_c/v_d)
      Ψ(n) = 2√π·Γ(1+n/2) / Γ((3+n)/2)
      μ    = (m·ω_c)^n / Ψ(n) · (√2 / f_ease)^(n-1)
      g    = v_d · (μ/f_ease)^(1/n)
      v_d  = ω_c·e_ease ;  v_c = ratio·v_d
  dt_max = 2/(n·μ·v_ss^(n-1)/m)，v_ss = (f_max/μ)^(1/n)（论文离散约束显式近似）

用法：
    python sfc_tune.py                     # 默认 P05R01, force_map_mass_kg=1.0, m=1, fc=1Hz, ratio=1.5
    python sfc_tune.py --force-map-mass 2.5 --accel-window-points 7

产物 sfc_tuning.json 保存全部输入/公式中间量/输出与 e_des 的 SHA-256，作为参数来源审计。
V1.2 旧格式（含 k_a_N_per_m 或缺新字段）由 load_tuning() 明确拒绝，不静默读取。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import config
import provenance
from apf_sfc import CausalAccelForceMapper

_E_DES_DEFAULT = config.PACKET_E_DES_FILE
_OUT_DEFAULT = config.PACKET_TUNING_FILE

# 无量纲压缩比取值建议（《DS 修改指南》节点5）
RECOMMENDED_RATIOS = [1.3, 1.5, 2.0]

METHOD_NAME = "acceleration_to_virtual_force"

# V1.3 整定文件必须包含的字段；缺任一（或含旧版 k_a_N_per_m）即视为旧格式。
REQUIRED_TUNING_KEYS = (
    "method", "force_map_mass_kg", "accel_window_points",
    "f_ease_N", "f_interf_N", "f_max_N",
    "a_est_rms_m_s2", "a_est_peak_m_s2", "m", "mu", "n", "g",
)


def load_tuning(path: Path) -> dict:
    """
    读 sfc_tuning.json 并校验为 V1.3 格式。

    旧格式（V1.2：含 k_a_N_per_m，或缺少 acceleration_to_virtual_force 新字段）
    **明确报错**，不静默读取——旧参数在新架构下已失效。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少 SFC 整定文件：{path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "k_a_N_per_m" in doc:
        raise ValueError(
            f"{path} 是 V1.2 旧格式整定文件（含 k_a_N_per_m），在新架构下已失效；"
            f"请重新运行 sfc_tune.py 生成 V1.3 整定文件。")
    missing = [k for k in REQUIRED_TUNING_KEYS if k not in doc]
    if missing:
        raise ValueError(
            f"{path} 缺少 V1.3 整定字段 {missing}；请重新运行 sfc_tune.py 生成 V1.3 整定文件。")
    if doc.get("method") != METHOD_NAME:
        raise ValueError(
            f"{path} 的 method={doc.get('method')!r}，期望 {METHOD_NAME!r}；"
            f"请重新运行 sfc_tune.py。")
    return doc


def tuning_from_e_des(
    t_s: np.ndarray,
    e_um: np.ndarray,
    force_map_mass_kg: float,
    m: float,
    fc_ease_hz: float,
    velocity_ratio: float,
    accel_window_points: int = 5,
) -> dict:
    """
    由 e_des 的 (t, 位移µm) 序列按“加速度→虚拟力”+ 论文 Algorithm 1 计算整定（纯函数，可测）。

    加速度用与正式控制相同的因果二次拟合（CausalAccelForceMapper）逐点估计。
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    e_um_arr = np.asarray(e_um, dtype=float).reshape(-1)
    if t.size == 0 or t.size != e_um_arr.size:
        raise ValueError("e_des 的 t 与位移列必须等长非空")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e_um_arr))):
        raise ValueError("e_des 需为非空有限序列")
    if np.any(np.diff(t) <= 0):
        raise ValueError("e_des 的 t 列必须严格递增（因果估计要求真实时间戳）")
    if velocity_ratio <= 1.0:
        raise ValueError(f"velocity_ratio 必须 >1（建议 {RECOMMENDED_RATIOS}）")
    if fc_ease_hz <= 0:
        raise ValueError("fc_ease 必须 >0 Hz")
    if not (force_map_mass_kg > 0.0):
        raise ValueError(f"force_map_mass_kg 必须 >0：{force_map_mass_kg}")

    e_m = e_um_arr * 1e-6                                   # µm -> m

    # 与正式控制一致的因果二次拟合；跳过未 ready 的前几项
    mapper = CausalAccelForceMapper(force_map_mass_kg, int(accel_window_points))
    a_ready: list[float] = []
    f_ready: list[float] = []
    for ti, ei in zip(t, e_m):
        a_est, f_vir, ready = mapper.step(float(ti), float(ei))
        if ready:
            a_ready.append(a_est)
            f_ready.append(f_vir)
    if not a_ready:
        raise ValueError("加速度估计器未产生有效样本（e_des 长度不足窗口点数）")
    a_series = np.asarray(a_ready, dtype=float)
    f_series = np.asarray(f_ready, dtype=float)
    f_abs = np.abs(f_series)

    f_ease = float(np.quantile(f_abs, 0.50))
    f_interf = float(np.quantile(f_abs, 0.99))
    f_max = float(np.max(f_abs))
    if not (f_ease > 0.0 < f_interf and f_interf > f_ease):
        raise ValueError(f"f_interf 必须大于 f_ease>0（实际 P50={f_ease}, P99={f_interf}）")

    e_abs = np.abs(e_m)
    e_ease_um = float(np.quantile(np.abs(e_um_arr), 0.50))
    e_ease = e_ease_um * 1e-6                               # 特征位移（m）

    n = math.log(f_interf / f_ease) / math.log(velocity_ratio)
    if not (1.0 < n <= 5.0):
        raise ValueError(
            f"整定得到 n={n:.4f}，超出正式范围 1<n≤5。P99/P50(|F_vir|)="
            f"{f_interf / f_ease:.3f} 与 velocity_ratio={velocity_ratio} 组合过陡；"
            f"请调整 velocity_ratio（建议 {RECOMMENDED_RATIOS}）或 force_map_mass_kg。")
    Psi = 2.0 * math.sqrt(math.pi) * math.gamma(1.0 + n / 2.0) / math.gamma((3.0 + n) / 2.0)
    omega_c = 2.0 * math.pi * fc_ease_hz          # rad/s
    v_d = omega_c * e_ease                        # m/s
    v_c = velocity_ratio * v_d
    mu = (m * omega_c) ** n / Psi * (math.sqrt(2.0) / f_ease) ** (n - 1.0)
    g = v_d * (mu / f_ease) ** (1.0 / n)

    # 论文离散约束（显式欧拉对线性化阻尼稳定上界），与 apf_sfc.sfc_dt_max 同口径
    v_ss = (f_max / mu) ** (1.0 / n)
    a_lin = n * mu * abs(v_ss) ** (n - 1.0) / m
    dt_max = 2.0 / a_lin if a_lin > 0 else float("inf")

    return {
        "method": METHOD_NAME,
        "force_map_mass_kg": float(force_map_mass_kg),
        "accel_window_points": int(accel_window_points),
        "m": float(m),
        "fc_ease_Hz": float(fc_ease_hz),
        "omega_c_ease_rad_s": float(omega_c),
        "velocity_ratio": float(velocity_ratio),
        "e_ease_quantile": 0.50,
        "e_interf_quantile": 0.99,
        "e_ease_um": round(e_ease_um, 9),
        "n_accel_samples": int(a_series.size),
        "n_accel_skipped": int(t.size - a_series.size),
        "f_ease_N": float(f_ease),
        "f_interf_N": float(f_interf),
        "f_max_N": float(f_max),
        "a_est_rms_m_s2": float(np.sqrt(np.mean(a_series ** 2))),
        "a_est_peak_m_s2": float(np.max(np.abs(a_series))),
        "Psi": float(Psi),
        "v_d_m_s": float(v_d),
        "v_c_m_s": float(v_c),
        "n": float(n),
        "mu": float(mu),
        "g": float(g),
        "dt_max_s": float(dt_max),
        "dt_note": "论文离散约束显式近似（欧拉对 v_ss 线性化阻尼，f_max 为最大虚拟力）；"
                   "非硬编码，换 e_des/映射系数自动重算",
    }


def tune_and_write(e_des_file: Path, force_map_mass_kg: float, m: float, fc_ease: float,
                   velocity_ratio: float, accel_window_points: int,
                   out_file: Path) -> dict:
    """读 e_des、整定、算哈希、写 sfc_tuning.json（V1.3 格式）。"""
    ed = np.loadtxt(str(e_des_file), delimiter=",", comments="#")
    if ed.ndim == 1 or ed.shape[1] < 2:
        raise ValueError(f"e_des 文件需 [t, um] 两列：{e_des_file}")
    doc = tuning_from_e_des(ed[:, 0], ed[:, 1], force_map_mass_kg, m,
                            fc_ease, velocity_ratio, accel_window_points)
    doc["source_e_des_file"] = str(e_des_file)
    doc["source_e_des_sha256"] = provenance.sha256_file(e_des_file)
    doc["f_vir_note"] = ("F_vir=+force_map_mass_kg·a_y_est（正号）；剪切增稠阻力由 SFC 方程"
                         "内部 -μ|v_s|^(n-1)·v_s 体现，输出端不加固定负号。")
    doc["discrete_note"] = ("first-round formal: 1000Hz(1ms) & 132Hz(~7.58ms) 均低于 dt_max；"
                            "超限自动中止见 run.py")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFC 加速度→虚拟力整定 -> sfc_tuning.json (V1.3)")
    p.add_argument("--e-des", default=str(_E_DES_DEFAULT), help="e_des_um.csv 路径")
    p.add_argument("--force-map-mass", type=float, default=1.0,
                   help="加速度→虚拟力映射系数 kg（归一化初值，改则整组重算）")
    p.add_argument("--accel-window-points", type=int, default=5,
                   help="加速度估计窗口点数（≥3 的奇数：3/5/7/9）")
    p.add_argument("--m", type=float, default=1.0, help="归一化虚拟惯量")
    p.add_argument("--fc-ease", type=float, default=1.0, help="数据过渡频率 fc_ease(Hz)")
    p.add_argument("--velocity-ratio", type=float, default=1.5,
                   help="速度压缩比 v_c/v_d（建议 1.3/1.5/2.0）")
    p.add_argument("--out", default=str(_OUT_DEFAULT), help="sfc_tuning.json 输出路径")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        doc = tune_and_write(Path(args.e_des), args.force_map_mass, args.m,
                             args.fc_ease, args.velocity_ratio,
                             args.accel_window_points, Path(args.out))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[sfc_tune] 失败：{exc}", file=sys.stderr)
        return 1
    print(f"[sfc_tune] e_des sha256={doc['source_e_des_sha256'][:16]}…")
    print(f"[sfc_tune] F_vir: P50={doc['f_ease_N']:.6g} N  P99={doc['f_interf_N']:.6g} N  "
          f"max={doc['f_max_N']:.6g} N  (a_peak={doc['a_est_peak_m_s2']:.6g} m/s²)")
    print(f"[sfc_tune] n={doc['n']:.7f}  mu={doc['mu']:.4f}  g={doc['g']:.8f}  "
          f"dt_max={doc['dt_max_s']:.4f}s")
    print(f"[sfc_tune] 完成 -> {Path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
