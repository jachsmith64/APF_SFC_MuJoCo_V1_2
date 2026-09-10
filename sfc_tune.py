"""
SFC 参数整定脚本（V1.2）：按 SFC 原论文 Algorithm 1 + P05R01 幅值分位统计生成 sfc_tuning.json。

论文核心：m·v̇ + μ|v|^(n-1)v = F_apf，dẏ = g·v。Algorithm 1 的输入是
(f_ease, f_interf, v_d, v_c, ω_c,ease, dt, m)，输出 (μ, n, g)。

本项目把它适配到“APF 虚拟力”：f_ease = k_a·e_ease、f_interf = k_a·e_interf，
其中 e_ease = P50(|e_des|)、e_interf = P99(|e_des|)，e_des 取随包 e_des_um.csv。

公式（与《DS 修改指南》§4.2 逐行一致）：
    n      = ln(f_interf/f_ease) / ln(v_c/v_d)
    Ψ(n)   = 2√π·Γ(1+n/2) / Γ((3+n)/2)
    μ      = (m·ω_c,ease)^n / Ψ(n) · (√2 / f_ease)^(n-1)
    g      = v_d · (μ/f_ease)^(1/n)
    v_d    = ω_c,ease · e_ease ;  v_c = ratio·v_d
    dt_max = 2/(n·μ·v_ss^(n-1)/m)，v_ss = (k_a·max|e_des|/μ)^(1/n)（论文离散约束显式近似）

用法：
    python sfc_tune.py                        # 用默认（P05R01, k_a=1500, m=1, fc=1Hz, ratio=1.5）
    python sfc_tune.py --fc-ease 1.3 --velocity-ratio 2.0
    python sfc_tune.py --k-a 2000             # 改 k_a 必须整组重算

产物 sfc_tuning.json 保存全部输入/公式中间量/输出与 e_des 的 SHA-256，作为参数来源审计。
正式 A/B 前必须与 config 默认（或 replay_params）一致；改动只经本脚本成对重算，不手调 mu/n/g。
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

_E_DES_DEFAULT = config.PACKET_E_DES_FILE
_OUT_DEFAULT = config.PACKET_TUNING_FILE

# 无量纲压缩比取值建议（《DS 修改指南》节点5）
RECOMMENDED_RATIOS = [1.3, 1.5, 2.0]


def tuning_from_e_des(
    e_abs_um: np.ndarray,
    k_a: float,
    m: float,
    fc_ease_hz: float,
    velocity_ratio: float,
) -> dict:
    """由 |e_des|(µm) 序列按论文 Algorithm 1 计算整定结果（纯函数，可测）。"""
    e_abs = np.asarray(e_abs_um, dtype=float)
    if e_abs.size == 0 or not np.all(np.isfinite(e_abs)):
        raise ValueError("e_des 需为非空有限序列")
    e_ease_um = float(np.quantile(e_abs, 0.50))
    e_interf_um = float(np.quantile(e_abs, 0.99))
    e_max_um = float(np.max(e_abs))
    e_ease = e_ease_um * 1e-6            # m
    e_interf = e_interf_um * 1e-6        # m
    e_max = e_max_um * 1e-6              # m
    f_ease = k_a * e_ease                # N
    f_interf = k_a * e_interf            # N
    f_max = k_a * e_max                  # N
    if not (f_ease > 0 < f_interf and f_interf > f_ease):
        raise ValueError("f_interf 必须大于 f_ease>0（P99>P50）")
    if velocity_ratio <= 1.0:
        raise ValueError(f"velocity_ratio 必须 >1（建议 {RECOMMENDED_RATIOS}）")
    if fc_ease_hz <= 0:
        raise ValueError("fc_ease 必须 >0 Hz")

    n = math.log(f_interf / f_ease) / math.log(velocity_ratio)
    Psi = 2.0 * math.sqrt(math.pi) * math.gamma(1.0 + n / 2.0) / math.gamma((3.0 + n) / 2.0)
    omega_c = 2.0 * math.pi * fc_ease_hz          # rad/s
    v_d = omega_c * e_ease                        # m/s
    v_c = velocity_ratio * v_d
    mu = (m * omega_c) ** n / Psi * (math.sqrt(2.0) / f_ease) ** (n - 1.0)
    g = v_d * (mu / f_ease) ** (1.0 / n)

    # 论文离散约束（显式欧拉对线性化阻尼稳定上界），0<... 见 apf_sfc.sfc_dt_max
    v_ss = (f_max / mu) ** (1.0 / n)
    a_lin = n * mu * abs(v_ss) ** (n - 1.0) / m
    dt_max = 2.0 / a_lin if a_lin > 0 else float("inf")

    return {
        "method": "paper_algorithm_1_adapted_to_apf_virtual_force",
        "k_a_N_per_m": float(k_a),
        "m": float(m),
        "fc_ease_Hz": float(fc_ease_hz),
        "omega_c_ease_rad_s": float(omega_c),
        "velocity_ratio": float(velocity_ratio),
        "e_ease_quantile": 0.50,
        "e_interf_quantile": 0.99,
        "e_ease_um": round(e_ease_um, 9),
        "e_interf_um": round(e_interf_um, 9),
        "e_max_um": round(e_max_um, 9),
        "f_ease_N": float(f_ease),
        "f_interf_N": float(f_interf),
        "f_max_N": float(f_max),
        "Psi": float(Psi),
        "v_d_m_s": float(v_d),
        "v_c_m_s": float(v_c),
        "n": float(n),
        "mu": float(mu),
        "g": float(g),
        "B0": 0.0,
        "K_v": 0.0,
        "dt_max_s": float(dt_max),
        "dt_note": "论文离散约束显式近似（欧拉对 v_ss 线性化阻尼）；非硬编码，换 e_des 自动重算",
    }


def tune_and_write(e_des_file: Path, k_a: float, m: float, fc_ease: float,
                   velocity_ratio: float, out_file: Path) -> dict:
    """读 e_des、整定、算哈希、写 sfc_tuning.json。"""
    ed = np.loadtxt(str(e_des_file), delimiter=",", comments="#")
    if ed.ndim == 1 or ed.shape[1] < 2:
        raise ValueError(f"e_des 文件需 [t, um] 两列：{e_des_file}")
    e_abs = np.abs(ed[:, 1])
    doc = tuning_from_e_des(e_abs, k_a, m, fc_ease, velocity_ratio)
    doc["source_e_des_file"] = str(e_des_file)
    doc["source_e_des_sha256"] = provenance.sha256_file(e_des_file)
    doc["B0_note"] = "论文整定不含线性阻尼；B0 仅作非论文扩展 b_eps，正式=0"
    doc["K_v_note"] = "论文整定不含回零刚度；K_v 仅作非论文扩展，正式=0"
    doc["discrete_note"] = ("first-round formal: 1000Hz(1ms) & 132Hz(~7.58ms) 均低于 dt_max；"
                            "超限自动中止见 run.py")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFC 论文 Algorithm1 整定 -> sfc_tuning.json")
    p.add_argument("--e-des", default=str(_E_DES_DEFAULT), help="e_des_um.csv 路径")
    p.add_argument("--k-a", type=float, default=1500.0, help="APF 刚度 N/m（改则整组重算）")
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
        doc = tune_and_write(Path(args.e_des), args.k_a, args.m,
                             args.fc_ease, args.velocity_ratio, Path(args.out))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[sfc_tune] 失败：{exc}", file=sys.stderr)
        return 1
    print(f"[sfc_tune] e_des sha256={doc['source_e_des_sha256'][:16]}…")
    print(f"[sfc_tune] n={doc['n']:.7f}  mu={doc['mu']:.4f}  g={doc['g']:.8f}  "
          f"dt_max={doc['dt_max_s']:.4f}s")
    print(f"[sfc_tune] 完成 -> {Path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
