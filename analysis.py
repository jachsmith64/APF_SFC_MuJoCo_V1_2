"""
离线分析 V1.2：读 run 目录的 trajectory.csv + run_fingerprint.json，出 A/B 对比。

V1.2 相对 V1.1 的正式改动（《DS 修改指南》§6 + R-008/R-014/R-015/R-017/R-021）：
- 频带不再写死 1-5/5-10/.../1-45：振动带与慢成分区间来自预实验冻结的
  spectral_profile.json（make_profile.py 产物），只有 A 组/预实验数据，B 不参与选带。
- 分析基于真实时间戳用“平均步长”均匀重采样（spectral.resample_uniform），不用 median
  （132Hz 7/8ms 交替若按 median→125Hz 会错频）；1000Hz 日志本已均匀则保持原样。
- “慢偏置主导频率”与“振动主导频率”分开报告（R-014），不把 0.5-0.8Hz 慢瓣当振动主峰。
- 只分析 completed 的正式 run；A/B 经 run_fingerprint 门控（同一 w/SHA、同一模型指纹、
  同一 INIT_Q、同一物理/控制频率与时长、params 指纹一致、模式分别为 baseline/apf_sfc），
  不满足即报错（R-017）。profile/w/指纹都记 SHA-256 进结果，防止换数据后旧谱带混用。

用法：
    python analysis.py --pair <runA目录> <runB目录> --labels A B --out <分析目录> [--strict]
    python analysis.py --run <单个run目录> --label X --out <目录>          # 单组诊断

输出：<out>/analysis_metrics.json + analysis_summary.txt + figures/ 下 4 张图：
  fig1_e_y_time.png / fig2_amp_sliding.png / fig3_psd.png / fig4_band_rms.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
import provenance
import spectral

_COLS = ["t", "x_ref", "y_ref", "z_ref", "x_cmd", "y_cmd",
         "x_act", "y_act", "z_act", "dy", "F_apf", "w_force",
         "dt_actual", "sfc_v_internal", "sfc_v_out", "sfc_shear_force"]

AMP_WIN_S, AMP_STEP_S = 1.0, 0.2
WELCH_WIN_S = 2.0
# 低于该时长的运动段不值得单独列指标（对应分析窗口的统计下限）
MIN_SEG_S = 0.3


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ----------------------------------------------------------------- 读取
def load_run_csv(run_dir: Path) -> dict[str, np.ndarray]:
    """按列名读 trajectory.csv，返回 {col: array}。"""
    run_dir = Path(run_dir)
    csv = run_dir / "trajectory.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"缺少 {csv}")
    with open(csv, encoding="utf-8") as fh:
        header = fh.readline()
    if not header.startswith("# "):
        raise ValueError(f"{csv} 首行应为列名注释，实际：{header[:60]!r}")
    names = [c.strip() for c in header[2:].split(",")]
    missing = [c for c in _COLS if c not in names]
    if missing:
        raise ValueError(f"{csv} 缺列 {missing}")
    data = np.loadtxt(str(csv), delimiter=",", comments="#")
    out = {name: data[:, names.index(name)] for name in _COLS if data.ndim == 2}
    return out


def load_fingerprint(run_dir: Path) -> dict:
    fp = Path(run_dir) / "run_fingerprint.json"
    if not fp.is_file():
        raise FileNotFoundError(f"缺少 {fp}（该 run 未完成写入，不能进入正式分析）")
    return json.loads(fp.read_text(encoding="utf-8"))


def locate_packet(run_dir: Path) -> Path:
    """从 run 指纹的 disturbance_file（项目相对路径）定位数据包目录。"""
    fp = load_fingerprint(run_dir)
    dw = fp.get("params", {}).get("disturbance_file", "")
    if dw:
        p = config.resolve_path(dw).parent
        if p.is_dir():
            return p
    raise FileNotFoundError(f"{run_dir} 无法从指纹定位数据包（disturbance_file={dw}）")


def load_profile(packet_dir: Path) -> dict:
    p = Path(packet_dir) / "spectral_profile.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"缺冻结频谱 profile {p}。请先对预实验数据跑 make_profile.py（V1.2 频带来源），"
            f"不能回退到写死的 1-45Hz。")
    return json.loads(p.read_text(encoding="utf-8"))


def gate_pair(dir_a: Path, dir_b: Path, strict: bool = True) -> dict:
    """
    A/B 公平性门控：返回 {ok, problems[], gate}。
    strict=True 时任一问题 → ok=False（形式对比必须过）；strict=False 仅记 warnings。
    """
    fa, fb = load_fingerprint(dir_a), load_fingerprint(dir_b)
    problems: list[str] = []
    mode_a, mode_b = fa.get("mode"), fb.get("mode")
    if not fa.get("completed"):
        problems.append(f"A 组 run 未完成（reason={fa.get('reason')}）")
    if not fb.get("completed"):
        problems.append(f"B 组 run 未完成（reason={fb.get('reason')}）")
    if not (mode_a == config.RUN_MODE_BASELINE and mode_b == config.RUN_MODE_APF_SFC):
        problems.append(f"组模式应为 baseline/apf_sfc，实际 {mode_a}/{mode_b}")
    if fa.get("params_fingerprint") != fb.get("params_fingerprint"):
        problems.append("A/B 参数指纹不一致")
    if fa.get("disturbance", {}).get("sha256") != fb.get("disturbance", {}).get("sha256"):
        problems.append("A/B 扰动文件 SHA-256 不一致")
    if fa.get("schedule", {}).get("sha256") != fb.get("schedule", {}).get("sha256"):
        problems.append("A/B schedule SHA-256 不一致")
    if fa.get("model") != fb.get("model"):
        problems.append("A/B 模型指纹不一致")
    if fa.get("init_q") != fb.get("init_q"):
        problems.append("A/B INIT_Q 不一致")
    for k in ("physics_hz", "control_hz", "duration_s"):
        va, vb = fa.get(k), fb.get(k)
        if va is None or vb is None or abs(float(va) - float(vb)) > 1e-6:
            problems.append(f"A/B {k} 不一致：{va} vs {vb}")
    if not fb.get("sfc_paper_consistent"):
        problems.append("B 组 SFC 不是论文一致参数（sfc_paper_consistent=False）")
    return {"ok": not problems, "problems": problems, "gate": "strict" if strict else "warn"}


# ----------------------------------------------------------------- 单 run 指标
def analyze_run(run_dir: Path, profile: dict, label: str) -> dict:
    """单组完整指标。返回 dict（µm 口径；谱基于平均步长均匀重采样）。"""
    run_dir = Path(run_dir)
    fp = load_fingerprint(run_dir)
    if not fp.get("completed"):
        raise ValueError(f"{run_dir} 未完成（reason={fp.get('reason')}），不可分析。")
    csv = load_run_csv(run_dir)
    t_raw = csv["t"]
    e_y_um = (csv["y_act"] - csv["y_ref"]) * 1e6

    # 真实时间戳 → 均匀网格（平均步长），供 Welch；1000Hz 本已均匀则为 no-op
    t_u, ey_u, fs = spectral.resample_uniform(t_raw, e_y_um)
    if fs <= 0:
        raise ValueError(f"{run_dir} 采样时间无效（fs={fs}）")

    slow_lo, slow_hi = profile["slow_component_range_Hz"]
    bands = [tuple(b) for b in profile["identified_vibration_bands_Hz"]]
    f_m, p_m = spectral.welch_psd(ey_u, fs, win_s=WELCH_WIN_S)

    def _rms(lo: float, hi: float) -> float:
        return spectral.band_rms(f_m, p_m, lo, hi)

    band_rms = {f"{i:02d}@{lo:.2f}-{hi:.2f}": _rms(lo, hi)
                for i, (lo, hi) in enumerate(bands)}
    vib_var = sum(v * v for v in band_rms.values())
    vib_total = float(np.sqrt(vib_var))
    slow_rms = _rms(slow_lo, slow_hi)

    # 主导频率：按 PSD 幅度比，只在目标频带内取最大；跳过 0Hz/DC bin（均值残差假峰）
    vib_mask = np.zeros(f_m.size, dtype=bool)
    for lo, hi in bands:
        vib_mask |= (f_m >= lo) & (f_m < hi)
    slow_mask = (f_m >= slow_lo) & (f_m < slow_hi)

    def _peak_hz(mask: np.ndarray) -> float:
        mm = mask & (f_m >= f_m[1])          # f>0：排除 DC bin
        if not mm.any():
            return float("nan")
        return float(f_m[int(np.argmax(np.where(mm, p_m, -np.inf)))])

    vib_dominant = _peak_hz(vib_mask)
    slow_dominant = _peak_hz(slow_mask)

    seg = _segment_stats(run_dir, fp, ey_u, t_u)

    out = {
        "label": label,
        "run_dir": str(run_dir),
        "mode": fp.get("mode"),
        "completed": fp.get("completed"),
        "fp": {
            "params_fingerprint": fp.get("params_fingerprint"),
            "w_sha256": (fp.get("disturbance") or {}).get("sha256"),
            "schedule_sha256": (fp.get("schedule") or {}).get("sha256"),
            "model": fp.get("model"),
            "init_q": fp.get("init_q"),
            "physics_hz": fp.get("physics_hz"),
            "control_hz": fp.get("control_hz"),
            "duration_s": fp.get("duration_s"),
            "sfc_paper_consistent": fp.get("sfc_paper_consistent"),
        },
        "sample": {
            "n_uniform": int(ey_u.size),
            "fs_hz": round(fs, 3),
            "resample": "uniform(mean-step)" if not np.allclose(
                np.diff(t_raw), np.diff(t_raw)[0], atol=1e-6) else "already-uniform",
            "t_span_s": [round(float(t_u[0]), 3), round(float(t_u[-1]), 3)],
            "dt_actual_median_s": float(np.median(csv["dt_actual"])) if "dt_actual" in csv else None,
        },
        "time": {
            "mean_um": float(np.mean(e_y_um)),
            "std_um": float(np.std(e_y_um, ddof=1)),
            "rms_um": float(np.sqrt(np.mean(e_y_um ** 2))),
            "ptp_um": float(np.ptp(e_y_um)),
        },
        "spectral": {
            "welch_window_s": WELCH_WIN_S,
            "resolution_hz": round(float(f_m[1] - f_m[0]), 4),
        },
        "slow": {
            "range_hz": [slow_lo, slow_hi],
            "profile_dominant_hz": profile.get("slow_bias_dominant_Hz"),
            "sim_dominant_hz": round(slow_dominant, 4),
            "rms_um": round(slow_rms, 4),
        },
        "vibration": {
            "bands_hz": [list(b) for b in bands],
            "profile_bands_from": profile["source_sha256"][:12],
            "band_rms_um": {k: round(v, 4) for k, v in band_rms.items()},
            "total_rms_um": round(vib_total, 4),
            "dominant_hz": round(vib_dominant, 4),
        },
        "segment": seg,
    }
    return out


def _segment_stats(run_dir: Path, fp: dict, ey_u: np.ndarray, t_u: np.ndarray) -> dict:
    """
    用数据包 wfit_meta.json 的 template.segments（与 run 时间轴一致）把稳态往返窗归
    outbound_steady / return_steady，其余归 edge_transient，分窗给时域 RMS。
    """
    packet = Path(fp.get("params", {}).get("disturbance_file", ""))
    if not packet:
        return {}
    meta_p = config.resolve_path(packet).parent / "wfit_meta.json"
    if not meta_p.is_file():
        return {}
    try:
        segs = json.loads(meta_p.read_text(encoding="utf-8"))["template"]["segments"]
    except (KeyError, OSError, ValueError):
        return {}

    classes: dict[str, list[tuple[float, float]]] = {"outbound_steady": [],
                                                     "return_steady": [], "edge_transient": []}
    for s in segs:
        name = s.get("name", "")
        t0, t1 = float(s["t0_s"]), float(s["t1_s"])
        if "mid_fwd" in name or "forward_steady" in name:
            classes["outbound_steady"].append((t0, t1))
        elif "mid_ret" in name or "return_steady" in name:
            classes["return_steady"].append((t0, t1))
        else:
            classes["edge_transient"].append((t0, t1))

    def _over_windows(ws: list[tuple[float, float]]) -> dict | None:
        vals = []
        total = 0.0
        for t0, t1 in ws:
            m = (t_u >= t0) & (t_u < t1)
            if m.any():
                v = ey_u[m]
                vals.append(v)
                total += float(t1 - t0)
        if not vals or total < MIN_SEG_S:
            return None
        allv = np.concatenate(vals)
        return {"n_windows": len(vals), "total_s": round(total, 3),
                "rms_um": round(float(np.sqrt(np.mean(allv ** 2))), 4),
                "mean_um": round(float(np.mean(allv)), 4),
                "ptp_um": round(float(np.ptp(allv)), 4)}

    out = {k: _over_windows(ws) for k, ws in classes.items()}
    return {k: v for k, v in out.items() if v is not None}


# ----------------------------------------------------------------- A/B 对比
def compare(ma: dict, mb: dict) -> dict:
    """构造对比字段（全部比值的分母为 A；A 为 0 时给 None）。"""
    a, b = ma, mb
    def _ratio(xa: float, xb: float):
        return round(xb / xa, 4) if xa and xa > 0 else None

    def _pct(xa: float, xb: float):
        return round((1.0 - xb / xa) * 100.0, 2) if xa and xa > 0 else None

    comp = {
        "base": a["label"], "ctrl": b["label"],
        "time_rms_um": {a["label"]: a["time"]["rms_um"], b["label"]: b["time"]["rms_um"]},
        "time_rms_ratio": _ratio(a["time"]["rms_um"], b["time"]["rms_um"]),
        "slow_rms_um": {a["label"]: a["slow"]["rms_um"], b["label"]: b["slow"]["rms_um"]},
        "slow_rms_ratio": _ratio(a["slow"]["rms_um"], b["slow"]["rms_um"]),
        "vibration_total_rms_um": {a["label"]: a["vibration"]["total_rms_um"],
                                   b["label"]: b["vibration"]["total_rms_um"]},
        "vibration_total_ratio": _ratio(a["vibration"]["total_rms_um"],
                                        b["vibration"]["total_rms_um"]),
        "vibration_total_suppression_pct": _pct(a["vibration"]["total_rms_um"],
                                                b["vibration"]["total_rms_um"]),
        "dominant_hz": {a["label"]: a["vibration"]["dominant_hz"],
                        b["label"]: b["vibration"]["dominant_hz"]},
        "slow_dominant_hz": {a["label"]: a["slow"]["sim_dominant_hz"],
                             b["label"]: b["slow"]["sim_dominant_hz"]},
        "band_rms_um": {k: {a["label"]: a["vibration"]["band_rms_um"][k],
                            b["label"]: b["vibration"]["band_rms_um"][k]}
                        for k in a["vibration"]["band_rms_um"]},
        "band_ratio": {k: _ratio(a["vibration"]["band_rms_um"][k],
                                 b["vibration"]["band_rms_um"][k])
                       for k in a["vibration"]["band_rms_um"]},
    }
    return comp


def _matplotlib_zh():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _uni(run_dir: Path):
    """均匀网格的 (t_u, ey_um, fs)，供画图。"""
    csv = load_run_csv(run_dir)
    e = (csv["y_act"] - csv["y_ref"]) * 1e6
    return spectral.resample_uniform(csv["t"], e)


def _plot_all(run_dirs: list[Path], labels: list[str], profile: dict, fig_dir: Path,
              metrics: dict, comp: dict | None) -> None:
    plt = _matplotlib_zh()
    fig_dir.mkdir(parents=True, exist_ok=True)
    slow_lo, slow_hi = profile["slow_component_range_Hz"]
    bands = profile["identified_vibration_bands_Hz"]

    # fig1 时域
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for lab, rd in zip(labels, run_dirs):
        t_u, ey, _ = _uni(rd)
        ax.plot(t_u, ey, lw=0.5, label=f"{lab} (e_y)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("时间 / s"); ax.set_ylabel("e_y = y_act−y_ref / µm")
    ax.set_title("e_y(t)（正式 A/B 同 w(t)/同参数）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_e_y_time.png", dpi=130); plt.close(fig)

    # fig2 滑窗 RMS
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for lab, rd in zip(labels, run_dirs):
        t_u, ey, fs = _uni(rd)
        n = int(round(AMP_WIN_S * fs)); step = int(round(AMP_STEP_S * fs))
        ct, rms = [], []
        for i in range(0, ey.size - n + 1, max(1, step)):
            ct.append(t_u[i + n // 2]); rms.append(float(np.sqrt(np.mean(ey[i:i + n] ** 2))))
        ax.plot(ct, rms, lw=1.0, label=f"{lab}")
    ax.set_xlabel("时间 / s"); ax.set_ylabel("1 s 滑窗 RMS / µm")
    ax.set_title("Y 向幅值（滑窗 RMS，抑制效果随时间）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_amp_sliding.png", dpi=130); plt.close(fig)

    # fig3 PSD + profile 带
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for lab, rd in zip(labels, run_dirs):
        t_u, ey, fs = _uni(rd)
        f, p = spectral.welch_psd(ey, fs, win_s=WELCH_WIN_S)
        ax.semilogy(f, p, lw=1.0, label=f"{lab}")
    ax.axvspan(slow_lo, slow_hi, color="C4", alpha=0.10, label=f"慢成分 {slow_lo:.1f}-{slow_hi:.1f} Hz")
    for lo, hi in bands:
        ax.axvspan(lo, hi, color="C2", alpha=0.14)
    for lab, rd in zip(labels, run_dirs):
        mm = metrics[lab]
        ax.axvline(mm["vibration"]["dominant_hz"], lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(0, 45); ax.set_xlabel("频率 / Hz"); ax.set_ylabel("PSD / (µm²/Hz)")
    ax.set_title("e_y Welch 谱（绿=冻结振动带；紫=慢成分；点线=各自振动主导峰）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_psd.png", dpi=130); plt.close(fig)

    # fig4 频带 RMS 柱状（慢 + 各振动带，需成对才有可比性）
    if len(labels) < 2:
        return
    names = [f"慢{slow_lo:.0f}-{slow_hi:.0f}"] + [f"{lo:.0f}-{hi:.0f}" for lo, hi in bands]
    vals_a = [metrics[labels[0]]["slow"]["rms_um"]] + \
             [metrics[labels[0]]["vibration"]["band_rms_um"].get(
                 f"{i:02d}@{lo:.2f}-{hi:.2f}", 0.0) for i, (lo, hi) in enumerate(bands)]
    vals_b = [metrics[labels[1]]["slow"]["rms_um"]] + \
             [metrics[labels[1]]["vibration"]["band_rms_um"].get(
                 f"{i:02d}@{lo:.2f}-{hi:.2f}", 0.0) for i, (lo, hi) in enumerate(bands)]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - w / 2, vals_a, w, label=labels[0])
    ax.bar(x + w / 2, vals_b, w, label=labels[1])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMS / µm"); ax.set_title("慢成分与各冻结振动带 RMS（µm）")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_band_rms.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------- 汇总输出
def analyze_pair(run_a: Path, run_b: Path, labels: list[str], out_dir: Path,
                 strict: bool = True, profile_dir: Path | None = None) -> dict:
    """正式 A/B：门控 + 指标 + 图 + JSON/摘要。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gate = gate_pair(Path(run_a), Path(run_b), strict=strict)
    if strict and not gate["ok"]:
        raise ValueError("A/B 门控未通过：\n  - " + "\n  - ".join(gate["problems"]))
    packet = profile_dir or locate_packet(Path(run_a))
    profile = load_profile(packet)
    metrics = {labels[i]: analyze_run(rd, profile, labels[i])
               for i, rd in enumerate([Path(run_a), Path(run_b)])}
    ma, mb = metrics[labels[0]], metrics[labels[1]]
    comp = compare(ma, mb)
    comp["segment"] = {
        k: {labels[0]: ma["segment"].get(k), labels[1]: mb["segment"].get(k)}
        for k in set(ma["segment"]) | set(mb["segment"])}
    comp["gate"] = gate

    _plot_all([Path(run_a), Path(run_b)], labels, profile, out_dir / "figures",
              metrics, comp)
    result = {
        "version": "v1.2",
        "profile_sha256": provenance.sha256_file(packet / "spectral_profile.json"),
        "profile_file": str(packet / "spectral_profile.json"),
        "metrics": metrics,
        "comparison": comp,
        "labels": labels,
    }
    (out_dir / "analysis_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(out_dir, result)
    return result


def analyze_single(run_dir: Path, label: str, out_dir: Path) -> dict:
    """单组诊断（不要求成对）。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    packet = locate_packet(Path(run_dir))
    profile = load_profile(packet)
    mm = analyze_run(Path(run_dir), profile, label)
    _plot_all([Path(run_dir)], [label], profile, out_dir / "figures", {label: mm}, None)
    result = {"version": "v1.2",
              "profile_sha256": provenance.sha256_file(packet / "spectral_profile.json"),
              "profile_file": str(packet / "spectral_profile.json"),
              "metrics": {label: mm}, "comparison": {}, "labels": [label]}
    (out_dir / "analysis_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(out_dir, result)
    return result


def _write_summary(out_dir: Path, result: dict) -> None:
    lines = ["# APF-SFC 分析摘要 (V1.2)"]
    comp = result.get("comparison", {})
    lines.append(f"冻结谱 profile: {Path(result['profile_file']).name} "
                 f"(sha12={result['profile_sha256'][:12]}…)")
    for lab, mm in result["metrics"].items():
        lines.append(f"\n## {lab}  [{mm['mode']}]  completed={mm['completed']}")
        lines.append(f"  采样 n={mm['sample']['n_uniform']} @ {mm['sample']['fs_hz']} Hz "
                     f"[{mm['sample']['t_span_s'][0]}..{mm['sample']['t_span_s'][1]}] s "
                     f"({mm['sample']['resample']})")
        lines.append(f"  时域 rms={mm['time']['rms_um']:.2f} µm  ptp={mm['time']['ptp_um']:.1f} µm")
        lines.append(f"  慢成分[{mm['slow']['range_hz'][0]:.2f},{mm['slow']['range_hz'][1]:.2f}]Hz "
                     f"rms={mm['slow']['rms_um']:.2f} µm  主频(profile)="
                     f"{mm['slow']['profile_dominant_hz']}Hz  主频(仿真)={mm['slow']['sim_dominant_hz']}Hz")
        lines.append(f"  振动带 RMS(µm) 总={mm['vibration']['total_rms_um']:.2f}  "
                     f"主导峰={mm['vibration']['dominant_hz']}Hz")
        for k, v in mm["vibration"]["band_rms_um"].items():
            lines.append(f"    带 {k}: {v:.3f} µm")
        for k, seg in mm.get("segment", {}).items():
            lines.append(f"  分段[{k}]: {seg['n_windows']}窗 {seg['total_s']}s "
                         f"rms={seg['rms_um']:.2f} µm")
    if comp.get("vibration_total_ratio") is not None:
        lines.append("\n## 对比")
        lines.append(f"  振动总 RMS 抑制比(B/A)={comp['vibration_total_ratio']} "
                     f"(−{comp.get('vibration_total_suppression_pct', 0):.1f}%)")
        lines.append(f"  时域 RMS 抑制比(B/A)={comp.get('time_rms_ratio')}")
        lines.append(f"  慢成分 RMS 抑制比(B/A)={comp.get('slow_rms_ratio')}")
        for k, r in comp.get("band_ratio", {}).items():
            if r is not None:
                lines.append(f"  带 {k} 抑制比(B/A)={r}")
    if comp.get("gate") and not comp["gate"]["ok"]:
        lines.append("\n## 门控未过（非正式结果）")
        for p in comp["gate"]["problems"]:
            lines.append(f"  - {p}")
    (out_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="APF-SFC 离线分析 V1.2")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair", nargs=2, metavar=("RUN_A", "RUN_B"), help="A/B 两个 run 目录")
    g.add_argument("--run", metavar="RUN", help="单组诊断")
    p.add_argument("--labels", nargs="+", help="标签（缺省用目录名）")
    p.add_argument("--out", required=True, help="分析输出目录")
    p.add_argument("--no-strict", action="store_true", help="门控不通过也继续（标为非正式）")
    p.add_argument("--profile-dir", default="", help="显式指定含 spectral_profile.json 的数据包目录")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    try:
        profile_dir = Path(args.profile_dir) if args.profile_dir else None
        if args.pair:
            dirs = [Path(x) for x in args.pair]
            labels = args.labels or [d.name for d in dirs]
            res = analyze_pair(dirs[0], dirs[1], labels, Path(args.out),
                               strict=not args.no_strict, profile_dir=profile_dir)
        else:
            rd = Path(args.run)
            labels = args.labels or [rd.name]
            res = analyze_single(rd, labels[0], Path(args.out))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[analysis] 失败：{exc}", file=sys.stderr)
        return 1
    c = res.get("comparison", {})
    ratio = c.get("vibration_total_ratio")
    extra = f"  振动总 RMS 抑制比={ratio}" if ratio is not None else ""
    print(f"[analysis] 完成 -> {Path(args.out)}  labels={res['labels']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
