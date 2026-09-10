"""
频谱 profile 生成（V1.2）：用预实验原始数据识别并冻结“慢成分/振动带/主导峰”。

对应《DS 修改指南》§6 / R-014 / R-015：评价频带必须来自预实验数据（A 组侧），
不能由 B 组结果选带、不能用固定 5-40Hz、不能把插值到 1kHz 的网格误当视觉信息。

判据（rule-based，全部写入 json，换数据集必须重跑重冻结）：
- 阶段：pre_motion + post_stop → 静态噪声；X_outbound + X_return → 运动。
- 每阶段块先按时间线性去趋势（把慢偏置从谱里剔除会破坏慢瓣信息，故去趋势只用于
  “块内偏置”，慢瓣仍留在谱中：见 slow_component_range），块两端余弦淡入淡出后拼接，
  统一在平均步长 dt_ref 上做 Welch PSD（运动/静态共用窗 ⇒ 同一频率网格可直接相减）。
- 慢成分：慢瓣主峰 slow_peak = 1.5Hz 内运动谱全局最大（≈0.5–0.8Hz，真实预实验偏置漂移）。
  slow_top = (slow_peak, 4.0Hz] 内运动谱第一个“低于 slow_peak−2dB”的局部极小
  （慢瓣与首个振动簇之间的谷）；无则回退为运动谱跌破 slow_peak−6dB 的频率。
  slow_component_range_Hz = [0, slow_top]。
- 振动带：f ≥ slow_top、运动谱高于静态谱 +6dB、且高于运动谱峰值 −40dB 底噪的连续区间，
  带间隙 < 1 个频率分辨率自动合并；至少 2 个 bin 宽。
- 主导峰：振动区内运动谱局域极大按能量降序（另存单段 X_outbound/X_return 交叉核对，
  证明 3.9Hz 基频与 ~7/12.5 谐波在每段里可复现，非拼接假象）。

产物 outputs/wfit_P05R01/spectral_profile.json + fig_profile.png。
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
import wfit

DEFAULT_CSV = config.PROJECT_DIR / "data" / "P05R01" / "D1_时序.csv"
DEFAULT_OUT = config.PACKET_PROFILE_FILE

SLOW_PEAK_SEARCH_HZ = 1.5    # 慢瓣主峰搜索上界（慢偏置主导频率必在亚Hz~1Hz 量级）
SLOW_SCAN_MAX_HZ = 4.0       # slow_top 谷搜索上界（超过即进入首个振动簇，不允许再判为“慢”）
VALLEY_DEPTH_DB = 2.0        # 谷深：相对 slow_peak 至少低 2dB 才算真谷
VALLEY_FALLBACK_DB = 6.0     # 无真谷时回退：跌破 slow_peak−6dB
FLOOR_DB_BELOW_PEAK = 40.0   # 底噪 = 运动谱峰值 −40 dB
EDGE_FADE_FRAC = 0.12        # 每阶段块两端淡入淡出比例（拼接防台阶）


def _phase_blocks(raw: dict) -> dict[str, list[tuple[int, int]]]:
    """每阶段名的连续样本区间列表（索引域，按原始行顺序）。"""
    blocks: dict[str, list[tuple[int, int]]] = {}
    pc = raw["phase_code"]
    names = raw["phase_names"]
    for nm in names:
        idx = np.where(pc == names.index(nm))[0]
        if idx.size == 0:
            continue
        split = np.where(np.diff(idx) > 1)[0] + 1
        for s in np.split(idx, split):
            blocks.setdefault(nm, []).append((int(s[0]), int(s[-1])))
    return blocks


def _class_series(raw: dict, phases: list[str], blocks: dict, dt_ref: float) -> np.ndarray:
    """把指定阶段的若干连续块（各自线性去趋势+两端淡入淡出）拼接成一条均匀序列。"""
    out: list[np.ndarray] = []
    t = raw["t"]; cr = raw["cross_um"]
    for nm in phases:
        for b0, b1 in blocks.get(nm, []):
            tb = t[b0:b1 + 1]; xb = cr[b0:b1 + 1]
            if xb.size < 4:
                continue
            xd = spectral.detrend_linear(xb)
            fade = max(2, int(xb.size * EDGE_FADE_FRAC))
            w = np.ones(xb.size)
            w[:fade] *= np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            w[-fade:] *= np.sin(np.linspace(np.pi / 2, 0, fade)) ** 2
            _, xr, _ = spectral.resample_uniform(tb, xd * w)
            out.append(xr)
    if not out:
        raise ValueError("无可用阶段样本（检查 phase 列/阶段名）")
    return np.concatenate(out)


def _local_peaks(freq: np.ndarray, psd: np.ndarray) -> list[float]:
    """按 PSD 能量从高到低排序的局域极大频率。"""
    pk: list[tuple[float, float]] = []
    for i in range(1, psd.size - 1):
        if psd[i] >= psd[i - 1] and psd[i] > psd[i + 1]:
            pk.append((float(freq[i]), float(psd[i])))
    return [f for f, _ in sorted(pk, key=lambda ab: ab[1], reverse=True)]


def _slow_top_hz(freq: np.ndarray, m_db: np.ndarray, res: float) -> tuple[float, float]:
    """
    返回 (slow_peak, slow_top)：慢瓣主峰频率 + 慢瓣与首个振动簇之间的谷频率。
    """
    in_slow = freq < SLOW_PEAK_SEARCH_HZ
    if not in_slow.any():
        raise ValueError("无 <1.5Hz 的慢瓣（数据异常，无法切分慢成分）")
    i0 = int(np.argmax(np.where(in_slow, m_db, -np.inf)))
    slow_peak = float(freq[i0])

    best: int | None = None
    imax = min(len(freq) - 1, int(round(SLOW_SCAN_MAX_HZ / res)))
    for i in range(i0 + 1, imax):
        if m_db[i - 1] >= m_db[i] and m_db[i] < m_db[i + 1] and m_db[i] <= m_db[i0] - VALLEY_DEPTH_DB:
            best = i
            break
    if best is None:
        thr = m_db[i0] - VALLEY_FALLBACK_DB
        below = np.where(m_db[i0:] < thr)[0]
        best = i0 + int(below[0]) if below.size else len(freq) - 1
    return slow_peak, float(freq[best])


def _stroke_spectra(raw: dict, blocks: dict) -> dict:
    """单段去趋势+重采样谱的主峰（交叉核对振动基频非拼接假象）。"""
    out: dict[str, list[float]] = {}
    t = raw["t"]
    for nm in ("X_outbound", "X_return"):
        b0, b1 = blocks.get(nm, [(-1, -1)])[0]
        if b0 < 0:
            continue
        xb = raw["cross_um"][b0:b1 + 1]
        _, xr, fs = spectral.resample_uniform(t[b0:b1 + 1], spectral.detrend_linear(xb))
        f, p = spectral.welch_psd(xr, fs, win_s=1.5)
        pk = [x for x in _local_peaks(f, p) if x >= 1.5]
        out[nm] = [round(x, 3) for x in pk[:6]]
    return out


def build_profile(csv_path: Path, out_json: Path | None = None, win_s: float = 2.0,
                  rel_db: float = 6.0) -> dict:
    """读原始 CSV、估计谱、识别慢/振动、写 json（并画 fig_profile.png）。返回完整 dict。"""
    raw = wfit.read_raw_d1(csv_path)
    blocks = _phase_blocks(raw)
    for nm in ("pre_motion", "post_stop", "X_outbound", "X_return"):
        if nm not in blocks:
            raise ValueError(f"原始数据缺少阶段 {nm}（现有 {list(blocks)}）")

    span = float(raw["t"][-1] - raw["t"][0])
    dt_ref = span / (raw["t"].size - 1)
    fs = 1.0 / dt_ref

    motion = _class_series(raw, ["X_outbound", "X_return"], blocks, dt_ref)
    static = _class_series(raw, ["pre_motion", "post_stop"], blocks, dt_ref)

    nwin_shared = max(4, int(round(win_s * fs)))
    nwin_shared = min(nwin_shared, min(motion.size, static.size))
    f_m, p_m = spectral.welch_psd(motion, fs, win_s=win_s, nwin=nwin_shared)
    f_s, p_s = spectral.welch_psd(static, fs, win_s=win_s, nwin=nwin_shared)
    if f_m.size != f_s.size or not np.allclose(f_m, f_s):
        raise RuntimeError("运动/静态 PSD 网格不一致")

    res = float(f_m[1] - f_m[0])
    eps = np.finfo(float).eps
    m_db = 10 * np.log10(p_m + eps)
    s_db = 10 * np.log10(p_s + eps)
    floor_db = float(np.max(m_db)) - FLOOR_DB_BELOW_PEAK

    slow_peak, slow_top = _slow_top_hz(f_m, m_db, res)
    ok = (f_m >= slow_top) & (m_db - s_db >= rel_db) & (m_db >= floor_db)

    bands: list[list[float]] = []
    in_band = False
    lo = 0
    for i, flag in enumerate(ok):
        if flag and not in_band:
            lo = i; in_band = True
        elif not flag and in_band:
            bands.append([float(f_m[lo]), float(f_m[i - 1])])
            in_band = False
    if in_band:
        bands.append([float(f_m[lo]), float(f_m[-1])])
    merged: list[list[float]] = []
    for b in bands:
        if merged and (b[0] - merged[-1][1]) < res:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    bands = [b for b in merged if int(round((b[1] - b[0]) / res)) >= 1]  # 至少 2 bin 宽

    peaks_all = _local_peaks(f_m, p_m)
    vib_peaks = [p for p in peaks_all if p >= slow_top]

    doc = {
        "method": "pre-experiment spectral profile (A-side only)",
        "source_file": str(csv_path),
        "source_sha256": provenance.sha256_file(csv_path),
        "data_id": _guess_data_id(csv_path),
        "stages_analyzed": {
            "static_phases": ["pre_motion", "post_stop"],
            "motion_phases": ["X_outbound", "X_return"],
            "phase_time_ranges_s": {nm: [[float(raw["t"][b0]), float(raw["t"][b1])]
                                         for b0, b1 in bl] for nm, bl in blocks.items()},
        },
        "sampling": raw["sampling"],
        "resample": {"target_fs_hz": round(fs, 3), "dt_ref_s": round(dt_ref, 9),
                     "interp": "linear"},
        "welch": {
            "window_s": win_s,
            "nwin_shared_samples": int(nwin_shared),
            "window_used_s": round(nwin_shared / fs, 5),
            "frequency_resolution_Hz": round(res, 5),
            "overlap": 0.5, "window_type": "hann",
        },
        "rule": {
            "threshold_db_over_static": rel_db,
            "floor_db_below_motion_peak": FLOOR_DB_BELOW_PEAK,
            "merge_gap_s": round(res, 5),
            "valley_depth_db": VALLEY_DEPTH_DB,
            "description": "振动带=运动谱比静态谱高+6dB 且高于峰值-40dB 底噪的连续区间，"
                           "间隙<1 分辨率自动合并；慢成分=慢瓣峰到首振动簇间谷",
        },
        "slow_component_range_Hz": [0.0, round(slow_top, 5)],
        "slow_bias_dominant_Hz": round(slow_peak, 5),
        "identified_vibration_bands_Hz": [[round(b[0], 5), round(b[1], 5)] for b in bands],
        "dominant_peaks_Hz": [round(p, 5) for p in vib_peaks[:8]],
        "dominant_peak_list_Hz": [round(p, 5) for p in vib_peaks[:12]],
        "stroke_peak_crosscheck": _stroke_spectra(raw, blocks),
    }
    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        _plot(out_json.parent / "fig_profile.png", f_m, m_db, s_db, doc, rel_db)
    return doc


def _guess_data_id(csv_path: Path) -> str:
    parent = Path(csv_path).parent.name.upper()
    return parent if parent else "unknown"


def _plot(path: Path, f: np.ndarray, m_db: np.ndarray, s_db: np.ndarray,
          doc: dict, rel_db: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    slow_hi = doc["slow_component_range_Hz"][1]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(f, m_db, lw=1.0, label="运动谱 (X_outbound+X_return)")
    ax.plot(f, s_db, lw=1.0, color="gray", ls="--", label="静态噪声谱 (pre+post)")
    ax.axhline(float(np.max(m_db)) - 40.0, color="k", lw=0.6, alpha=0.5, ls=":")
    ax.axvspan(0.0, slow_hi, color="C4", alpha=0.12,
               label=f"慢成分 [0, {slow_hi:.2f}]Hz")
    for lo, hi in doc["identified_vibration_bands_Hz"]:
        ax.axvspan(lo, hi, color="C2", alpha=0.15)
    for pk in doc["dominant_peak_list_Hz"]:
        ax.axvline(pk, color="C3", lw=0.6, alpha=0.6)
    ax.axvline(doc["slow_bias_dominant_Hz"], color="C1", lw=1.2, ls="--")
    ax.set_xlim(0, max(25.0, res := doc["welch"]["frequency_resolution_Hz"] * 90))
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("PSD / dB")
    ax.set_title(f"预实验频谱 profile P05R01（慢成分=紫；振动带=绿；主导峰=红细线；"
                 f"慢瓣主峰=黄虚线；+{rel_db:.0f}dB 判据）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="预实验数据 -> 冻结 spectral_profile.json")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="原始 D1_时序.csv")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="spectral_profile.json")
    p.add_argument("--window-s", type=float, default=2.0, help="Welch 窗长 s")
    p.add_argument("--rel-db", type=float, default=6.0, help="相对静态噪声的 dB 阈值")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        doc = build_profile(Path(args.csv), Path(args.out),
                            win_s=args.window_s, rel_db=args.rel_db)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[profile] 失败：{exc}", file=sys.stderr)
        return 1
    print(f"[profile] 源 sha12={doc['source_sha256'][:12]}…")
    print(f"[profile] 慢成分 {doc['slow_component_range_Hz']} Hz（慢瓣主峰 "
          f"{doc['slow_bias_dominant_Hz']} Hz）")
    print(f"[profile] 振动带 {doc['identified_vibration_bands_Hz']} Hz")
    print(f"[profile] 主导峰 {doc['dominant_peaks_Hz'][:6]} Hz")
    print(f"[profile] 单段交叉核对 {doc['stroke_peak_crosscheck']}")
    print(f"[profile] 完成 -> {Path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
