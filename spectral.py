"""
频谱公共工具（V1.2）：非均匀时间戳重采样、Welch PSD、频带 RMS。

给 make_profile.py（预实验原始数据，~132Hz 缺帧）与 analysis.py（仿真日志，
1000Hz 均匀或 132Hz 7/8ms 交替）共用，保证口径一致。

要点（原 R-008 / R-015）：
- 不做“median(diff)=8ms → 125Hz”的错判；按真实时间戳用“平均步长”重采样到均匀网格。
- 若已是严格均匀（1000Hz），重采样为 no-op（仅可能端点差一帧），保证 PSD 频率轴正确。
"""

from __future__ import annotations

import numpy as np


def resample_uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    把 (t,x) 重采样到均匀网格，返回 (t_u, x_u, fs_hz)。

    dt_ref = (t[-1]-t[0])/(N-1)（平均步长），fs = 1/dt_ref。
    线性插值（interp 文档与 profile 记录一致）。端点未覆盖部分夹紧。
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 3:
        return t, x, (1.0 / (t[-1] - t[0])) if t.size > 1 and t[-1] > t[0] else 1.0
    span = float(t[-1] - t[0])
    dt_ref = span / (t.size - 1)
    n = int(t.size)
    t_u = np.arange(n, dtype=float) * dt_ref + float(t[0])
    if np.allclose(np.diff(t), dt_ref * np.ones(t.size - 1), atol=dt_ref * 1e-6):
        x_u = x.copy()                # 本就均匀：保持原始（避免重复插值）
        t_u = t
    else:
        x_u = np.interp(t_u, t, x)
    fs = 1.0 / dt_ref if dt_ref > 0 else 1.0
    return t_u, x_u, float(fs)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """按样本索引线性去趋势。"""
    xx = np.arange(x.size, dtype=float)
    coef = np.polyfit(xx, x, 1)
    return x - np.polyval(coef, xx)


def welch_psd(x: np.ndarray, fs: float, win_s: float = 2.0,
              overlap: float = 0.5, nwin: int | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    平均周期图（Hann 窗），PSD 单边（除 DC/Nyquist 乘 2）。返回 (freq, psd)。
    窗长自动缩到 ≤ 数据一半，保证至少可分 2 段；极短数据退化为整段周期图。
    nwin 传入时强制用该窗样本数（用于让两条序列落到同一频率网格直接相减），
    仍钳制到 ≥4 且 ≥ 数据可容纳（不足时 nseg=1 并补零，退化为整段周期图）。
    """
    if nwin is None:
        nwin = max(2, int(round(win_s * fs)))
        if nwin > x.size // 2:
            nwin = x.size // 2
    else:
        nwin = max(4, int(nwin))
    if nwin > x.size:
        nwin = x.size
    if nwin < 4:
        nwin = x.size
    win = np.hanning(nwin)
    xd = x - x.mean()
    step = max(1, int(nwin * (1.0 - overlap)))
    nseg = max(1, 1 + (x.size - nwin) // step)
    psd = np.zeros(nwin // 2 + 1)
    for k in range(nseg):
        seg = xd[k * step: k * step + nwin]
        if seg.size < nwin:
            seg = np.pad(seg, (0, nwin - seg.size))
        sp = np.fft.rfft(seg * win)
        psd += (np.abs(sp) ** 2) / (np.sum(win ** 2) * fs)
    psd /= nseg
    psd[1:-1] *= 2.0
    freq = np.fft.rfftfreq(nwin, d=1.0 / fs)
    return freq, psd


def band_rms(freq: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """带内 PSD 积分开方得带内 RMS（psd 单位²/Hz → 返回单位²开方）。"""
    m = (freq >= lo) & (freq < hi)
    if not np.any(m):
        return 0.0
    return float(np.sqrt(max(float(np.trapezoid(psd[m], freq[m])), 0.0)))
