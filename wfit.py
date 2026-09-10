"""
真实预实验数据 -> MuJoCo 等效 Y 扰动力 w(t) 的拟合核心（wfit = w fit）。

V1.2 定位（《DS 修改指南》§5.2）：一次性数据准备工具，不是日常 run 的一部分。
- 输入只能来自明确指定的预实验 CSV；切窗规则升级为显式 dataset profile（P05R01），
  条件未命中必须报错，不再伪装成通用自动算法（原 R-017）。
- 提供原始时间戳读取（read_raw_d1），供频谱 profile 生成（不把 ~132Hz 缺帧误当 1kHz，
  原 R-015）。
- 植物缓存写入模型/物理/姿态指纹，不一致自动重测（原 R-007）。
- 主频口径区分“慢偏置主频(全谱)”与“振动主频(1..45Hz)”，不再把方向变化 0.02Hz 当振动
  主频（原 R-014）。

仍保留：Plant 频响标定（sum-of-sines 前增加稳定段，丢弃初始伺服瞬态，原 R-016）、
单程拉长模板、带限反演。命令入口在 fit_w.py。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import config
import provenance

# ---------------------------------------------------------------------------
# 1. 数据集 profile（显式规则；条件未命中报错，不伪装通用自动切窗）
# ---------------------------------------------------------------------------
DATASET_PROFILE_NAME: str = "P05R01"
DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "P05R01": {
        "phase_order": ["pre_motion", "X_outbound", "far_end_turn", "X_return", "post_stop"],
        # iA 启动段结束（进入前向稳态）：前向沿程 ≥ 3.2 mm
        "iA_along_mm": 3.2,
        # iB 前向稳态结束（开始刹车）：沿程 ≥ 11.0 mm
        "iB_along_mm": 11.0,
        # iP 换向顶点：前向沿程最大
        # iC 返程瞬态结束：X_return 前 1.3 s 的最低点之后回升越过 -45 µm
        "iC_search_s": 1.3,
        "iC_cross_um": -45.0,
        # iD 返程稳态结束：返程速度最后一次 < -3.5 mm/s（开始减速停）
        "iD_speed_mm_s": -3.5,
    },
}


# ---------------------------------------------------------------------------
# 2. 读真实数据：原始时间戳 + 1ms 网格
# ---------------------------------------------------------------------------
GRID_DT_S = 0.001
_COL_TIME, _COL_ALONG, _COL_CROSS, _COL_SPEED, _COL_PHASE = (
    "time_s", "nominal_along_mm", "cross_track_um", "vision_speed_mm_s", "phase")


def _decode_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """返回 (表头列名, 数据行列表)；自动尝试 utf-8-sig / utf-8 / gbk。"""
    import csv
    raw_rows = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                raw_rows = list(csv.reader(f))
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if not raw_rows:
        raise ValueError(f"无法解码 {path}")
    hdr = [h.replace("﻿", "").strip() for h in raw_rows[0]]
    idx: dict[str, int] = {}
    for want in (_COL_TIME, _COL_ALONG, _COL_CROSS, _COL_SPEED, _COL_PHASE):
        hit = [i for i, h in enumerate(hdr) if h == want]
        if not hit:
            raise ValueError(f"{path} 缺列 {want}（表头 {hdr[:8]}…）")
        idx[want] = hit[0]
    rows = [r for r in raw_rows[1:] if len(r) > max(idx.values()) and r[idx[_COL_TIME]]]
    return idx, rows


def _coerce_col(rows: list[list[str]], i: int) -> np.ndarray:
    return np.array([float(r[i]) for r in rows])


def read_raw_d1(path: Path) -> dict[str, Any]:
    """
    读原始行（不做重采样），返回与原始采样对齐的数组与采样统计。

    返回 dict：t/along_mm/cross_um/speed_mm_s（与原始行对齐）、phase_names、phase_code、
    sampling={n, dt_median_s, dt_mean_s, dt_min_s, dt_max_s, gap_pct}。
    采样统计用原始 diff 直接算；gap_pct = 超过 1.5×中位间隔的帧占比（=丢帧率上界）。
    """
    idx, rows = _decode_rows(path)
    t = _coerce_col(rows, idx[_COL_TIME])
    along = _coerce_col(rows, idx[_COL_ALONG])
    cross = _coerce_col(rows, idx[_COL_CROSS])
    speed = _coerce_col(rows, idx[_COL_SPEED])
    phase_str = [r[idx[_COL_PHASE]].strip() for r in rows]
    phase_names = list(dict.fromkeys(phase_str))
    code_map = {n: i for i, n in enumerate(phase_names)}
    phase_code = np.array([code_map[p] for p in phase_str])

    dt = np.diff(t)
    med = float(np.median(dt)) if dt.size else float("nan")
    gap_pct = 100.0 * float(np.mean(dt > 1.5 * med)) if (dt.size and med > 0) else 0.0
    sampling = {
        "n": int(t.size),
        "dt_median_s": round(med, 9),
        "dt_mean_s": round(float(np.mean(dt)), 9) if dt.size else None,
        "dt_min_s": round(float(np.min(dt)), 9) if dt.size else None,
        "dt_max_s": round(float(np.max(dt)), 9) if dt.size else None,
        "gap_pct": round(gap_pct, 2),
        "origin": "raw camera frames (~132 Hz, 含丢帧)，未重采样成 1 kHz",
    }
    return {"t": t, "along_mm": along, "cross_um": cross, "speed_mm_s": speed,
            "phase_names": phase_names, "phase_code": phase_code, "sampling": sampling}


def read_d1(path: Path) -> dict[str, np.ndarray]:
    """读 D1_时序.csv 并重采样到 1ms 网格（供切窗/拼接/FFT 用）。"""
    raw = read_raw_d1(path)
    t0, t1 = float(raw["t"][0]), float(raw["t"][-1])
    tg = np.arange(t0, t1 + GRID_DT_S / 2, GRID_DT_S)
    gi = np.clip(np.searchsorted(raw["t"], tg, side="right") - 1, 0, len(raw["t"]) - 1)
    return {
        "t": tg,
        "phase_names": raw["phase_names"],
        "phase_code": raw["phase_code"][gi],
        "along_mm": np.interp(tg, raw["t"], raw["along_mm"]),
        "cross_um": np.interp(tg, raw["t"], raw["cross_um"]),
        "speed_mm_s": np.interp(tg, raw["t"], raw["speed_mm_s"]),
    }


def _phase_window(d: dict, name: str) -> tuple[int, int]:
    if name not in d["phase_names"]:
        raise ValueError(f"数据缺少阶段 {name}（现有 {d['phase_names']}）")
    pc = d["phase_code"]
    code = d["phase_names"].index(name)
    hit = np.where(pc == code)[0]
    if hit.size == 0:
        raise ValueError(f"阶段 {name} 无样本")
    return int(hit[0]), int(hit[-1])


def detect_cut_points(d: dict[str, np.ndarray], profile_name: str | None = None) -> dict[str, int]:
    """
    按显式 dataset profile 切出模板窗口索引（网格域，按时间顺序）：
      iA 启动段结束(进前向稳态)  iB 前向稳态结束(开始刹车)
      iP 远端换向顶点           iC 返程瞬态结束(回稳态)
      iD 返程稳态结束(开始减速停)
    任何阈值条件不命中即报错（不再 argmax(False) 静默返回 0）。
    """
    profile_name = profile_name or DATASET_PROFILE_NAME
    if profile_name not in DATASET_PROFILES:
        raise ValueError(f"未知 dataset profile：{profile_name}（有 {list(DATASET_PROFILES)}）")
    prof = DATASET_PROFILES[profile_name]
    al = d["along_mm"]; cr = d["cross_um"]; sp = d["speed_mm_s"]

    def _hit(cond: np.ndarray, where: str, how: str) -> int:
        hits = np.where(cond)[0]
        if hits.size == 0:
            raise ValueError(f"[profile={profile_name}] {where} 未命中 {how}（数据不满足该数据集规则）")
        return int(hits[0] if how.startswith("first") else hits[-1])

    i_o0, i_o1 = _phase_window(d, "X_outbound")
    i_r0, i_r1 = _phase_window(d, "X_return")
    o = slice(i_o0, i_o1 + 1)
    iA = _hit(al[o] >= prof["iA_along_mm"], "X_outbound",
              f"first(沿程≥{prof['iA_along_mm']}mm)") + i_o0
    iB = _hit(al[o] >= prof["iB_along_mm"], "X_outbound",
              f"first(沿程≥{prof['iB_along_mm']}mm)") + i_o0
    iP = _hit(al[o] == np.max(al[o]), "X_outbound", "last(沿程最大=换向顶点)") + i_o0

    seg = cr[i_r0:i_r1 + 1]
    lim = min(seg.size, int(prof["iC_search_s"] / GRID_DT_S) + 1)
    kmin = int(np.argmin(seg[:lim]))                    # X_return 开头 1.3s 的最低点
    up = np.where(seg[kmin:] >= prof["iC_cross_um"])[0]
    if up.size == 0:
        raise ValueError(f"[profile={profile_name}] 返程未回升越过 {prof['iC_cross_um']}µm")
    iC = i_r0 + kmin + int(up[0])

    mv = np.where(sp[i_r0:i_r1 + 1] < prof["iD_speed_mm_s"])[0]
    if mv.size == 0:
        raise ValueError(f"[profile={profile_name}] 返程无 <{prof['iD_speed_mm_s']}mm/s 的全速段")
    iD = i_r0 + int(mv[-1])
    if not (iA < iB < iP < iC < iD):
        raise ValueError(f"[profile={profile_name}] 切窗顺序异常 iA..iD={iA},{iB},{iP},{iC},{iD}")
    return {"iA": iA, "iB": iB, "iP": iP, "iC": iC, "iD": iD}


# ---------------------------------------------------------------------------
# 3. “单程拉长”模板
# ---------------------------------------------------------------------------
SEAM_SMOOTH_S = 0.10        # 拼接缝平滑半窗（s，两侧各）
DEFAULT_DURATION_S = 60.0


@dataclass
class Template:
    t: np.ndarray           # s（均匀网格 dt=GRID_DT_S）
    along_mm: np.ndarray    # 从 0 起的沿程，mm（先 +X 到远端再回 0）
    e_um: np.ndarray        # 期望横向偏差模板，µm
    segments: list[tuple[str, int, int]] = field(default_factory=list)
    seam_jumps_um: list[float] = field(default_factory=list)
    # 溯源：每个稳态窗来自哪个原片段时间范围（s），便于元数据记录
    source_windows: list[dict[str, Any]] = field(default_factory=list)
    n_forward_repeat: int = 1
    n_return_repeat: int = 0


def _smooth_seams(x: np.ndarray, seams: list[int], half: int) -> np.ndarray:
    out = x.copy()
    n = len(x)
    for s in seams:
        a = max(0, s - half); b = min(n - 1, s + half)
        if b - a < 4:
            continue
        line = np.linspace(out[a], out[b], b - a + 1)
        d = np.linspace(0.0, 1.0, b - a + 1)
        w = np.sin(np.pi * d) ** 2
        out[a:b + 1] = (1 - w) * out[a:b + 1] + w * line
    return out


def build_template(d: dict[str, np.ndarray], n_stroke: int | None = None,
                   duration_s: float = DEFAULT_DURATION_S,
                   profile_name: str | None = None) -> Template:
    """拼“单程拉长”模板（切窗规则来自 detect_cut_points 的 dataset profile）。"""
    t = d["t"]; al = d["along_mm"]; cr = d["cross_um"]
    c = detect_cut_points(d, profile_name)
    iA, iB, iP, iC, iD = c["iA"], c["iB"], c["iP"], c["iC"], c["iD"]
    iE = len(t) - 1
    hh = int(round(SEAM_SMOOTH_S / GRID_DT_S))

    def unit(sl: slice) -> tuple[np.ndarray, np.ndarray]:
        dd = al[sl] - al[sl.start]
        return dd.copy(), cr[sl].copy()

    dd0, ce0 = unit(slice(0, iA + 1))
    ddu, ceu = unit(slice(iA, iB + 1))       # 前向稳态 UF
    dda, cea = unit(slice(iB, iP + 1))       # 远端刹车->换向顶点
    ddb, ceb = unit(slice(iP, iC + 1))       # 换向返程瞬态
    ddr, cer = unit(slice(iC, iD + 1))       # 返程稳态 UR
    dde, cee = unit(slice(iD, iE + 1))       # 减速+停止+静止

    launch_d = float(dd0[-1])
    duf = float(ddu[-1])
    fwd_turn = float(dda[-1])
    turn_ret = float(-ddb[-1])
    ur_drop = float(-ddr[-1])
    stop_drop = float(-dde[-1])

    def assemble(n: int) -> tuple[np.ndarray, np.ndarray, list, list[int], int, float, float]:
        Lf = launch_d + duf * n + fwd_turn
        need_ur = max((Lf - turn_ret) - stop_drop, 0.0)
        m_full = int(need_ur // ur_drop)
        rem = need_ur - m_full * ur_drop
        segs: list[tuple[str, int, int]] = []
        seams: list[int] = []
        al_parts: list[np.ndarray] = []
        e_parts: list[np.ndarray] = []
        pos = 0

        def place(name: str, dloc: np.ndarray, e: np.ndarray, base: float, seam: bool) -> float:
            nonlocal pos
            if seam and al_parts:
                seams.append(pos)
            al_parts.append(base + dloc)
            e_parts.append(e)
            segs.append((name, pos, pos + len(dloc)))
            pos += len(dloc)
            return float(base + dloc[-1])

        base = 0.0
        base = place("start", dd0, ce0, base, seam=False)
        for k in range(n):
            base = place("mid_fwd", ddu, ceu, base, seam=(k >= 1))
        base = place("turn_fwd", dda, cea, base, seam=False)
        base = place("turn_ret", ddb, ceb, base, seam=False)
        for k in range(m_full):
            base = place("mid_ret", ddr, cer, base, seam=(k >= 1))
        if rem > 1e-6 and len(ddr) > 10:
            take = int(np.clip(int(np.searchsorted(np.abs(ddr), rem)) + 1, 2, len(ddr)))
            base = place("mid_ret_last", ddr[:take], cer[:take], base, seam=True)
        base = place("stop", dde, cee, base, seam=False)
        along = np.concatenate(al_parts)
        e = np.concatenate(e_parts)
        e = _smooth_seams(e, seams, hh)
        return along, e, segs, seams, n, m_full, along.size * GRID_DT_S

    def total_for(n: int) -> float:
        return assemble(n)[-1]

    if n_stroke is None:
        n_stroke = 1
        while total_for(n_stroke) < duration_s - 1e-6:
            n_stroke += 1
    along, e, segs, seams, nf, nr, dur = assemble(n_stroke)
    # 溯源：记录每个稳态窗来源（原片段时间范围 s）
    src = [
        {"window": "start",   "src_s": [float(t[0]), float(t[iA])]},
        {"window": "mid_fwd", "src_s": [float(t[iA]), float(t[iB])],
         "repeat": nf},
        {"window": "turn",    "src_s": [float(t[iB]), float(t[iC])]},
        {"window": "mid_ret", "src_s": [float(t[iC]), float(t[iD])],
         "repeat": nr},
        {"window": "stop",    "src_s": [float(t[iD]), float(t[iE])]},
    ]
    return Template(
        t=np.arange(along.size, dtype=float) * GRID_DT_S,
        along_mm=along, e_um=e, segments=segs,
        seam_jumps_um=[float(abs(e[i] - e[i - 1])) for i in seams if 0 < i < len(e)],
        source_windows=src, n_forward_repeat=nf, n_return_repeat=nr,
    )


# ---------------------------------------------------------------------------
# 4. Plant 标定与缓存指纹
# ---------------------------------------------------------------------------
PROBE_DURATION_S = 60.0
PROBE_FMAX_HZ = 35.0
PROBE_AMP_N = 0.05
PLANT_CACHE = config.OUTPUT_ROOT / "plant_cache.npz"
_PLANT_SETTLE_STEPS = 300          # 扫频前稳定段（丢弃初始伺服瞬态，R-016）


def _plant_fingerprint(params: dict[str, Any]) -> str:
    """植物缓存键：模型/初始姿态/物理频率/探针常量任一变化都应重测。"""
    m = provenance.model_fingerprint()
    key = {
        "model": {k: v["sha256"] for k, v in m.items()},
        "init_q": list(config.INIT_Q),
        "physics_hz": int(params["physics_hz"]),
        "probe_dur_s": PROBE_DURATION_S, "probe_fmax_hz": PROBE_FMAX_HZ,
        "probe_amp_n": PROBE_AMP_N, "settle_steps": _PLANT_SETTLE_STEPS,
    }
    return provenance.json_fingerprint(key)


@dataclass
class Plant:
    freq_hz: np.ndarray
    gain: np.ndarray               # 复增益 [µm/N]
    dc_um_per_n: float

    def interp(self, f: float) -> complex:
        if f <= 0:
            return complex(self.dc_um_per_n)
        fs = np.asarray(self.freq_hz); g = np.asarray(self.gain)
        return complex(float(np.interp(f, fs, g.real)) + 1j * float(np.interp(f, fs, g.imag)))

    def at(self, freqs: np.ndarray) -> np.ndarray:
        f = np.asarray(freqs, dtype=float)
        fs = np.asarray(self.freq_hz)
        real = np.interp(f, fs, self.gain.real)
        imag = np.interp(f, fs, self.gain.imag)
        dc = f <= 0.0
        real[dc] = self.dc_um_per_n; imag[dc] = 0.0
        return real + 1j * imag


def _schroeder_phase(k: int, n: int) -> float:
    return 0.0 if k == 0 else (np.pi * k * (k - 1) / n if n else 0.0)


def measure_plant(params: dict[str, Any] | None = None, cache: Path = PLANT_CACHE) -> Plant:
    """sum-of-sines 标定侧向频响并缓存；缓存指纹不一致自动重测（R-007）。"""
    cache = Path(cache)
    if params is None:
        params = config.parameter_defaults()
    fp = _plant_fingerprint(params)

    if cache.is_file():
        z = np.load(str(cache))
        if "fp" in z and str(z["fp"]) == fp:
            p = Plant(freq_hz=z["freq"], gain=z["gain"], dc_um_per_n=float(z["dc"]))
            print(f"[wfit] plant 读缓存（指纹一致，{p.freq_hz.size} 点，DC={p.dc_um_per_n:.1f} um/N）")
            return p
        print("[wfit] plant 缓存指纹不一致，重测（模型/姿态/物理频率/探针常量变化）")

    from simenv import SimEnv
    dt = 1.0 / float(params["physics_hz"])
    n = int(round(PROBE_DURATION_S / dt))
    tr = PROBE_DURATION_S
    kmax = int(round(PROBE_FMAX_HZ * tr))
    freq_bins = np.arange(1, kmax + 1) / tr
    ph = np.array([_schroeder_phase(k, kmax + 1) for k in range(1, kmax + 1)])
    amp = PROBE_AMP_N
    tt = np.arange(n, dtype=float) * dt
    force = np.zeros(n)
    blk = 500
    for s in range(0, freq_bins.size, blk):
        fb = freq_bins[s:s + blk, None]; pb = ph[s:s + blk, None]
        force += np.sum(amp * np.sin(2 * np.pi * fb * tt[None, :] + pb), axis=0)

    env = SimEnv(params)
    env.reset(config.INIT_Q)
    for _ in range(_PLANT_SETTLE_STEPS):       # 稳定段：托重力并收敛，丢弃瞬态
        env.step()
    y0 = float(env.data.site_xpos[env.site_id, 1])
    yum = np.zeros(n)
    for i in range(n):
        env.apply_force_world(np.array([0.0, force[i], 0.0]))
        env.step()
        yum[i] = (env.data.site_xpos[env.site_id, 1] - y0) * 1e6

    Y = np.fft.rfft(yum); F = np.fft.rfft(force)
    freqs = np.fft.rfftfreq(n, d=dt)
    g = np.full_like(freqs, np.nan, dtype=complex)
    kidx = (freqs * tr + 0.5).astype(int)
    valid = (kidx >= 1) & (kidx <= kmax)
    g[valid] = Y[valid] / F[valid]
    dc_um_per_n = _measure_dc(params)

    keep = ~np.isnan(g)
    p = Plant(freq_hz=freqs[keep], gain=g[keep], dc_um_per_n=dc_um_per_n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache), freq=p.freq_hz, gain=p.gain, dc=dc_um_per_n,
             probe_dur=PROBE_DURATION_S, probe_fmax=PROBE_FMAX_HZ, fp=fp)
    print(f"[wfit] 标定 plant：{p.freq_hz.size} 个 bin（{p.freq_hz[0]:.3f}..{p.freq_hz[-1]:.1f} Hz），"
          f"DC={dc_um_per_n:.1f} um/N -> {cache}")
    return p


def _measure_dc(params: dict[str, Any]) -> float:
    """静柔度：1 N Y 恒力稳定后的 TCP Y 偏移 (µm/N)。"""
    from simenv import SimEnv
    dt = 1.0 / float(params["physics_hz"])
    env = SimEnv(params)
    env.reset(config.INIT_Q)
    for _ in range(500):
        env.step()
    y0 = float(env.data.site_xpos[env.site_id, 1])
    n = int(round(1.0 / dt))
    for _ in range(n):
        env.apply_force_world(np.array([0.0, 1.0, 0.0]))
        env.step()
    y1 = float(env.data.site_xpos[env.site_id, 1])
    return float((y1 - y0) * 1e6)


# ---------------------------------------------------------------------------
# 5. 带限反演 与 主频
# ---------------------------------------------------------------------------
INV_F_PASS_HZ = 11.0
INV_F_STOP_HZ = 16.0


def _cos_taper(freq_hz: np.ndarray, f_pass: float, f_stop: float) -> np.ndarray:
    taper = np.ones_like(freq_hz)
    up = (freq_hz > f_pass) & (freq_hz < f_stop)
    if np.any(up):
        taper[up] = 0.5 * (1.0 + np.cos(np.pi * (freq_hz[up] - f_pass) / (f_stop - f_pass)))
    taper[freq_hz >= f_stop] = 0.0
    return taper


def invert_series(x_um: np.ndarray, plant: Plant, f_pass_hz: float = INV_F_PASS_HZ,
                  f_stop_hz: float = INV_F_STOP_HZ, dt_s: float = GRID_DT_S) -> np.ndarray:
    x = np.asarray(x_um, dtype=float)
    n = int(x.size)
    Ef = np.fft.rfft(x)
    freq = np.fft.rfftfreq(n, d=dt_s)
    G = plant.at(freq)
    taper = _cos_taper(freq, f_pass_hz, f_stop_hz)
    W = np.zeros_like(Ef, dtype=complex)
    ok = (taper > 0.0) & (np.abs(G) > 1e-6)
    W[ok] = (Ef[ok] / G[ok]) * taper[ok]
    return np.fft.irfft(W, n=n)


def invert_to_force(T: Template, plant: Plant, f_pass_hz: float = INV_F_PASS_HZ,
                    f_stop_hz: float = INV_F_STOP_HZ) -> tuple[np.ndarray, dict[str, float]]:
    e = np.asarray(T.e_um, dtype=float)
    n = int(e.size)
    w = invert_series(e, plant, f_pass_hz, f_stop_hz)
    meta = {
        "n_samples": int(n), "dur_s": round(float(n * GRID_DT_S), 4),
        "f_pass_hz": f_pass_hz, "f_stop_hz": f_stop_hz,
        "w_rms_N": float(np.sqrt(np.mean(w ** 2))),
        "w_ptp_N": float(np.ptp(w)),
        "w_peak_abs_N": float(np.max(np.abs(w))),
        "e_rms_um": float(np.sqrt(np.mean(e ** 2))),
        "e_ptp_um": float(np.ptp(e)),
    }
    return w, meta


def dominant_freq_hz(e_um: np.ndarray, dt_s: float = GRID_DT_S) -> float:
    """全谱主频（可能落在慢偏置方向变化上；配合 vibration_dominant_hz 区分 R-014）。"""
    x = np.asarray(e_um, dtype=float) - float(np.mean(e_um))
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=dt_s)
    return float(f[np.argmax(P)])


def vibration_dominant_hz(e_um: np.ndarray, lo_hz: float = 1.0, hi_hz: float = 45.0,
                          dt_s: float = GRID_DT_S) -> float:
    """目标频带(默认 1..45 Hz)内功率谱主频，用作“振动主频”（区别于慢偏置）。"""
    x = np.asarray(e_um, dtype=float) - float(np.mean(e_um))
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, d=dt_s)
    m = (f >= lo_hz) & (f <= hi_hz)
    if not np.any(m):
        return float("nan")
    return float(f[m][np.argmax(P[m])])
