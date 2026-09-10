"""
V1.3 单元自检（无 MuJoCo、无长仿真）：纯函数/纯配置层校验。

用法：<venv>/python.exe tests/test_v13.py
逐项打印 PASS/FAIL；有失败时退出码非 0（便于 CI/验收）。

V1.3 覆盖的重点（相对 V1.2 测试）：
- config：删除 k_a/sfc_B0/sfc_K_v，新增 force_map_mass_kg/accel_window_points（≥3 奇数）；
- sfc_tune：加速度→虚拟力整定口径（f_ease=P50、f_interf=P99、f_max=max |F_vir|），
  旧格式（含 k_a_N_per_m）必须显式拒绝；
- apf_sfc.CausalAccelForceMapper：样本不足不 ready、二次信号 → a_y_est=a0、
  F_vir=+force_map_mass_kg·a_y_est（正号）；
- apf_sfc.ApfSfc：正 F_vir → 正 v_s/v_sfc_out，负 → 负；F_vir=0 且初速 0 → 输出恒 0。
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import disturbance
import provenance
import spectral
from apf_sfc import ApfSfc, ApfSfcNumericalError, CausalAccelForceMapper, sfc_dt_max
from sfc_tune import load_tuning, tuning_from_e_des

RESULTS: list[tuple[str, bool, str]] = []

# 随包 P05R01 的 V1.3 整定值（sfc_tuning.json，accel_window_points=9）
PKG_N = 4.6576146
PKG_MU = 253358215.17
PKG_G = 0.01561887


def check(name: str, cond: bool, info: str = ""):
    RESULTS.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def _eps(name: str, a: float, b: float, tol: float = 1e-4):
    check(name, math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a:.6g} vs {b:.6g}")


# ---------------------------------------------------------------- config
def t_config():
    p = config.parameter_defaults()
    check("config: 无 k_a/sfc_B0/sfc_K_v",
          not ({"k_a", "sfc_B0", "sfc_K_v"} & set(p)))
    check("config: force_map_mass_kg 默认 1.0", p["force_map_mass_kg"] == 1.0)
    check("config: accel_window_points 默认 5", p["accel_window_points"] == 5)
    ro = {f["key"] for f in config.flatten_groups() if f.get("readonly")}
    check("config: SFC 字段只读", ro >= {"sfc_m", "sfc_mu", "sfc_n", "sfc_g"})
    fmap = config.find_field("force_map_mass_kg")
    check("config: 映射系数 help 提示需重新整定",
          "重新整定" in fmap["help"] and "归一化初值" in fmap["help"])
    # 校验拦截
    bad = dict(p); bad["disturbance_file"] = ""
    check("config: 缺 disturbance 报错", any("必填" in e for e in config.validate_params(bad)))
    bad2 = dict(p); bad2["control_hz"] = 2000.0
    check("config: control>physics 报错", any("不能高于" in e for e in config.validate_params(bad2)))
    bad3 = dict(p); bad3["sfc_n"] = 6.0
    check("config: n>5 报错", any("1..5" in e for e in config.validate_params(bad3)))
    bad4 = dict(p); bad4["disturbance_file"] = "no/such/file.csv"
    check("config: 文件不存在报错", any("文件不存在" in e for e in config.validate_params(bad4)))
    for w in (2, 4, 4.0, 11):
        bw = dict(p); bw["accel_window_points"] = w
        check(f"config: 窗口 {w} 非≥3奇数报错",
              any("accel_window_points" in e for e in config.validate_params(bw)))
    for w in (3, 5, 7, 9):
        gw = dict(p); gw["accel_window_points"] = w
        gw["disturbance_file"] = "outputs/wfit_P05R01/w_force.csv"
        check(f"config: 窗口 {w} 通过校验", config.validate_params(gw) == [])
    good = dict(p); good["disturbance_file"] = "outputs/wfit_P05R01/w_force.csv"
    check("config: 正式默认通过", config.validate_params(good) == [],
          "; ".join(config.validate_params(good)))
    # 路径相对化
    check("config: project_relative 往返", config.resolve_path(
        config.project_relative_str(config.PACKET_DIR)) == config.PACKET_DIR)


# ---------------------------------------------------------------- provenance
def t_provenance():
    csv = ROOT / "data" / "P05R01" / "D1_时序.csv"
    h = provenance.sha256_file(csv)
    check("provenance: 数据 CSV 哈希冻结", h ==
          "d8d76ea3a259f8dc631547391ed0ec546b39014c608b90d16e229b68d8ad99fe", h[:12])
    d = {"b": 1, "a": [3, 1]}
    f1 = provenance.json_fingerprint(d)
    f2 = provenance.json_fingerprint({"a": [3, 1], "b": 1})
    check("provenance: JSON 指纹键序无关", f1 == f2)
    check("provenance: JSON 指纹稳定", f1 == provenance.json_fingerprint(d))
    m = provenance.model_fingerprint()
    check("provenance: 模型指纹含双 xml", {"scene.xml", "ur10e.xml"} <= set(m))
    rec = provenance.file_record(config.PACKET_W_FILE)
    check("provenance: file_record 形状", set(rec) == {"path", "sha256"} and len(rec["sha256"]) == 64)


# ---------------------------------------------------------------- SFC 整定（V1.3 加速度口径）
def t_sfc_tune():
    ed = np.loadtxt(str(config.PACKET_E_DES_FILE), delimiter=",", comments="#")

    def _tune(w: int) -> dict:
        return tuning_from_e_des(ed[:, 0], ed[:, 1], 1.0, 1.0, 1.0, 1.5, w)

    # 窗口点数越大，加速度估计越平滑 → P99/P50 越小 → n 越小；窗口 3/5/7 超正式范围要显式报错
    ns: dict[int, float] = {}
    for w in (3, 5, 7, 9):
        try:
            ns[w] = _tune(w)["n"]
        except ValueError as exc:
            check(f"sfc: 窗口 {w} 超正式范围(n>5) 显式报错", "超出正式范围" in str(exc),
                  f"w{w}")
    check("sfc: n 随窗口单调下降",
          all(ns[a] > ns[b] for a, b in ((9, 7), (7, 5), (5, 3)) if a in ns and b in ns),
          " ".join(f"w{w}={v:.4f}" for w, v in sorted(ns.items())))
    check("sfc: 窗口 3/5/7 均超正式范围（本 e_des 经验值）",
          all(w not in ns for w in (3, 5, 7)),
          " ".join(f"w{w}={'>5' if w not in ns else f'{ns[w]:.4f}'}" for w in (3, 5, 7)))
    t = _tune(9)
    check("sfc: 默认窗口组合的 n 在 1..5", 1.0 < t["n"] <= 5.0, f"n={t['n']:.7f}")
    _eps("sfc: n=随包值", t["n"], PKG_N, 1e-6)
    _eps("sfc: mu=随包值", t["mu"], PKG_MU, 1e-6)
    _eps("sfc: g=随包值", t["g"], PKG_G, 1e-5)
    check("sfc: method=acceleration_to_virtual_force",
          t["method"] == "acceleration_to_virtual_force")
    check("sfc: 无旧字段 k_a/B0/K_v", not ({"k_a_N_per_m", "B0", "K_v"} & set(t)))
    check("sfc: 分位口径 f_max≥f_interf≥f_ease>0",
          t["f_max_N"] >= t["f_interf_N"] >= t["f_ease_N"] > 0.0,
          f"{t['f_ease_N']:.4g}/{t['f_interf_N']:.4g}/{t['f_max_N']:.4g}")
    check("sfc: 映射系数记录为 1.0", t["force_map_mass_kg"] == 1.0)
    check("sfc: 窗口记录为 9", t["accel_window_points"] == 9)
    # f_max_N 是 max|F_vir|，与 force_map_mass_kg 线性：(F_vir=F_mass·a)
    _eps("sfc: f_max_N = 1.0×a_est_peak", t["f_max_N"], t["a_est_peak_m_s2"], 1e-9)
    # 离散上界与 run.py 启动复算口径一致（apf_sfc.sfc_dt_max 用同一 f_max_N）
    dm = sfc_dt_max(t["m"], t["mu"], t["n"], t["f_max_N"])
    _eps("sfc: sfc_dt_max 复算一致", dm, t["dt_max_s"], 1e-9)
    check("sfc: dt_max>1/132", t["dt_max_s"] > 1.0 / 132.0, f"dt_max={t['dt_max_s']:.6g}s")
    check("sfc: dt_max>1/1000", t["dt_max_s"] > 1.0 / 1000.0)
    # 随包 sfc_tuning.json 必须是 V1.3 格式且与现算一致
    pkg = load_tuning(config.PACKET_TUNING_FILE)
    _eps("sfc: 随包 tuning.n 与现算一致", float(pkg["n"]), t["n"], 1e-9)
    _eps("sfc: 随包 tuning.mu 与现算一致", float(pkg["mu"]), t["mu"], 1e-9)
    # 旧格式必须显式拒绝（不静默读取）
    with tempfile.TemporaryDirectory() as td:
        old = Path(td) / "sfc_tuning.json"
        old.write_text(json.dumps({"k_a_N_per_m": 1500.0, "m": 1.0, "mu": 1.0,
                                   "n": 2.0, "g": 0.01}), encoding="utf-8")
        try:
            load_tuning(old)
            check("sfc: 旧格式应显式报错", False)
        except ValueError as exc:
            check("sfc: 旧格式应显式报错", "k_a_N_per_m" in str(exc) and "sfc_tune" in str(exc))
        miss = Path(td) / "partial.json"
        miss.write_text(json.dumps({"method": "acceleration_to_virtual_force"}),
                        encoding="utf-8")
        try:
            load_tuning(miss)
            check("sfc: 缺字段应显式报错", False)
        except ValueError as exc:
            check("sfc: 缺字段应显式报错", "缺少 V1.3 整定字段" in str(exc))
    try:
        tuning_from_e_des(ed[:, 0], ed[:, 1], 1.0, 1.0, 1.0, 1.0, 9)
        check("sfc: ratio≤1 应报错", False)
    except ValueError:
        check("sfc: ratio≤1 应报错", True)
    try:
        tuning_from_e_des(ed[:, 0][::-1], ed[:, 1], 1.0, 1.0, 1.0, 1.5, 9)
        check("sfc: t 非递增应报错", False)
    except ValueError:
        check("sfc: t 非递增应报错", True)


# ---------------------------------------------------------------- 加速度→虚拟力映射器
def t_mapper():
    a0 = 2.0
    w = 5
    m = CausalAccelForceMapper(force_map_mass_kg=1.0, accel_window_points=w)
    check("mapper: 初始未 ready", m.ready is False)
    a, f, ready = m.step(0.0, 0.0)
    check("mapper: 样本不足 → ready=False 且输出 0",
          ready is False and a == 0.0 and f == 0.0)
    t_s = 0.0
    out = None
    for _ in range(w):
        t_s += 0.001
        out = m.step(t_s, 0.5 * a0 * t_s ** 2)
    a, f, ready = out
    check("mapper: 满窗后 ready", ready is True)
    _eps("mapper: e=0.5·a0·t² → a_y_est=a0", a, a0, 1e-6)
    _eps("mapper: F_vir=+force_map_mass_kg·a_y_est", f, 1.0 * a0, 1e-6)
    check("mapper: a0>0 → F_vir>0", f > 0.0)
    m2 = CausalAccelForceMapper(2.5, 5)
    t_s = 0.0
    for _ in range(5):
        t_s += 0.001
        a2, f2, _ = m2.step(t_s, 0.5 * (-3.0) * t_s ** 2)
    check("mapper: a0<0 → F_vir<0", f2 < 0.0, f"F_vir={f2:.6g}")
    _eps("mapper: 负加速度估计", a2, -3.0, 1e-6)
    _eps("mapper: F_vir=2.5×a_y_est", f2, 2.5 * (-3.0), 1e-6)
    # 非严格递增时间戳 / 非有限输入
    for bad_t, bad_e, name in ((t_s, 0.0, "时间戳非递增应报错"),
                               (float("nan"), 0.0, "t 非有限应报错")):
        try:
            m2.step(bad_t, bad_e)
            check(f"mapper: {name}", False)
        except ApfSfcNumericalError:
            check(f"mapper: {name}", True)
    try:
        m2.step(t_s + 0.001, float("nan"))
        check("mapper: e_y 非有限应报错", False)
    except ApfSfcNumericalError:
        check("mapper: e_y 非有限应报错", True)
    # 窗口点数非法
    for bad in (2, 4, 0, -1):
        try:
            CausalAccelForceMapper(1.0, bad)
            check(f"mapper: 窗口 {bad} 应报错", False)
        except ValueError:
            check(f"mapper: 窗口 {bad} 应报错", True)


# ---------------------------------------------------------------- apf_sfc 核心
def t_apf_sfc():
    c = ApfSfc(m=1.0, mu=PKG_MU, n=PKG_N, g=PKG_G)
    check("apf: 论文一致(正式)", c.is_formal)
    check("apf: logs 键", set(c.logs()) == {"sfc_a_internal", "sfc_v_internal",
                                            "sfc_v_out", "sfc_shear_force", "F_vir"})
    check("apf: params 只有 m/mu/n/g", set(c.params_dict()) == {"m", "mu", "n", "g"})
    v_out = c.step(0.001, 0.1)
    check("apf: F_vir>0 → sfc_v_out>0", v_out > 0.0, f"v_out={v_out:.4g}")
    check("apf: F_vir>0 → a/v 内部同号",
          c.a > 0.0 and c.v > 0.0 and c.shear >= 0.0)
    c2 = ApfSfc(m=1.0, mu=PKG_MU, n=PKG_N, g=PKG_G)
    v_out2 = c2.step(0.001, -0.1)
    check("apf: F_vir<0 → sfc_v_out<0", v_out2 < 0.0, f"v_out={v_out2:.4g}")
    check("apf: F_vir<0 → a/v 内部同号",
          c2.a < 0.0 and c2.v < 0.0 and c2.shear <= 0.0)
    # F_vir≡0 且初速 0：全部输出保持 0（剪切项 0·|0|^n=0）
    c3 = ApfSfc(m=1.0, mu=PKG_MU, n=PKG_N, g=PKG_G)
    zeros = [c3.step(0.001, 0.0) for _ in range(10)]
    check("apf: F_vir=0 初速 0 → 输出恒 0", all(x == 0.0 for x in zeros))
    check("apf: F_vir=0 → 内部 a/v/shear 恒 0",
          c3.a == 0.0 and c3.v == 0.0 and c3.shear == 0.0)
    # 正负对称
    _eps("apf: 正负输入对称", v_out, -v_out2, 1e-12)
    # 内部方程口径：a_s=(F_vir-μ·sign(v)·|v|^n)/m
    c4 = ApfSfc(m=2.0, mu=0.0, n=3.0, g=0.5)
    v4 = c4.step(0.01, 4.0)
    _eps("apf: μ=0 时 a=F/m", c4.a, 2.0, 1e-12)
    _eps("apf: v=v+a·dt", c4.v, 0.02, 1e-12)
    _eps("apf: v_out=g·v", v4, 0.01, 1e-12)
    # 非有限/非法 dt 防护
    try:
        c.step(0.001, float("nan"))
        check("apf: NaN F_vir 应报错", False)
    except ApfSfcNumericalError:
        check("apf: NaN F_vir 应报错", True)
    try:
        c.step(0.0, 0.1)
        check("apf: dt=0 应报错", False)
    except ApfSfcNumericalError:
        check("apf: dt=0 应报错", True)
    # dt_max 与论文离散口径
    dm = sfc_dt_max(1.0, PKG_MU, PKG_N, 0.837816)
    check("apf: dt_max 有限且>1/132", math.isfinite(dm) and dm > 1.0 / 132.0, f"{dm:.6g}")
    check("apf: μ=0 → dt_max=inf", sfc_dt_max(1.0, 0.0, PKG_N, 1.0) == float("inf"))


# ---------------------------------------------------------------- disturbance
def t_disturbance():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "w.csv"
        t = np.linspace(0, 5, 501)
        np.savetxt(f, np.column_stack([t, 0.01 * np.sin(2 * np.pi * 3.0 * t)]),
                   delimiter=",", header="t_s,value", comments="#")
        p = {"disturbance_file": str(f)}
        d = disturbance.load_fixed(p, required_t_end=4.9)
        check("dist: 加载成功且覆盖", d.t[-1] >= 4.9 - 1e-9)
        check("dist: w() 有限", math.isfinite(d.w(2.5)))
        try:
            disturbance.load_fixed(p, required_t_end=6.0)
            check("dist: 覆盖不足应报错", False)
        except disturbance.DisturbanceError:
            check("dist: 覆盖不足应报错", True)
        bad = np.array([[0.0, 0.0], [1.0, 0.1], [1.0, 0.2]])
        fb = Path(td) / "bad.csv"
        np.savetxt(fb, bad, delimiter=",")
        try:
            disturbance.load_fixed({"disturbance_file": str(fb)}, required_t_end=0.5)
            check("dist: 非严格递增应报错", False)
        except disturbance.DisturbanceError:
            check("dist: 非严格递增应报错", True)


# ---------------------------------------------------------------- spectral
def t_spectral():
    fs = 1000.0
    t = np.arange(0, 8, 1 / fs)
    x = 3.0 * np.sin(2 * np.pi * 3.5 * t)
    f, psd = spectral.welch_psd(x, fs, win_s=2.0)
    k = int(np.argmax(psd))
    _eps("spectral: 恢复 3.5Hz", f[k], 3.5, 0.05)
    rms_full = spectral.band_rms(f, psd, 0.0, fs / 2.0)
    _eps("spectral: 全轴 RMS≈幅值/√2", rms_full, 3.0 / math.sqrt(2.0), 0.03)
    rms_band = spectral.band_rms(f, psd, 3.0, 4.0)
    check("spectral: 窄带 RMS<全轴且>0", 0.0 < rms_band < rms_full, f"{rms_band:.3f}")
    tu = np.cumsum(np.r_[0.0, np.where(np.arange(1, 501) % 3 == 0, 0.009, 0.007)])
    xr = np.sin(2 * np.pi * 2.0 * tu)
    tu2, x2, fs2 = spectral.resample_uniform(tu, xr)
    check("spectral: 重采样均匀", np.allclose(np.diff(tu2), np.diff(tu2)[0], atol=1e-12))
    check("spectral: fs≈平均步长", abs(fs2 - 1 / (tu[-1] / (tu.size - 1))) < 1e-6)


# ---------------------------------------------------------------- make_profile
def t_profile():
    import make_profile
    doc = make_profile.build_profile(make_profile.DEFAULT_CSV, out_json=None)
    check("profile: 源 SHA", doc["source_sha256"][:12] == "d8d76ea3a259")
    bands = doc["identified_vibration_bands_Hz"]
    check("profile: 有振动带", len(bands) >= 1)
    slow_top = doc["slow_component_range_Hz"][1]
    check("profile: 慢上界含 0.5Hz 慢瓣", doc["slow_bias_dominant_Hz"] < slow_top)
    check("profile: 慢<首振动带下界", slow_top <= bands[0][0] + 1e-6)
    ck = doc["stroke_peak_crosscheck"]
    check("profile: 单段交叉核对有 3.9Hz 基频",
          any(3.5 <= pk <= 4.5 for pk in ck.get("X_outbound", [])) and
          any(3.5 <= pk <= 4.5 for pk in ck.get("X_return", [])))


def main() -> int:
    for fn in (t_config, t_provenance, t_sfc_tune, t_mapper, t_apf_sfc,
               t_disturbance, t_spectral, t_profile):
        try:
            fn()
        except Exception as exc:
            RESULTS.append((fn.__name__, False, f"异常：{exc!r}"))
            print(f"[FAIL] {fn.__name__} 抛异常：{exc!r}")
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n共 {len(RESULTS)} 项：PASS {len(RESULTS) - len(failed)}，FAIL {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        for s in (sys.stdout, sys.stderr):
            s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
