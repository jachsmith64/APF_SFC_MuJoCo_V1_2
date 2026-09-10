"""
V1.2 单元自检（无 MuJoCo、无长仿真）：纯函数/纯配置层校验。

用法：<venv>/python.exe tests/test_v12.py
逐项打印 PASS/FAIL；有失败时退出码非 0（便于 CI/验收）。
"""

from __future__ import annotations

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
from apf_sfc import ApfSfc, sfc_dt_max
from sfc_tune import tuning_from_e_des

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = ""):
    RESULTS.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def _eps(name: str, a: float, b: float, tol: float = 1e-4):
    check(name, math.isclose(a, b, rel_tol=tol, abs_tol=tol), f"{a:.6g} vs {b:.6g}")


# ---------------------------------------------------------------- config
def t_config():
    p = config.parameter_defaults()
    check("config: mu 默认论文整定", abs(p["sfc_mu"] - 76150.4348) < 1e-4)
    check("config: B0/K_v 正式=0", p["sfc_B0"] == 0.0 and p["sfc_K_v"] == 0.0)
    ro = {f["key"] for f in config.flatten_groups() if f.get("readonly")}
    check("config: SFC 字段只读", ro >= {"sfc_m", "sfc_mu", "sfc_n", "sfc_g", "sfc_B0", "sfc_K_v"})
    # 校验拦截
    bad = dict(p); bad["disturbance_file"] = ""
    check("config: 缺 disturbance 报错", any("必填" in e for e in config.validate_params(bad)))
    bad2 = dict(p); bad2["control_hz"] = 2000.0
    check("config: control>physics 报错", any("不能高于" in e for e in config.validate_params(bad2)))
    bad3 = dict(p); bad3["sfc_n"] = 6.0
    check("config: n>5 报错", any("1..5" in e for e in config.validate_params(bad3)))
    bad4 = dict(p); bad4["disturbance_file"] = "no/such/file.csv"
    check("config: 文件不存在报错", any("文件不存在" in e for e in config.validate_params(bad4)))
    good = dict(p); good["disturbance_file"] = "outputs/wfit_P05R01/w_force.csv"
    check("config: 正式默认通过", config.validate_params(good) == [])
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


# ---------------------------------------------------------------- SFC 整定 E-009
def t_sfc_tune():
    e_des = np.loadtxt(str(config.PACKET_E_DES_FILE), delimiter=",", comments="#")[:, 1]
    e_abs = np.abs(e_des)
    t = tuning_from_e_des(e_abs, 1500.0, 1.0, 1.0, 1.5)
    _eps("sfc: n=E-009", t["n"], 2.8279765, 1e-4)
    _eps("sfc: mu=E-009", t["mu"], 76150.4348, 1e-3)
    _eps("sfc: g=E-009", t["g"], 0.0241538, 1e-3)
    _eps("sfc: dt_max≈0.0391", t["dt_max_s"], 0.0391227, 1e-3)
    check("sfc: B0/K_v=0", t["B0"] == 0.0 and t["K_v"] == 0.0)
    check("sfc: n 在 1..5", 1.0 < t["n"] <= 5.0)
    # 离散上界与 run.py 启动复算口径一致（apf_sfc.sfc_dt_max 用同文件 e_max）
    dm = sfc_dt_max(1500.0, 1.0, t["mu"], t["n"], float(np.max(e_abs)) * 1e-6)
    _eps("sfc: sfc_dt_max 复算一致", dm, t["dt_max_s"], 1e-9)
    check("sfc: dt_max>1/132", t["dt_max_s"] > 1.0 / 132.0)
    try:
        tuning_from_e_des(e_abs, 1500.0, 1.0, 1.0, 1.0)
        check("sfc: ratio≤1 应报错", False)
    except ValueError:
        check("sfc: ratio≤1 应报错", True)


# ---------------------------------------------------------------- apf_sfc
def t_apf_sfc():
    c = ApfSfc(k_a=1500.0, m=1.0, mu=76150.4348, n=2.8279765, g=0.0241538, b_eps=0.0, K_v=0.0)
    check("apf: 论文一致(正式)", c.is_formal)
    c2 = ApfSfc(k_a=1500.0, m=1.0, mu=76150.4348, n=2.8279765, g=0.0241538, b_eps=0.1, K_v=0.0)
    check("apf: B0>0 非正式", not c2.is_formal)
    dy = c.step(0.001, 1e-5)          # e_y>0 → F_apf<0 → dy 应 <0
    check("apf: dy 与 e 反号(负反馈)", dy < 0, f"dy={dy:.2e}")
    check("apf: logs 键", set(c.logs()) == {"sfc_v_internal", "sfc_v_out", "sfc_shear_force"})
    # 非有限输入防护
    try:
        c.step(0.001, float("nan"))
        check("apf: NaN e 应报错", False)
    except Exception:
        check("apf: NaN e 应报错", True)


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
        # 不覆盖 → 报错
        try:
            disturbance.load_fixed(p, required_t_end=6.0)
            check("dist: 覆盖不足应报错", False)
        except disturbance.DisturbanceError:
            check("dist: 覆盖不足应报错", True)
        # 非严格递增 → 报错
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
    # Parseval：单边 PSD 全轴积分 ≈ 信号标准差（谱峰能量必然泄漏到旁瓣，
    # 故只在窄带内积分会低估；全轴积分才是幅值恢复的稳健判据）。
    rms_full = spectral.band_rms(f, psd, 0.0, fs / 2.0)
    _eps("spectral: 全轴 RMS≈幅值/√2", rms_full, 3.0 / math.sqrt(2.0), 0.03)
    rms_band = spectral.band_rms(f, psd, 3.0, 4.0)
    check("spectral: 窄带 RMS<全轴且>0", 0.0 < rms_band < rms_full, f"{rms_band:.3f}")
    # 非均匀 → 均匀
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
    for fn in (t_config, t_provenance, t_sfc_tune, t_apf_sfc, t_disturbance, t_spectral, t_profile):
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
