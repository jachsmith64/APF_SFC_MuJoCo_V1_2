"""
V1.3 数据包验收（只读，不改任何冻结字节）：
校验 outputs/wfit_P05R01 随包文件之间的溯源/参数/频带自洽。

V1.3 同步：sfc_tuning.json 改为“加速度→虚拟力”格式（method=acceleration_to_virtual_force，
含 force_map_mass_kg / accel_window_points / f_ease_N / f_interf_N / f_max_N）；
replay_params 与整定文件的一致性检查改为 force_map_mass_kg/accel_window_points +
sfc_m/mu/n/g，旧 k_a/B0/K_v 检查删除。

用法：<venv>/python.exe tests/acceptance.py
全部通过打印 OK 且退出码 0；任何不一致打印具体原因并以非 0 退出。
它是对“冻结文件不可被悄悄改动 / replay_params 与 sfc_tuning 脱节 /
profile 不匹配源数据”三类回归的第一道关卡。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import provenance

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = ""):
    RESULTS.append((name, bool(cond), info))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def _j(rel: str) -> dict:
    return json.loads((config.PACKET_DIR / rel).read_text(encoding="utf-8"))


def main() -> int:
    meta = _j("wfit_meta.json")
    tuning = _j("sfc_tuning.json")
    profile = _j("spectral_profile.json")
    params = config.load_parameters(config.PACKET_DIR / "replay_params.json")

    data_csv = config.PROJECT_DIR / "data" / "P05R01" / "D1_时序.csv"
    data_sha = provenance.sha256_file(data_csv)
    fixed_data_sha = "d8d76ea3a259f8dc631547391ed0ec546b39014c608b90d16e229b68d8ad99fe"

    # ---- 1. 溯源：meta 里记录的源/输出/模型哈希与磁盘字节一致 -----------------
    check("accept: data_id=P05R01", meta.get("data_id") == "P05R01")
    check("accept: 原始 CSV 未变", data_sha == fixed_data_sha, data_sha[:12])
    check("accept: meta.source_csv_sha256=原CSV",
          meta.get("source_csv_sha256") == data_sha)
    for name in ("w_force.csv", "schedule.csv", "e_des_um.csv"):
        rec = meta["outputs_sha256"][name]
        got = provenance.sha256_file(config.PACKET_DIR / name)
        check(f"accept: {name} 字节未变", got == rec["sha256"], got[:12])
    mrec = meta["model"]
    check("accept: scene.xml 模型哈希一致",
          mrec["scene.xml"]["sha256"] == provenance.sha256_file(config.MODEL_XML_PATH))
    check("accept: ur10e.xml 模型哈希一致",
          mrec["ur10e.xml"]["sha256"] == provenance.sha256_file(config.ASSET_DIR / "ur10e.xml"))

    # ---- 2. replay_params 与 sfc_tuning 脱节检查 ------------------------------
    for k, tk in (("sfc_m", "m"), ("sfc_mu", "mu"), ("sfc_n", "n"), ("sfc_g", "g")):
        check(f"accept: params.{k}=tuning.{tk}",
              math.isclose(float(params[k]), float(tuning[tk]), rel_tol=1e-12),
              f"{params[k]:.12g} vs {tuning[tk]:.12g}")
    # V1.3：映射系数/窗口点数与整定文件一致；旧 k_a/B0/K_v 字段已不存在
    check("accept: params.force_map_mass_kg=tuning",
          math.isclose(float(params["force_map_mass_kg"]),
                       float(tuning["force_map_mass_kg"]), rel_tol=1e-12),
          f"{params['force_map_mass_kg']} vs {tuning['force_map_mass_kg']}")
    check("accept: params.accel_window_points=tuning",
          int(params["accel_window_points"]) == int(tuning["accel_window_points"]),
          f"{params['accel_window_points']} vs {tuning['accel_window_points']}")
    aw = int(tuning["accel_window_points"])
    check("accept: accel_window_points 为 ≥3 的奇数", aw >= 3 and aw % 2 == 1, f"w={aw}")
    check("accept: tuning 为 V1.3 格式（无 k_a/B0/K_v）",
          all(k not in tuning for k in ("k_a_N_per_m", "B0", "K_v"))
          and tuning.get("method") == "acceleration_to_virtual_force")
    # f_ease/f_interf/f_max 与 F_vir=P50/P99/max(|F_vir|) 语义自洽
    check("accept: F_vir 分位递增 f_max≥f_interf≥f_ease>0",
          float(tuning["f_max_N"]) >= float(tuning["f_interf_N"])
          >= float(tuning["f_ease_N"]) > 0.0)
    check("accept: params.duration=模板时长",
          math.isclose(float(params["duration_s"]),
                       float(meta["template"]["dur_s"]), rel_tol=0.0, abs_tol=1e-9),
          f"{params['duration_s']} vs {meta['template']['dur_s']}")
    # 正式 SFC：1<n≤5；离散上界覆盖 132Hz 控制步
    check("accept: SFC 正式 n∈(1,5]", 1.0 < float(tuning["n"]) <= 5.0, f"n={tuning['n']:.7f}")
    check("accept: dt_max_s>1/132", float(tuning["dt_max_s"]) > 1.0 / 132.0,
          f"dt_max={tuning['dt_max_s']:.6g}s")
    # 冻结 w(t)/schedule 可直接被 config/disturbance 消费。
    # 覆盖目标与 run.py 一致：required_t_end = schedule 最后时间戳（trajectory.t_end()），
    # 不是 duration_s（= n·dt 的名义循环时长，比最后样本戳大一个 dt）。
    errs = config.validate_params(params)
    check("accept: replay_params 通过校验", errs == [], "; ".join(errs))
    import numpy as np
    import disturbance
    sched = np.genfromtxt(str(config.PACKET_SCHEDULE_FILE), delimiter=",", comments="#")
    sched_end = float(sched[:, 0][-1])
    d = disturbance.load_fixed(params, required_t_end=sched_end)
    check("accept: w(t) 覆盖 schedule 终点", math.isclose(d.t_end, sched_end, abs_tol=1e-9),
          f"w_end={d.t_end:.6f} sched_end={sched_end:.6f}")
    check("accept: w(t) SHA 与 meta 记录一致",
          d.sha256 == meta["outputs_sha256"]["w_force.csv"]["sha256"], d.sha256[:12])

    # ---- 3. 冻结频带 profile 与源数据/主导峰自洽 ------------------------------
    check("accept: profile 源=原 CSV", profile["source_sha256"] == fixed_data_sha)
    slow_top = float(profile["slow_component_range_Hz"][1])
    slow_dom = float(profile["slow_bias_dominant_Hz"])
    check("accept: 慢成分远离 3.9Hz 基频且慢瓣主峰在其中",
          slow_top < 3.0 and slow_dom < slow_top, f"top={slow_top} dom={slow_dom}")
    bands = profile["identified_vibration_bands_Hz"]
    check("accept: 有振动带", len(bands) >= 1)
    check("accept: 首带从慢上界起", math.isclose(bands[0][0], slow_top, abs_tol=1e-5))
    covers_base = any(lo <= 3.9 <= hi for lo, hi in bands)
    dom_has_base = any(3.0 <= p <= 4.5 for p in profile["dominant_peaks_Hz"])
    ck = profile["stroke_peak_crosscheck"]
    seg_has_base = (any(3.5 <= p <= 4.5 for p in ck["X_outbound"])
                    and any(3.5 <= p <= 4.5 for p in ck["X_return"]))
    check("accept: 振动带与主导峰覆盖 ~3.9Hz 基频",
          covers_base and dom_has_base and seg_has_base,
          f"bands_cover={covers_base} dom_has={dom_has_base} seg_has={seg_has_base}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n验收共 {len(RESULTS)} 项：OK {len(RESULTS) - len(failed)}，FAIL {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
