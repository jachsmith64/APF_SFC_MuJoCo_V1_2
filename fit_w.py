"""
真实预实验数据 -> MuJoCo 等效 Y 扰动力 w(t) 的 CLI 入口（V1.2：一次性数据准备工具）。

典型调用：
    python fit_w.py --csv "D:\\...\\X_P05_R01\\D1_时序.csv" --name P05R01 [--refine 3]

产物（out 目录，默认 outputs/wfit_<name>/）：
  w_force.csv         [t(s), Y力(N)]         冻结；正式 A/B 只读它，不自动重拟合
  schedule.csv        [t(s), 沿程(mm)]       冻结（0→远点→0 往返）
  e_des_um.csv        [t(s), 横向偏差(µm)]   冻结（模板目标振动）
  replay_params.json  一整份可运行参数（相对项目路径；UI 载入或 run.py --params）
  wfit_meta.json      拟合元数据（含全部 SHA-256/模型指纹/算法版本）
  wfit_summary.txt    人读摘要
  sfc_tuning.json     （由 sfc_tune.py 另生成）SFC 论文整定依据

V1.2 关键约束：
- 冻结文件 w/s/e 只写一次；meta/replay_params/summary 可随时 finalize_packet() 安全刷新
  （只加溯源字段，绝不动 w/s/e 字节——否则会破坏已发布哈希）。
- replay_params.json 用项目相对路径，不再存 C:\\ D:\\ 绝对路径（R-011）。
- 切窗阈值来自 wfit 的 dataset profile（P05R01），条件未命中报错（R-017）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
import provenance
import wfit
from wfit import SEAM_SMOOTH_S, Template

ALGORITHM_VERSION = "v1.2-wfit"
ALGORITHM_RECIPE = ("单程拉长：一次启动→前向稳态窗复制 N 次→一次换向→返程稳态窗复制→一次停止；"
                    "仅确定性复制真实稳态片段并做接缝平滑，不含新正弦/随机相位/随机噪声。")


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _write_csv(path: Path, t: np.ndarray, val: np.ndarray, fmt: str = "%.6f") -> None:
    np.savetxt(str(path), np.column_stack([t, val]), delimiter=",", fmt=fmt,
               header="t_s,value", comments="#")


# ----------------------------------------------------------------- 元数据（完整）
def _stretch_counts(meta: dict) -> dict[str, Any]:
    """从 template.segments 推导前向/返向稳态复制次数（meta 溯源补充）。"""
    segs = meta.get("template", {}).get("segments", [])
    nf = sum(1 for s in segs if s["name"] == "mid_fwd")
    nr = sum(1 for s in segs if s["name"] == "mid_ret")
    cp = meta.get("cut_points", {})
    return {
        "recipe": "单程拉长",
        "seam_smooth_s": SEAM_SMOOTH_S,
        "n_forward_repeat": nf,
        "n_return_repeat": nr,
        "source_approx_s": {
            "start": [0.0, cp.get("iA")],
            "mid_fwd": [cp.get("iA"), cp.get("iB")],
            "turn": [cp.get("iB"), cp.get("iC")],
            "mid_ret": [cp.get("iC"), cp.get("iD")],
            "stop": [cp.get("iD"), None],
        },
    }


def _build_final_meta(out_dir: Path) -> dict:
    """读取现存基础 meta，补齐全部溯源/哈希字段后返回（不写文件）。"""
    out_dir = Path(out_dir)
    meta_p = out_dir / "wfit_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    src = Path(meta.get("source_csv", ""))
    outputs = {
        "e_des_um.csv": provenance.file_record(out_dir / "e_des_um.csv"),
        "w_force.csv": provenance.file_record(out_dir / "w_force.csv"),
        "schedule.csv": provenance.file_record(out_dir / "schedule.csv"),
    }
    pc = config.OUTPUT_ROOT / "plant_cache.npz"
    plant = meta.get("plant_dc_um_per_n")
    meta.update({
        "version": "v1.2-meta",
        "source_csv": str(src),
        "source_csv_sha256": provenance.sha256_file(src),
        "source_deterministic": True,
        "outputs_sha256": outputs,
        "model": provenance.model_fingerprint(),
        "init_q": [round(float(x), 12) for x in config.INIT_Q],
        "physics_hz": 1000,
        "control_hz": 1000.0,
        "servo_note": config.SERVO_NOTE,
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_recipe": ALGORITHM_RECIPE,
        "plant_cache": {
            "path": str(pc),
            "sha256": provenance.sha256_file(pc),
            "dc_um_per_n": plant,
            "fingerprint_note": "缓存自带模型/物理/姿态指纹，不一致自动重测（R-007）",
        },
        "stretch": _stretch_counts(meta),
        "original_sampling_note": "视觉采样 ~132Hz 且缺帧；重采样到 1ms 仅供切窗/拼接/FFT，"
                                   "频谱 profile 另读原始时间戳（见 make_profile.py）",
    })
    return meta


def _write_replay_params(out_dir: Path, meta: dict) -> Path:
    """
    按冻结数据包写可运行 params（项目相对路径；V1.3 SFC/映射参数取 sfc_tuning.json）。

    V1.3 起 replay_params 读 force_map_mass_kg / accel_window_points + SFC(m,μ,n,g)；
    不再有 k_a / sfc_B0 / sfc_K_v。sfc_tuning.json 为 V1.2 旧格式或缺字段时必须
    **显式报错**（sfc_tune.load_tuning），不做静默退化。
    """
    from sfc_tune import load_tuning
    p = config.parameter_defaults()
    p["duration_s"] = round(float(meta["template"]["dur_s"]), 3)
    p["motion_direction"] = "+X"
    w = out_dir / "w_force.csv"
    sched = out_dir / "schedule.csv"
    p["disturbance_file"] = config.project_relative_str(w)
    p["replay_schedule"] = config.project_relative_str(sched)
    tune = out_dir / "sfc_tuning.json"
    if not tune.is_file():
        tune = config.PACKET_TUNING_FILE
    if tune.is_file():
        td = load_tuning(tune)          # 旧格式在此抛 ValueError，向上传播
        p.update({
            "force_map_mass_kg": float(td["force_map_mass_kg"]),
            "accel_window_points": int(td["accel_window_points"]),
            "sfc_m": float(td["m"]), "sfc_n": float(td["n"]),
            "sfc_mu": float(td["mu"]), "sfc_g": float(td["g"]),
        })
    params_p = out_dir / "replay_params.json"
    config.save_parameters(p, params_p)
    return params_p


def finalize_packet(out_dir: Path) -> dict[str, Path]:
    """
    安全刷新数据包元数据/参数/摘要（只读 w/e/s 字节算哈希，绝不改写三个冻结文件）。
    可对新拟合结果或升级旧 V1.1 数据包调用。
    """
    out_dir = Path(out_dir)
    meta = _build_final_meta(out_dir)
    paths = {
        "w_force": out_dir / "w_force.csv",
        "schedule": out_dir / "schedule.csv",
        "e_des": out_dir / "e_des_um.csv",
        "params": out_dir / "replay_params.json",
        "meta": out_dir / "wfit_meta.json",
        "summary": out_dir / "wfit_summary.txt",
    }
    (paths["meta"]).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    params_p = _write_replay_params(out_dir, meta)
    paths["params"] = params_p

    # 摘要
    lines = [
        f"# w(t) 拟合：{meta['data_id']}",
        f"源数据：{meta['source_csv']}",
        f"源 CSV SHA-256：{meta['source_csv_sha256'][:16]}…",
        f"模板时长 {meta['template']['dur_s']:.2f} s（{meta['template']['n_samples']} 点 @ "
        f"{wfit.GRID_DT_S * 1e3:.0f} ms），远点 {meta['template']['max_along_mm']:.1f} mm",
        f"算法版本：{meta['algorithm_version']}；{meta['algorithm_recipe']}",
        f"稳定窗复制：前向 x{meta['stretch']['n_forward_repeat']}，返向 x"
        f"{meta['stretch']['n_return_repeat']}；接缝平滑 {meta['stretch']['seam_smooth_s']} s",
        f"反演：f_pass={meta['inversion']['f_pass_hz']:.0f} Hz，f_stop={meta['inversion']['f_stop_hz']:.0f} Hz",
    ]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


# ----------------------------------------------------------------- 生成（完整流程）
def write_outputs(
    out_dir: Path,
    name: str,
    csv_src: str,
    T: Template,
    w: np.ndarray,
    plant: wfit.Plant,
    meta: dict,
    cuts: dict,
) -> dict[str, Path]:
    """写 w/e/s + 基础 meta，最后 finalize 补全溯源（w/e/s 只写这一次）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = T.t
    dur = meta["dur_s"]

    _write_csv(out_dir / "w_force.csv", t, w)
    _write_csv(out_dir / "schedule.csv", t, T.along_mm)
    _write_csv(out_dir / "e_des_um.csv", t, T.e_um)

    seams = np.asarray(T.seam_jumps_um, dtype=float) if T.seam_jumps_um else np.zeros(0)
    base = {
        "data_id": name,
        "source_csv": csv_src,
        "plant_dc_um_per_n": plant.dc_um_per_n,
        "cut_points": cuts,
        "template": {
            "dur_s": dur,
            "n_samples": int(T.t.size),
            "max_along_mm": float(T.along_mm.max()),
            "along_end_mm": float(T.along_mm[-1]),
            "n_seams": len(T.seam_jumps_um),
            "seam_jump_mean_um": float(seams.mean()) if seams.size else 0.0,
            "seam_jump_max_um": float(seams.max()) if seams.size else 0.0,
            "segments": [{"name": nm, "t0_s": round(float(T.t[a]), 3),
                          "t1_s": round(float(T.t[b - 1]), 3), "n": int(b - a)}
                         for nm, a, b in T.segments],
        },
        "inversion": meta,
        "refined": False,
    }
    (out_dir / "wfit_meta.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = finalize_packet(out_dir)
    return paths


def _baseline_e_sim(params: dict, out_dir: Path) -> np.ndarray:
    """跑一次 baseline 回放，返回与模板网格对齐的 e_sim(µm)。"""
    from run import run_single
    out_dir = Path(out_dir)
    run_single(params, config.RUN_MODE_BASELINE, out_dir=out_dir, verbose=False)
    raw = np.loadtxt(str(out_dir / "trajectory.csv"), delimiter=",", comments="#")
    return (raw[:, 7] - raw[:, 2]) * 1e6          # y_act - y_ref -> µm


def refine_w(
    out_dir: Path,
    n_iter: int,
    lr: float = 0.7,
    f_pass: float = wfit.INV_F_PASS_HZ,
    f_stop: float = wfit.INV_F_STOP_HZ,
) -> dict:
    """残差迭代：e_des−e_sim 逐次反演成附加力加进 w（补偿远端柔度漂移）。就地改写 w_force 与 meta。"""
    out_dir = Path(out_dir)
    e_des = np.loadtxt(str(out_dir / "e_des_um.csv"), delimiter=",", comments="#")[:, 1]
    w = np.loadtxt(str(out_dir / "w_force.csv"), delimiter=",", comments="#")[:, 1]
    params = config.load_parameters(out_dir / "replay_params.json")
    plant = wfit.measure_plant(cache=wfit.PLANT_CACHE)
    tmp = config.OUTPUT_ROOT / "tmp_refine_baseline"

    hist: list[dict] = []
    for i in range(int(n_iter)):
        params["disturbance_file"] = str((out_dir / "w_force.csv").resolve())
        e_sim = _baseline_e_sim(params, tmp)
        res = e_des - e_sim
        rr = float(np.sqrt(np.mean(res ** 2)))
        cc = float(np.corrcoef(e_des, e_sim)[0, 1])
        hist.append({"iter": i + 1, "residual_rms_um": rr, "corr": round(cc, 4),
                     "e_sim_rms_um": float(np.sqrt(np.mean(e_sim ** 2)))})
        print(f"[refine] iter {i + 1}: e_sim rms={hist[-1]['e_sim_rms_um']:.2f} um  "
              f"corr={cc:.3f}  residual rms={rr:.2f} um", flush=True)
        if rr < 2.0:
            break
        dw = wfit.invert_series(res, plant, f_pass, f_stop) * lr
        w = w + dw
        _write_csv(out_dir / "w_force.csv",
                   np.arange(w.size, dtype=float) * wfit.GRID_DT_S, w)

    meta_p = out_dir / "wfit_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["inversion"].update({
        "w_rms_N": float(np.sqrt(np.mean(w ** 2))),
        "w_ptp_N": float(np.ptp(w)),
        "w_peak_abs_N": float(np.max(np.abs(w))),
    })
    meta["refined"] = True
    meta["refine_iterations"] = hist
    meta["refine_lr"] = lr
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "wfit_summary.txt").open("a", encoding="utf-8") as fh:
        fh.write("\n## 残差修正\n")
        for h in hist:
            fh.write(f"  iter {h['iter']}: e_sim rms={h['e_sim_rms_um']:.2f} µm  "
                     f"corr={h['corr']}  residual rms={h['residual_rms_um']:.2f} µm\n")
        fh.write(f"  最终 w：rms {meta['inversion']['w_rms_N']:.3f} N，"
                 f"峰值 {meta['inversion']['w_peak_abs_N']:.3f} N（lr={lr}）\n")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    finalize_packet(out_dir)          # 刷新 w_force 新哈希与摘要
    return {"done": len(hist), "hist": hist, "final": meta["inversion"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="真实预实验数据 -> MuJoCo w(t) 拟合（一次性工具）")
    p.add_argument("--csv", required=True, help="D1_时序.csv 路径")
    p.add_argument("--name", default="", help="数据标识（缺省取 csv 所在目录名，如 P05R01）")
    p.add_argument("--out", default="", help="输出目录（缺省 outputs/wfit_<name>）")
    p.add_argument("--duration", type=float, default=60.0,
                   help="目标时长 s（模板时长略超，因稳态窗整窗复制）")
    p.add_argument("--n-stroke", type=int, default=None,
                   help="固定前向稳态复制次数（缺省自动涨到覆盖目标时长）")
    p.add_argument("--f-pass", type=float, default=wfit.INV_F_PASS_HZ)
    p.add_argument("--f-stop", type=float, default=wfit.INV_F_STOP_HZ)
    p.add_argument("--plant-cache", default=str(wfit.PLANT_CACHE))
    p.add_argument("--remeasure-plant", action="store_true", help="忽略缓存重新标定 plant")
    p.add_argument("--refine", type=int, default=0,
                   help=">0：做 N 轮残差迭代修正（每轮跑一次 ~63s baseline）")
    p.add_argument("--refine-lr", type=float, default=0.7)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    csv_src = str(Path(args.csv).resolve())
    name = args.name or Path(args.csv).parent.name
    out_dir = Path(args.out) if args.out else config.OUTPUT_ROOT / f"wfit_{name}"

    try:
        d = wfit.read_d1(csv_src)
        cuts = {k: round(float(d["t"][v]), 3) for k, v in wfit.detect_cut_points(d).items()}
        print(f"[fit_w] 切窗(s)：{cuts}")
        T = wfit.build_template(d, n_stroke=args.n_stroke, duration_s=args.duration)
        print(f"[fit_w] 模板 {T.t[-1]:.2f} s（{T.t.size} 点），远点 {T.along_mm.max():.1f} mm，"
              f"终点 {T.along_mm[-1]:.3f} mm，振动主频(1-45Hz) "
              f"{wfit.vibration_dominant_hz(T.e_um):.2f} Hz")

        plant = wfit.measure_plant(cache=Path(args.plant_cache), params=None)
        if args.remeasure_plant:
            Path(args.plant_cache).unlink(missing_ok=True)
            plant = wfit.measure_plant(cache=Path(args.plant_cache), params=None)

        w, meta = wfit.invert_to_force(T, plant, args.f_pass, args.f_stop)
        print(f"[fit_w] 反演 w(t)：rms {meta['w_rms_N']:.3f} N，峰值 |w| {meta['w_peak_abs_N']:.3f} N，"
              f"带限 <= {args.f_pass:.0f} Hz（{args.f_stop:.0f} 渐减 0）")

        paths = write_outputs(out_dir, name, csv_src, T, w, plant, meta, cuts)
        if args.refine > 0:
            print(f"[fit_w] 残差修正 {args.refine} 轮（每轮跑一次 ~63s baseline）...", flush=True)
            rinfo = refine_w(out_dir, args.refine, args.refine_lr,
                             f_pass=args.f_pass, f_stop=args.f_stop)
            print(f"[fit_w] 残差修正完成：{rinfo['done']} 轮，"
                  f"残差 rms 最终 {rinfo['hist'][-1]['residual_rms_um']:.2f} µm")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[fit_w] 失败：{exc}", file=sys.stderr)
        return 1

    print(f"[fit_w] 完成 -> {out_dir}")
    print(f"[fit_w] 摘要：{paths['summary'].relative_to(config.PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
