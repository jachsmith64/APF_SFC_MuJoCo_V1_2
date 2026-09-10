"""
正式 A/B 对（V1.2）：一次跑“同一 params 快照”下的 baseline 与 apf_sfc，做成唯一 pair。

公平性（《DS 修改指南》§7.1 + R-017）：
- A、B 共用同一份 params（含同一冻结 w/schedule、同一 INIT_Q、同一物理/控制频率/时长），
  唯一差别是 mode。run_fingerprint 在 analysis.gate_pair 里逐项核对，不满足即报错。
- 每个 pair 落在唯一目录 outputs/pair_YYYYMMDD_HHMMSS/，内含 run_A/、run_B/、analysis/、
  manifest.json；绝不覆盖旧结果（试跑才用 run_*_baseline 之类）。
- manifest.json 先写 provisional（进行中），全部完成后由 finalize 覆盖为正式；中途失败
  留 provisional=True 与 reason，供 UI/审查识别“此 pair 未生效”。

用法：
    python pair.py --params outputs/wfit_P05R01/replay_params.json            # 默认 control_hz 用 params
    python pair.py --params ... --control-hz 132                              # 覆盖控制频率
    python pair.py --params ... --no-analysis                                 # 只跑不分析
    python pair.py --params ... --no-strict                                    # 门控不过仍写分析(标非正式)
    python pair.py --params ... --show                                         # 严格观赛 A/B 两段（任一窗失败→t=0 前报错）
    python pair.py --params ... --show-best-effort                             # 尽力观赛：任一窗失败则降级无头继续

观赛（V1.2 改造）：默认无头；A/B 两段共用同一 window_s=10 与同一 y_limit_um（由冻结扰动包
e_des_um.csv 按 ceil(1.2×max|e_des|) 算一次），标题显著标 A baseline / B APF-SFC，A 结束、B
开窗前打段间提示。仅 --show/--show-best-effort 且 pace>0 时才按真实时推进画面。

产物：run_A/run_B/(trajectory.csv, run_fingerprint.json, ...)、analysis/analysis_metrics.json、
      analysis_summary.txt、manifest.json。
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import config
import provenance
from analysis import analyze_pair, gate_pair
from liveview import DEFAULT_Y_LIMIT_UM, derive_y_limit_um

MANIFEST_NAME = "manifest.json"


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _pos_pace(text: str) -> float:
    try:
        v = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--pace 必须是数值") from None
    if not (v > 0.0):
        raise argparse.ArgumentTypeError("--pace 必须 > 0（不可为 0/负 = 不限速）")
    return v


def _resolve_pair_y_limit(params: dict) -> tuple[float, str]:
    """A/B 共用纵轴：从冻结扰动包目录算一次 ceil(1.2×max|e_des_um|)；缺则默认并记来源。"""
    wdir = None
    if params.get("disturbance_file"):
        try:
            wdir = config.resolve_path(params["disturbance_file"]).parent
        except Exception:  # noqa: BLE001  resolve 失败交给下文默认
            wdir = None
    if wdir is not None:
        v, s = derive_y_limit_um(wdir)
        if v is not None:
            return float(v), s
        return DEFAULT_Y_LIMIT_UM, f"default(缺 e_des_um.csv: {s})"
    return DEFAULT_Y_LIMIT_UM, "default(未指定 disturbance_file)"


def make_pair_dir(base: Path | None = None) -> Path:
    base = base or config.OUTPUT_ROOT
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(10000):
        cand = base / (f"pair_{stamp}" if i == 0 else f"pair_{stamp}_{i}")
        if not cand.exists():
            cand.mkdir(parents=True)
            return cand
    raise RuntimeError("pair 目录命名冲突（应不可能）")


def _write_manifest(out_dir: Path, payload: dict) -> Path:
    p = out_dir / MANIFEST_NAME
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _provisional(out_dir: Path, params: dict, note: str) -> dict:
    return {
        "version": "v1.2",
        "provisional": True,
        "note": note,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params_fingerprint": provenance.json_fingerprint(params),
        "disturbance": {"file": params.get("disturbance_file"),
                        "sha256": provenance.sha256_file(config.resolve_path(
                            params.get("disturbance_file", ""))) if params.get("disturbance_file")
                        else None},
    }


def run_pair(
    params: dict,
    out_dir: Path | None = None,
    verbose: bool = True,
    do_analysis: bool = True,
    strict: bool = True,
    source_file: str | None = None,
    stop_path: Path | None = None,
    show: bool = False,
    pace_rate: float = 1.0,
    show_best_effort: bool = False,
) -> dict:
    """跑完整 A/B pair，写 manifest。出错：保留 provisional manifest 后抛异常。"""
    import run as run_mod

    errs = config.validate_params(params)
    if errs:
        raise ValueError("参数校验失败：\n  - " + "\n  - ".join(errs))

    out_dir = Path(out_dir) if out_dir else make_pair_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_a = out_dir / "run_A"
    run_b = out_dir / "run_B"
    analysis_dir = out_dir / "analysis"

    # 观赛 A/B 视觉公平：同一 10s 窗、同一对称纵轴（按冻结包算一次，A/B 同值）
    y_lim, y_src = _resolve_pair_y_limit(params)

    _write_manifest(out_dir, _provisional(out_dir, params, "pair 进行中"))
    if verbose:
        print(f"[pair] 输出目录 {out_dir}", flush=True)
        if show:
            print(f"[pair] 观赛：A/B 共用 window_s=10s、纵轴 ±{y_lim:g}µm（来源：{y_src}），"
                  f"标题 A baseline / B APF-SFC", flush=True)

    try:
        t0 = time.perf_counter()
        fa = run_mod.run_single(params, config.RUN_MODE_BASELINE, out_dir=run_a,
                                verbose=verbose, stop_path=stop_path,
                                show=show, pace_rate=pace_rate,
                                show_best_effort=show_best_effort,
                                window_s=10.0, y_limit_um=y_lim, y_limit_source=y_src,
                                show_label="A baseline")
        if show and verbose:
            print("\n[pair] —— A(baseline) 段结束；即将开启 B(APF-SFC) 段观赛 ——\n", flush=True)
        fb = run_mod.run_single(params, config.RUN_MODE_APF_SFC, out_dir=run_b,
                                verbose=verbose, stop_path=stop_path,
                                show=show, pace_rate=pace_rate,
                                show_best_effort=show_best_effort,
                                window_s=10.0, y_limit_um=y_lim, y_limit_source=y_src,
                                show_label="B APF-SFC")
        wall = time.perf_counter() - t0

        gate = gate_pair(run_a, run_b, strict=strict)
        if not fa["completed"] or not fb["completed"]:
            raise RuntimeError("A/B 未全部完成，正式 pair 无效。")
        if strict and not gate["ok"]:
            raise RuntimeError("A/B 门控未通过：\n  - " + "\n  - ".join(gate["problems"]))

        analysis_ok = False
        analysis_sha = None
        if do_analysis:
            try:
                res = analyze_pair(run_a, run_b, ["A", "B"], analysis_dir,
                                   strict=strict, profile_dir=None)
                analysis_ok = True
                analysis_sha = provenance.sha256_file(analysis_dir / "analysis_metrics.json")
            except Exception as exc:
                if strict:
                    raise
                print(f"[pair] 分析失败（非正式继续）：{exc}", file=sys.stderr, flush=True)

        payload = {
            "version": "v1.2",
            "provisional": False,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_s": round(wall, 1),
            "out_dir": str(out_dir),
            "params_fingerprint": provenance.json_fingerprint(params),
            "params_file": source_file,
            "disturbance": {
                "file": config.project_relative_str(config.resolve_path(params["disturbance_file"])),
                "sha256": provenance.sha256_file(config.resolve_path(params["disturbance_file"])),
            },
            "schedule": {"sha256": provenance.sha256_file(
                config.resolve_path(params["replay_schedule"]))
                if params.get("replay_schedule") else None},
            "model": provenance.model_fingerprint(),
            "init_q": [round(float(x), 12) for x in config.INIT_Q],
            "control_hz": float(params["control_hz"]),
            "physics_hz": int(params["physics_hz"]),
            "duration_s": float(params["duration_s"]),
            "runs": {
                "A": {"dir": str(run_a), "mode": "baseline",
                      "rows": fa["rows"], "t_end_s": round(fa["t_end"], 3)},
                "B": {"dir": str(run_b), "mode": "apf_sfc",
                      "rows": fb["rows"], "t_end_s": round(fb["t_end"], 3)},
            },
            "gate": gate,
            "analysis": {"done": analysis_ok,
                         "dir": str(analysis_dir) if analysis_ok else None,
                         "metrics_sha256": analysis_sha},
        }
        _write_manifest(out_dir, payload)
        if verbose:
            print(f"[pair] 完成：gate={gate['ok']}，analysis={analysis_ok}，耗时 {wall:.1f}s",
                  flush=True)
        return payload
    except Exception:
        _write_manifest(out_dir, _provisional(out_dir, params, f"失败：{sys.exc_info()[1]}"))
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="正式 A/B pair（唯一目录 + manifest + 门控）")
    p.add_argument("--params", required=True, help="replay_params.json 或任意参数 JSON")
    p.add_argument("--out", default="", help="指定 pair 目录（缺省自动唯一命名）")
    p.add_argument("--control-hz", type=float, default=None, help="覆盖控制频率（1000/132）")
    p.add_argument("--no-analysis", action="store_true", help="跑完不自动分析")
    p.add_argument("--no-strict", action="store_true", help="门控不过仍继续（标记非正式）")
    p.add_argument("--stop-path", default="", help="存在即优雅停止当前 run 的信号文件")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true",
                   help="严格观赛：A/B 各开 MuJoCo 3D + 实时 e_y 图；任一窗口未就绪在 t=0 前报错退出")
    g.add_argument("--show-best-effort", action="store_true",
                   help="尽力观赛：任一窗口失败则降级无头继续（不报错）")
    p.add_argument("--pace", type=_pos_pace, default=1.0,
                   help="真实时倍率（1=真实时；大值≈尽量快跑；仅与 --show/--show-best-effort 同用时生效）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = parse_args(argv)
    try:
        params = config.load_parameters(Path(args.params))
        if args.control_hz is not None:
            params["control_hz"] = float(args.control_hz)
        stop_path = Path(args.stop_path) if args.stop_path else None
        payload = run_pair(params,
                           out_dir=Path(args.out) if args.out else None,
                           do_analysis=not args.no_analysis,
                           strict=not args.no_strict,
                           source_file=str(Path(args.params).resolve()),
                           stop_path=stop_path,
                           show=bool(args.show or args.show_best_effort),
                           show_best_effort=args.show_best_effort,
                           pace_rate=args.pace)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[pair] 失败：{exc}", file=sys.stderr)
        return 1
    print(f"[pair] manifest -> {Path(payload['out_dir']) / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
