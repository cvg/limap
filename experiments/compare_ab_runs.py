#!/usr/bin/env python
"""Paired A/B comparison of two benchmark_sfm.py output trees.

For a code change that cannot be toggled at runtime, the two arms must come
from two builds. Both are run with the same dataset, scenes, config and method,
into two different --output_dir trees.

A point-only reference is reported alongside, since what matters is not only
whether the change moved the structure numbers but whether structure still
improves on points alone -- and by how much. That reference is unaffected by
line-side changes, so it only needs to exist in one of the two trees.

    # build A (baseline), then:
    python experiments/benchmark_sfm.py --dataset scannetpp --scenes <ids> \
        --methods holistic points_only --output_dir outputs/<run_A>

    # build B (with the change), then the same with --output_dir outputs/<run_B>

    python experiments/compare_ab_runs.py \
        --baseline outputs/<run_A> --treatment outputs/<run_B> \
        --dataset scannetpp
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))  # so this runs from any working directory

import limap.util.config as cfgutils  # noqa: E402
from benchmark_sfm import (  # noqa: E402
    METHODS,
    THRESHOLDS,
    evaluate_dataset,
)
from sfm_datasets import (  # noqa: E402
    DATASET_NAMES,
    DEFAULT_DATA_ROOTS,
    get_loader,
)


def _resolve(p):
    """Resolve a path against the repo root when it is not already valid."""
    p = Path(p).expanduser()
    return p if p.is_absolute() or p.exists() else _REPO / p


def collect(cfg, dataset, scenes, out_dir):
    """Return {(method_display_name, scene_id): row} for one output tree."""
    rows = evaluate_dataset(cfg, dataset, scenes, Path(out_dir), list(METHODS))
    return {(r["method"], r["scene"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--dataset", default="scannetpp", choices=DATASET_NAMES)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--method", default="Holistic", help="Method under test")
    ap.add_argument(
        "--reference",
        default="Points only",
        help="Point-only reference reported alongside (blank to disable)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="Threshold used for the per-scene table",
    )
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Source release root (default: the dataset's own, see "
        "sfm_datasets.DEFAULT_DATA_ROOTS)",
    )
    ap.add_argument(
        "-c",
        "--config_file",
        default="cfgs/structure_incremental_reconstruction/default.yaml",
    )
    args = ap.parse_args()

    config_file = str(_resolve(args.config_file))
    cfg = cfgutils.load_config(config_file, default_path=config_file)
    cfg["data_dir"] = str(
        _resolve(args.data_dir or DEFAULT_DATA_ROOTS[args.dataset])
    )

    loader = get_loader(args.dataset, cfg, _resolve(args.baseline))
    scenes = [
        s for s in loader.default_scenes if not args.scenes or s in args.scenes
    ]
    base = collect(cfg, args.dataset, scenes, _resolve(args.baseline))
    treat = collect(cfg, args.dataset, scenes, _resolve(args.treatment))

    m, ref = args.method, args.reference
    common = [s for s in scenes if (m, s) in base and (m, s) in treat]
    missing = [s for s in scenes if s not in common]
    if missing:
        print(f"WARNING: excluded (missing from an arm): {', '.join(missing)}")
    if not common:
        print("No scenes present in both arms -- nothing to compare.")
        return

    def ref_auc(scene, t):
        """Point-only AUC, from whichever tree has it."""
        if not ref:
            return None
        for tree in (treat, base):
            row = tree.get((ref, scene))
            if row is not None and row["auc"] is not None:
                return row["auc"][t]
        return None

    t0 = (
        args.threshold
        if args.threshold in THRESHOLDS
        else THRESHOLDS[len(THRESHOLDS) // 2]
    )

    print(f"\nPer-scene AUC@{t0} (percent of pairs under threshold)")
    hdr = (
        f"{'Scene':<16}| {'#img':>5} | {'PtsOnly':>8} | {'A base':>8} "
        f"| {'B treat':>8} | {'A-Pts':>7} | {'B-Pts':>7} | {'B-A':>7} |"
    )
    sep = "-" * len(hdr)
    print(sep + "\n" + hdr + "\n" + sep)

    agg = {
        t: {"a": 0.0, "b": 0.0, "r": 0.0, "nr": 0, "win": 0, "loss": 0}
        for t in THRESHOLDS
    }
    d_reg = 0
    for s in common:
        a_row, b_row = base[(m, s)], treat[(m, s)]
        a, b = a_row["auc"], b_row["auc"]
        if a is None or b is None:
            print(f"{s:<16}| no GT -- nothing to compare")
            continue
        r = ref_auc(s, t0)
        d_reg += b_row["n_reg"] - a_row["n_reg"]
        rs = f"{r:>8.1f}" if r is not None else f"{'N/A':>8}"
        ap_ = f"{a[t0] - r:>+7.2f}" if r is not None else f"{'N/A':>7}"
        bp_ = f"{b[t0] - r:>+7.2f}" if r is not None else f"{'N/A':>7}"
        print(
            f"{s:<16}| {b_row['n_reg']:>5} | {rs} | {a[t0]:>8.1f} "
            f"| {b[t0]:>8.1f} | {ap_} | {bp_} "
            f"| {b[t0] - a[t0]:>+7.2f} |"
        )
        for t in THRESHOLDS:
            agg[t]["a"] += a[t]
            agg[t]["b"] += b[t]
            rt = ref_auc(s, t)
            if rt is not None:
                agg[t]["r"] += rt
                agg[t]["nr"] += 1
            d = b[t] - a[t]
            if d > 1e-9:
                agg[t]["win"] += 1
            elif d < -1e-9:
                agg[t]["loss"] += 1
    print(sep)

    n = len(common)
    print(f"\nAggregate over {n} scene(s); d#Reg (B-A) = {d_reg:+d}")
    hdr2 = (
        f"{'Threshold':<12}| {'PtsOnly':>8} | {'A base':>8} | {'B treat':>8} "
        f"| {'A-Pts':>7} | {'B-Pts':>7} | {'B-A':>7} | {'W/L':>7} |"
    )
    sep2 = "-" * len(hdr2)
    print(sep2 + "\n" + hdr2 + "\n" + sep2)
    for t in THRESHOLDS:
        g = agg[t]
        a, b = g["a"] / n, g["b"] / n
        if g["nr"]:
            r = g["r"] / g["nr"]
            rs, ap_, bp_ = f"{r:>8.1f}", f"{a - r:>+7.2f}", f"{b - r:>+7.2f}"
        else:
            rs = ap_ = bp_ = f"{'N/A':>7}"
        print(
            f"AUC@{t:<8}| {rs} | {a:>8.1f} | {b:>8.1f} | {ap_} | {bp_} "
            f"| {b - a:>+7.2f} | {str(g['win']) + '/' + str(g['loss']):>7} |"
        )
    print(sep2)
    print(
        "\nA-Pts / B-Pts are the lift of structure over the point-only "
        "baseline;\nB-A is the effect of the change itself. "
        "W/L counts scenes better/worse in B."
    )


if __name__ == "__main__":
    main()
