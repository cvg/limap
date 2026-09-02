#!/usr/bin/env python
"""Benchmark structure-aware incremental SfM against baselines.

One driver for every SfM dataset. Each scene is reconstructed by one or more
methods, then estimated poses are scored against ground truth with relative
pose error AUC. 1DSfM has no reliable GT, so it reports runtime and
reconstruction size only.

Datasets (--dataset), each read straight from its own public release:
    hypersim    Hypersim
    scannetpp   DA3-BENCH (depth-anything/DA3-BENCH)
    7scenes     DA3-BENCH, same root
    eth3d       ETH3D DSLR, undistorted release
    1dsfm       internet photo collections (retrieval-based, no GT)

Views needing undistortion or resizing are prepared once and cached under
<output_dir>/<dataset>/<scene>/images/, so reruns skip that work.

Methods (--methods). Default is `holistic` alone:
    holistic        full pipeline: points + lines + groups + wireframe
    points_only     our mapper with line and group triangulation switched off
    pycolmap        the external baseline: COLMAP's own incremental mapper,
                    via `pycolmap.incremental_mapping`, with the intrinsics
                    treated as in the other arms

Only `holistic` runs the frontend; the other arms reuse the databases it
wrote, so every arm sees identical features and matches and the comparison
isolates the mapper. Run `holistic` first (or in the same invocation).

The run is split into two phases so that one crash does not lose the rest:
reconstruction catches exceptions per scene/method, and evaluation reads
models back from disk. `--eval_only` runs the second phase alone.

Output layout:
    <output_dir>/<dataset>/<scene>/images/
    <output_dir>/<dataset>/<scene>/<method_subdir>/{models,timing.json}

Usage:
    # One dataset, the full method
    GLOG_v=1 python experiments/benchmark_sfm.py \
        --dataset hypersim \
        --output_dir outputs/benchmark_sfm \
        --skip_exists

    # With the point-only baseline (reuses the structure run's frontend)
    GLOG_v=1 python experiments/benchmark_sfm.py \
        --dataset hypersim --methods holistic points_only \
        --output_dir outputs/benchmark_sfm --skip_exists

    # Everything
    GLOG_v=1 python experiments/benchmark_sfm.py --dataset all \
        --output_dir outputs/benchmark_sfm --skip_exists

    # Re-print the tables without re-running
    python experiments/benchmark_sfm.py --dataset all \
        --output_dir outputs/benchmark_sfm --eval_only
"""

import argparse
import copy
import json
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from dacite import Config, from_dict
from pycolmap import logging

import limap.runners
import limap.util.config as cfgutils
from sfm_datasets import DATASET_NAMES, DEFAULT_DATA_ROOTS, get_loader

THRESHOLDS = [0.25, 0.5, 1, 3, 5, 10]

# method key -> (display name, output subdirectory)
METHODS = {
    "holistic": ("Holistic", "holistic"),
    "points_only": ("Points only", "points_only"),
    "pycolmap": ("pycolmap", "pycolmap"),
}

# The arm that runs the frontend; every other arm reuses what it wrote.
FRONTEND_METHOD = "holistic"
# Only the full method runs by default. The rest are ablation arms: they
# reuse the `structure` run's frontend, so they must be asked for explicitly
# and only after (or alongside) `structure`.
DEFAULT_METHODS = [FRONTEND_METHOD]


# ---------------------------------------------------------------------------
# Method arms
# ---------------------------------------------------------------------------
def _make_options(cfg):
    """Build reconstruction options from the config dict."""
    options = from_dict(
        data_class=limap.runners.AutomaticStructureIncrementalReconstructionOptions,
        data=cfg,
        config=Config(strict=False, cast=[Path]),
    )
    options.colmap_options.setdefault("ba_refine_focal_length", False)
    options.colmap_options.setdefault("ba_refine_principal_point", False)
    options.colmap_options.setdefault("ba_refine_extra_params", False)
    options.colmap_options.setdefault("multiple_models", False)
    return options


def _arm_points_only(options):
    """Our mapper with line and group triangulation switched off.

    An ablation of our own pipeline. Use the `pycolmap` arm to compare
    against COLMAP itself.
    """
    options.structure_options["use_lines_for_registration"] = False
    options.structure_options.setdefault("triangulation", {})
    options.structure_options["triangulation"]["triangulate_lines"] = False
    options.structure_options["triangulation"]["triangulate_groups"] = False


METHOD_ARMS = {
    "holistic": None,
    "points_only": _arm_points_only,
    "pycolmap": None,
}


def _frontend_paths(output_dir, scene):
    """Frontend databases, always those written by the `holistic` run."""
    _, subdir = METHODS[FRONTEND_METHOD]
    d = output_dir / scene.dataset / scene.scene_id / subdir
    return d / "database.db", d / "structure_database.db"


def run_method(cfg, scene, output_dir, method):
    """Run one arm of the pipeline on a scene.

    `holistic` runs the frontend and reports its cost; every other arm
    reuses the databases it wrote.

    Returns a timing dict.
    """
    _, subdir = METHODS[method]
    result_dir = output_dir / scene.dataset / scene.scene_id / subdir
    result_dir.mkdir(parents=True, exist_ok=True)

    options = _make_options(copy.deepcopy(cfg))
    arm = METHOD_ARMS[method]
    if arm is not None:
        arm(options)

    # The pipeline mutates the reconstruction it is given.
    recon = copy.deepcopy(scene.recon)
    timing = {}

    if method == FRONTEND_METHOD:
        neighbors = None
        if scene.neighbors_fn is not None:
            neighbors, retrieval_sec = scene.neighbors_fn(result_dir)
            timing["retrieval_sec"] = retrieval_sec

        frontend_options = limap.runners.StructureFrontendOptions(
            max_image_dim=options.max_image_dim,
            metainfo=options.metainfo,
            image_description=options.image_description,
            image_association=options.image_association,
        )
        t0 = time.time()
        frontend_outputs = limap.runners.structure_frontend_from_images(
            frontend_options,
            scene.image_dir,
            recon,
            result_dir,
            neighbors=neighbors,
            skip_if_exists=options.skip_frontend_if_exists_database,
        )
        timing["frontend_sec"] = time.time() - t0
        db_path = frontend_outputs.db_path
        structure_db_path = frontend_outputs.structure_db_path
    else:
        db_path, structure_db_path = _frontend_paths(output_dir, scene)

    if method == "pycolmap":
        models = result_dir / "models"
        if models.exists():
            shutil.rmtree(models)
        models.mkdir(parents=True, exist_ok=True)
        copts = pycolmap.IncrementalPipelineOptions()
        # Refine the focal only when it is not given.
        uncalibrated = cfg.get("uncalibrated", False)
        copts.ba_refine_focal_length = uncalibrated
        copts.ba_refine_principal_point = False
        copts.ba_refine_extra_params = False
        copts.multiple_models = False
        # -1 leaves the global PRNG seeded as it was, which makes repeated
        # runs identical. Pass a seed to vary them.
        seed = cfg.get("colmap_options", {}).get("random_seed")
        if seed is not None:
            copts.random_seed = int(seed)
            copts.mapper.random_seed = int(seed)
        t0 = time.time()
        pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(scene.image_dir),
            output_path=str(models),
            options=copts,
        )
        timing["sfm_sec"] = time.time() - t0
        return timing

    t0 = time.time()
    limap.runners.automatic_structure_incremental_reconstruction(
        options,
        scene.image_dir,
        result_dir,
        recon,
        db_path=db_path,
        structure_db_path=structure_db_path,
    )
    timing["sfm_sec"] = time.time() - t0
    return timing


# ---------------------------------------------------------------------------
# Phase 1: reconstruct
# ---------------------------------------------------------------------------
def _has_results(result_dir):
    timing = result_dir / "timing.json"
    models = result_dir / "models"
    return timing.exists() and models.exists() and any(models.iterdir())


def run_dataset(cfg, dataset_name, scenes, output_dir, methods, skip_exists):
    """Run the selected methods over one dataset's scenes."""
    loader = get_loader(dataset_name, cfg, output_dir)

    for scene_id in scenes:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"[{dataset_name}] Scene: {scene_id}")
        logging.info(f"{'=' * 60}")

        try:
            scene = loader.load(scene_id)
        except Exception:
            logging.error(f"Failed to load scene {dataset_name}/{scene_id}:")
            traceback.print_exc()
            continue

        for method in methods:
            display, subdir = METHODS[method]
            result_dir = output_dir / dataset_name / scene_id / subdir

            if skip_exists and _has_results(result_dir):
                logging.info(f"--- {display}: skipping (results exist) ---")
                continue

            # Every derived method reads the frontend the structure run wrote.
            if method != FRONTEND_METHOD:
                db_path, _ = _frontend_paths(output_dir, scene)
                if not db_path.exists():
                    logging.warning(
                        f"Skipping {display} for {dataset_name}/{scene_id}: "
                        f"no frontend database at {db_path}. "
                        f"Run the '{FRONTEND_METHOD}' method first."
                    )
                    continue

            logging.info(f"--- Running {display} ---")
            try:
                timing = run_method(cfg, scene, output_dir, method)
                (result_dir / "timing.json").write_text(json.dumps(timing))
                logging.info(
                    f"{display} finished: "
                    + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items())
                )
            except Exception:
                logging.error(
                    f"{display} FAILED for {dataset_name}/{scene_id}:"
                )
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Phase 2: evaluate
# ---------------------------------------------------------------------------
def _rot_angle_deg(R):
    rod = cv2.Rodrigues(R)[0]
    return float(np.linalg.norm(rod) * 180.0 / np.pi)


def compute_relpose_errors(rec, gt_poses, index_list):
    """Max of relative rotation and translation-direction error, per pair.

    Image pairs where either view failed to register score the worst
    possible error (180 deg), so that dropping images is penalised.
    """
    reg_ids = set(rec.reg_image_ids()) if rec is not None else set()
    est = {}
    for img_id in index_list:
        if img_id in reg_ids:
            mat = np.asarray(rec.images[img_id].cam_from_world().matrix())
            est[img_id] = (mat[:3, :3], mat[:3, 3])

    errs = []
    n = len(index_list)
    for i in range(n - 1):
        id_i = index_list[i]
        for j in range(i + 1, n):
            id_j = index_list[j]
            if id_i not in est or id_j not in est:
                errs.append(180.0)
                continue
            Ri, Ti = est[id_i]
            Rj, Tj = est[id_j]
            Ri_gt, Ti_gt = gt_poses[id_i]
            Rj_gt, Tj_gt = gt_poses[id_j]

            relR = Ri @ Rj.T
            relT = Ti - relR @ Tj
            relT_n = np.linalg.norm(relT)
            relT_vec = relT / relT_n if relT_n > 1e-10 else relT

            relR_gt = Ri_gt @ Rj_gt.T
            relT_gt = Ti_gt - relR_gt @ Tj_gt
            relT_gt_n = np.linalg.norm(relT_gt)
            relT_gt_vec = relT_gt / relT_gt_n if relT_gt_n > 1e-10 else relT_gt

            rot_err = _rot_angle_deg(relR.T @ relR_gt)
            t_angle = float(
                np.arccos(np.clip(np.abs(relT_vec @ relT_gt_vec), -1, 1))
                * 180.0
                / np.pi
            )
            errs.append(max(rot_err, t_angle))
    return np.array(errs)


def compute_auc(errors, thresholds):
    return {t: float((errors < t).mean() * 100) for t in thresholds}


def load_best_reconstruction(models_dir):
    """Largest reconstruction under models_dir, or None."""
    if not models_dir.exists():
        return None
    best_rec, best_n = None, 0
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            rec = pycolmap.Reconstruction(sub)
        except Exception:
            continue
        n = rec.num_reg_images()
        if n > best_n:
            best_rec, best_n = rec, n
    return best_rec


def evaluate_dataset(cfg, dataset_name, scenes, output_dir, methods):
    """Score every method on every scene. Returns a list of row dicts."""
    loader = get_loader(dataset_name, cfg, output_dir)
    rows = []

    for scene_id in scenes:
        try:
            scene = loader.load(scene_id, with_images=False)
        except Exception:
            logging.error(f"Failed to load GT for {dataset_name}/{scene_id}:")
            traceback.print_exc()
            continue

        for method in methods:
            display, subdir = METHODS[method]
            result_dir = output_dir / dataset_name / scene_id / subdir
            if not (result_dir / "models").exists():
                continue

            rec = load_best_reconstruction(result_dir / "models")
            timing_path = result_dir / "timing.json"
            timing = (
                json.loads(timing_path.read_text())
                if timing_path.exists()
                else {}
            )

            auc = None
            if scene.has_gt:
                errs = compute_relpose_errors(
                    rec, scene.gt_poses, scene.index_list
                )
                auc = compute_auc(errs, THRESHOLDS)

            rows.append(
                {
                    "dataset": dataset_name,
                    "scene": scene_id,
                    "method": display,
                    "n_reg": rec.num_reg_images() if rec else 0,
                    "auc": auc,
                    "retrieval_sec": timing.get("retrieval_sec"),
                    "frontend_sec": timing.get("frontend_sec"),
                    "sfm_sec": timing.get("sfm_sec"),
                }
            )

    return rows


def _fmt_time(sec):
    return f"{sec:>9.1f}s" if sec is not None else "       N/A"


def _mean_time(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{np.mean(vals):>9.1f}s" if vals else "       N/A"


def print_summary(rows, methods):
    """Print a per-dataset table, with AUC columns where GT exists."""
    if not rows:
        print("\nNo results found.")
        return

    method_names = [METHODS[m][0] for m in methods]
    show_retrieval = any(r["retrieval_sec"] is not None for r in rows)
    show_auc = any(r["auc"] is not None for r in rows)

    hdr = f"{'Dataset':<12}| {'Scene':<22}| {'Method':<14}| {'#Reg':>5} "
    if show_retrieval:
        hdr += f"| {'Retrieval':>10} "
    hdr += f"| {'Frontend':>10} | {'SfM':>10} |"
    if show_auc:
        for t in THRESHOLDS:
            hdr += f" AUC@{t:<4}|"
    sep = "-" * len(hdr)

    def _emit(
        dataset_label, scene_label, method, n_reg, auc, retrieval, frontend, sfm
    ):
        line = (
            f"{dataset_label:<12}| {scene_label:<22}| {method:<14}| {n_reg:>5} "
        )
        if show_retrieval:
            line += f"| {retrieval} "
        line += f"| {frontend} | {sfm} |"
        if show_auc:
            for t in THRESHOLDS:
                line += f" {auc[t]:>6.1f} |" if auc else "     -  |"
        print(line)

    print(f"\n{sep}")
    print(hdr)
    print(sep)

    datasets_seen = list(dict.fromkeys(r["dataset"] for r in rows))
    for dataset in datasets_seen:
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        for r in dataset_rows:
            _emit(
                r["dataset"],
                r["scene"],
                r["method"],
                r["n_reg"],
                r["auc"],
                _fmt_time(r["retrieval_sec"]),
                _fmt_time(r["frontend_sec"]),
                _fmt_time(r["sfm_sec"]),
            )

        print(sep)
        for name in method_names:
            method_rows = [r for r in dataset_rows if r["method"] == name]
            if not method_rows:
                continue
            aucs = [r["auc"] for r in method_rows if r["auc"] is not None]
            avg_auc = (
                {t: np.mean([a[t] for a in aucs]) for t in THRESHOLDS}
                if aucs
                else None
            )
            n_regs = [r["n_reg"] for r in method_rows if r["n_reg"] > 0]
            _emit(
                f"{dataset} avg",
                "",
                name,
                int(np.mean(n_regs)) if n_regs else 0,
                avg_auc,
                _mean_time(method_rows, "retrieval_sec"),
                _mean_time(method_rows, "frontend_sec"),
                _mean_time(method_rows, "sfm_sec"),
            )
        print(sep)

    if len(datasets_seen) > 1:
        for name in method_names:
            method_rows = [r for r in rows if r["method"] == name]
            if not method_rows:
                continue
            aucs = [r["auc"] for r in method_rows if r["auc"] is not None]
            avg_auc = (
                {t: np.mean([a[t] for a in aucs]) for t in THRESHOLDS}
                if aucs
                else None
            )
            n_regs = [r["n_reg"] for r in method_rows if r["n_reg"] > 0]
            _emit(
                "Overall",
                "",
                name,
                int(np.mean(n_regs)) if n_regs else 0,
                avg_auc,
                _mean_time(method_rows, "retrieval_sec"),
                _mean_time(method_rows, "frontend_sec"),
                _mean_time(method_rows, "sfm_sec"),
            )
        print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark structure-aware incremental SfM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["hypersim"],
        help="Datasets to run, or 'all'. Choose from: "
        f"{' '.join(DATASET_NAMES)}",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="Scene ids to run (default: the dataset's own list)",
    )
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Methods to run. Choose from: {' '.join(METHODS)}",
    )
    ap.add_argument("--output_dir", type=str, default="outputs/benchmark_sfm")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Source release root for the selected dataset. Defaults per "
        "dataset: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_DATA_ROOTS.items()),
    )
    ap.add_argument(
        "--target_views",
        type=int,
        default=None,
        help="Views per scene to sample (default: 100 for scannetpp and "
        "7scenes, all for eth3d). 0 keeps all.",
    )
    ap.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Drop the focal-length prior on the input cameras "
        "(has_prior_focal_length=False). Default is calibrated: the "
        "priors are kept, which stops COLMAP's shared-focal solver "
        "from taking over calibrated pairs.",
    )
    ap.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=str(CONFIG_DIR / "default.yaml"),
        help="Override the config. By default each dataset uses its own "
        "<dataset>.yaml when one exists, else default.yaml.",
    )
    ap.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/structure_incremental_reconstruction/default.yaml",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip reconstruction, only read models back and evaluate",
    )
    ap.add_argument(
        "--skip_exists",
        action="store_true",
        help="Skip scene/method pairs that already have results",
    )
    ap.add_argument(
        "--all_1dsfm_scenes",
        action="store_true",
        help="Use all eligible 1DSfM scenes instead of the 3 smallest",
    )

    args, unknown = ap.parse_known_args()

    if "all" in args.dataset:
        args.dataset = list(DATASET_NAMES)
    unknown_datasets = [d for d in args.dataset if d not in DATASET_NAMES]
    if unknown_datasets:
        ap.error(
            f"unknown dataset(s): {' '.join(unknown_datasets)}. "
            f"Choose from: {' '.join(DATASET_NAMES)}"
        )
    unknown_methods = [m for m in args.methods if m not in METHODS]
    if unknown_methods:
        ap.error(
            f"unknown method(s): {' '.join(unknown_methods)}. "
            f"Choose from: {' '.join(METHODS)}"
        )

    # Config is resolved per dataset (see _cfg_for), since a dataset may ship
    # its own -- Hypersim's carries input_n_views / input_stride / cam_id,
    # which the default config does not define.
    args.overrides = unknown
    return args


CONFIG_DIR = Path("cfgs/structure_incremental_reconstruction")


def _cfg_for(dataset_name, args):
    """Config for one dataset: its own yaml if it ships one, else the default.

    Resolved per dataset rather than once, so `--dataset hypersim scannetpp`
    gives each the config it needs instead of forcing one on both.
    """
    config_file = args.config_file
    if config_file == str(CONFIG_DIR / "default.yaml"):
        dataset_cfg = CONFIG_DIR / f"{dataset_name}.yaml"
        if dataset_cfg.exists():
            config_file = str(dataset_cfg)

    cfg = cfgutils.load_config(
        config_file, default_path=args.default_config_file
    )
    shortcuts = {"-nv": "--n_visible_views", "-nn": "--n_neighbors"}
    cfg = cfgutils.update_config(cfg, args.overrides, shortcuts)
    cfg["output_dir"] = args.output_dir
    cfg["data_dir"] = _data_dir_for(dataset_name, args)
    if args.target_views is not None:
        cfg["target_views"] = args.target_views
    # Always set explicitly: the loaders read this key, and leaving it absent
    # makes "not requested" and "requested calibrated" indistinguishable.
    cfg["uncalibrated"] = args.uncalibrated
    return cfg


def _scenes_for(dataset_name, cfg, args, output_dir):
    """Scene list for one dataset, honouring --scenes as a filter."""
    loader = get_loader(dataset_name, cfg, output_dir)
    available = list(loader.default_scenes)
    if dataset_name == "1dsfm" and args.all_1dsfm_scenes:
        available = list(loader.all_scenes)
    if args.scenes is None:
        return available
    # --scenes spans datasets, so keep only the ids this dataset knows.
    return [s for s in args.scenes if s in available]


def _data_dir_for(dataset_name, args):
    """Source release root for one dataset."""
    root = args.data_dir or DEFAULT_DATA_ROOTS[dataset_name]
    return str(Path(root).expanduser())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset_name in args.dataset:
        dataset_cfg = _cfg_for(dataset_name, args)

        scenes = _scenes_for(dataset_name, dataset_cfg, args, output_dir)
        if not scenes:
            logging.warning(f"No scenes selected for {dataset_name}, skipping")
            continue

        if not args.eval_only:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Phase 1: reconstruction — {dataset_name}")
            logging.info(f"{'=' * 60}")
            run_dataset(
                dataset_cfg,
                dataset_name,
                scenes,
                output_dir,
                args.methods,
                args.skip_exists,
            )

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Phase 2: evaluation — {dataset_name}")
        logging.info(f"{'=' * 60}")
        all_rows += evaluate_dataset(
            dataset_cfg, dataset_name, scenes, output_dir, args.methods
        )

    print_summary(all_rows, args.methods)

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(all_rows, indent=2))
    logging.info(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
