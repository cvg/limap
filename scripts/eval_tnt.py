import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.evaluation as limap_eval
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis


def plot_curve(fname, thresholds, data):
    plt.plot(thresholds, data)
    plt.savefig(fname)


def report_error_to_GT(evaluator, lines):
    lengths = np.array([line.length() for line in lines])
    thresholds = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    list_recall, list_precision = [], []
    for threshold in thresholds:
        ratios = np.array(
            [evaluator.ComputeInlierRatio(line, threshold) for line in lines]
        )
        length_recall = (lengths * ratios).sum()
        list_recall.append(length_recall)
        precision = 100 * (ratios > 0).astype(int).sum() / ratios.shape[0]
        list_precision.append(precision)
    for idx, threshold in enumerate(thresholds):
        logging.info(
            f"R / P at {int(threshold * 1000)}mm: "
            f"{list_recall[idx]:.2f} / {list_precision[idx]:.2f}"
        )
    return evaluator


def report_pc_recall_for_GT(evaluator, lines):
    """
    To compute invert point recall
    """
    thresholds = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    point_dists = evaluator.ComputeDistsforEachPoint(lines)
    # point_dists = evaluator.ComputeDistsforEachPoint_KDTree(lines)
    point_dists = np.array(point_dists)
    n_points = point_dists.shape[0]
    logging.info("Compute point recall metrics.")
    for threshold in thresholds.tolist():
        num_inliers = (point_dists < threshold).sum()
        point_recall = 100 * num_inliers / n_points
        logging.info(
            f"{int(threshold * 1000):.0f}mm, inliers = {num_inliers}, "
            f"point recall = {point_recall:.2f}"
        )
    return evaluator


def read_ply(fname):
    from plyfile import PlyData

    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data["x"])
    y = np.asarray(plydata.elements[0].data["y"])
    z = np.asarray(plydata.elements[0].data["z"])
    points = np.stack([x, y, z], axis=1)
    logging.info(f"number of points: {points.shape[0]}")
    return points


def write_ply(fname, points):
    from plyfile import PlyData, PlyElement

    points = [
        (points[i, 0], points[i, 1], points[i, 2])
        for i in range(points.shape[0])
    ]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    PlyData([el], text=True).write(fname)


def report_error_to_mesh(mesh_fname, lines):
    # Hypersim
    MPAU = 0.02539999969303608
    evaluator = limap_eval.MeshEvaluator(mesh_fname, MPAU)
    return report_error_to_GT(evaluator, lines)


def report_error_to_point_cloud(points, lines, kdtree_dir=None):
    evaluator = limap_eval.PointCloudEvaluator(points)
    if kdtree_dir is None:
        evaluator.Build()
        evaluator.Save("tmp/kdtree.bin")
    else:
        evaluator.Load(kdtree_dir)
    # evaluator = report_pc_recall_for_GT(evaluator, lines)
    evaluator = report_error_to_GT(evaluator, lines)
    return evaluator


def eval_tnt(cfg, lines, ref_lines=None):
    # eval w.r.t psuedo gt lines
    if ref_lines is not None:
        pass
    if cfg["mesh_dir"] is not None:
        # eval w.r.t mesh
        evaluator = report_error_to_mesh(cfg["mesh_dir"], lines)
    elif cfg["pc_dir"] is not None:
        points = read_ply(cfg["pc_dir"])
        if cfg["use_ranges"]:
            n_lines = len(lines)
            ranges = [points.min(0) - 0.1, points.max(0) + 0.1]
            lines = [
                line
                for line in lines
                if limapvis.test_line_inside_ranges(line, ranges)
            ]
            logging.info(f"Filtering by range: {len(lines)} / {n_lines}")
        evaluator = report_error_to_point_cloud(
            points, lines, kdtree_dir=cfg["kdtree_dir"]
        )
    else:
        raise NotImplementedError
    if cfg["visualize"]:
        thresholds = np.arange(1, 11, 1) * 0.001
        for threshold in tqdm(thresholds.tolist()):
            inlier_lines = evaluator.ComputeInlierSegs(lines, threshold)
            inlier_lines_np = np.array(
                [line.as_array() for line in inlier_lines]
            )
            limapio.save_obj(
                f"tmp/inliers_th_{threshold:.4f}.obj", inlier_lines_np
            )
            outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)
            outlier_lines_np = np.array(
                [line.as_array() for line in outlier_lines]
            )
            limapio.save_obj(
                f"tmp/outliers_th_{threshold:.4f}.obj", outlier_lines_np
            )


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="eval 3d lines on hypersim"
    )
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/eval/tnt.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/eval/default.yaml",
        help="default config file",
    )
    arg_parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input line file [*.npy, *.obj, [folder-to-linetracks]]",
    )
    arg_parser.add_argument(
        "-r",
        "--reference_dir",
        type=str,
        default=None,
        help="reference pseudo gt lines [*.npy, *.obj]",
    )
    arg_parser.add_argument(
        "--mesh_dir", type=str, default=None, help="path to the .obj file"
    )
    arg_parser.add_argument(
        "--pc_dir", type=str, default=None, help="path to the .ply file"
    )
    arg_parser.add_argument(
        "--kdtree_dir",
        type=str,
        default=None,
        help="path to the saved index for kd tree",
    )
    arg_parser.add_argument(
        "--noeval",
        action="store_true",
        help="if enabled, the evaluation is not performed",
    )
    arg_parser.add_argument(
        "--transform_txt", type=str, default=None, help="txt to transformation"
    )
    arg_parser.add_argument(
        "--use_ranges", action="store_true", help="use ranges for testing"
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_dir"] = args.input_dir
    cfg["reference_dir"] = args.reference_dir
    cfg["mesh_dir"] = args.mesh_dir
    cfg["pc_dir"] = args.pc_dir
    cfg["kdtree_dir"] = args.kdtree_dir
    cfg["noeval"] = args.noeval
    cfg["transform_txt"] = args.transform_txt
    cfg["use_ranges"] = args.use_ranges
    return cfg


def transform_lines(fname, lines):
    with open(fname) as f:
        flines = f.readlines()
    mat = []
    for fline in flines:
        fline = fline.strip().split()
        mat.append([float(k) for k in fline])
    trans = np.array(mat)
    new_lines = []
    for line in lines:
        newstart = trans[:3, :3] @ line.start + trans[:3, 3]
        newend = trans[:3, :3] @ line.end + trans[:3, 3]
        newline = base.Line3d(newstart, newend)
        new_lines.append(newline)
    return new_lines


def init_workspace():
    if not os.path.exists("tmp"):
        os.makedirs("tmp")


def main():
    cfg = parse_config()
    init_workspace()

    # read lines
    lines, linetracks = limapio.read_lines_from_input(cfg["input_dir"])
    if linetracks is not None:
        lines = [
            track.line
            for track in linetracks
            if track.count_images() >= cfg["n_visible_views"]
        ]
        linetracks = [
            track
            for track in linetracks
            if track.count_images() >= cfg["n_visible_views"]
        ]
    if cfg["transform_txt"]:
        lines = transform_lines(cfg["transform_txt"], lines)
        limapio.save_obj("tmp/lines_transform.obj", lines)
    if cfg["noeval"]:
        return
    ref_lines = None
    if cfg["reference_dir"] is not None:
        ref_lines = limapio.read_lines_from_input(
            cfg["reference_dir"], n_visible_views=4
        )
    eval_tnt(cfg, lines, ref_lines=ref_lines)

    # report track quality
    if linetracks is not None:
        sup_image_counts = np.array(
            [track.count_images() for track in linetracks]
        )
        sup_line_counts = np.array(
            [track.count_lines() for track in linetracks]
        )
        logging.info(
            f"supporting images / lines: ({sup_image_counts.mean():.2f} "
            f"/ {sup_line_counts.mean():.2f})"
        )


if __name__ == "__main__":
    main()
