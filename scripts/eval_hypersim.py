import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from pycolmap import logging
from tqdm import tqdm

import limap.evaluation as limap_eval
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis
from runners.hypersim.Hypersim import Hypersim

# hypersim
MPAU = 0.02539999969303608


def plot_curve(fname, thresholds, data):
    plt.plot(thresholds, data)
    plt.savefig(fname)


def visualize_error_to_GT(evaluator, lines, threshold):
    # get inlier and outlier segments
    inlier_lines = evaluator.ComputeInlierSegs(lines, threshold)
    outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)

    # visualize
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    inlier_line_set = limapvis.open3d_get_line_set(
        inlier_lines, color=[0.0, 1.0, 0.0]
    )
    vis.add_geometry(inlier_line_set)
    outlier_line_set = limapvis.open3d_get_line_set(
        outlier_lines, color=[1.0, 0.0, 0.0]
    )
    vis.add_geometry(outlier_line_set)
    vis.run()
    vis.destroy_window()


def report_error_to_GT(evaluator, lines, vis_err_th=None):
    # [optional] visualize
    if vis_err_th is not None:
        visualize_error_to_GT(evaluator, lines, vis_err_th)
    lengths = np.array([line.length() for line in lines])
    thresholds = [0.001, 0.005, 0.01]
    list_recall, list_precision = [], []
    for threshold in thresholds:
        ratios = np.array(
            [evaluator.ComputeInlierRatio(line, threshold) for line in lines]
        )
        length_recall = (lengths * ratios).sum()
        list_recall.append(length_recall)
        precision = 100 * (ratios > 0).astype(int).sum() / ratios.shape[0]
        list_precision.append(precision)
    logging.info("R: recall, P: precision")
    for idx, threshold in enumerate(thresholds):
        logging.info(
            f"R / P at {int(threshold * 1000)}mm: "
            f"{list_recall[idx]:.2f} / {list_precision[idx]:.2f}"
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


def report_error_to_mesh(mesh_fname, lines, vis_err_th=None):
    evaluator = limap_eval.MeshEvaluator(mesh_fname, MPAU)
    return report_error_to_GT(evaluator, lines, vis_err_th=vis_err_th)


def report_error_to_point_cloud(
    points, lines, kdtree_dir=None, vis_err_th=None
):
    evaluator = limap_eval.PointCloudEvaluator(points, vis_err_th=vis_err_th)
    if kdtree_dir is None:
        evaluator.Build()
        evaluator.Save("tmp/kdtree.bin")
    else:
        evaluator.Load(kdtree_dir)
    return report_error_to_GT(evaluator, lines)


def eval_hypersim(
    cfg, lines, dataset_hypersim, scene_id, cam_id=0, vis_err_th=None
):
    # set scene id
    dataset_hypersim.set_scene_id(scene_id)
    dataset_hypersim.set_max_dim(cfg["max_image_dim"])
    dataset_hypersim.load_cameras(cam_id=cam_id)

    # generate image indexes
    index_list = np.arange(
        0, cfg["input_n_views"], cfg["input_stride"]
    ).tolist()
    index_list = dataset_hypersim.filter_index_list(index_list, cam_id=cam_id)

    if cfg["mesh_dir"] is not None:
        # eval w.r.t mesh
        evaluator = report_error_to_mesh(
            cfg["mesh_dir"], lines, vis_err_th=vis_err_th
        )
    elif cfg["pc_dir"] is not None:
        points = read_ply(cfg["pc_dir"])
        evaluator = report_error_to_point_cloud(
            points, lines, kdtree_dir=cfg["kdtree_dir"], vis_err_th=vis_err_th
        )
    else:
        # eval w.r.t point cloud
        points = dataset_hypersim.get_point_cloud_from_list(
            index_list, cam_id=cam_id
        )
        evaluator = report_error_to_point_cloud(
            points, lines, kdtree_dir=cfg["kdtree_dir"], vis_err_th=vis_err_th
        )
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
        default="cfgs/eval/hypersim.yaml",
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
        "--ref_dir",
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
        "--visualize_error",
        action="store_true",
        help="whether to visualize accurate lines",
    )
    arg_parser.add_argument(
        "--visualize_error_threshold",
        type=float,
        default=0.001,
        help="visualize threshold",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file
    )
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_dir"] = args.input_dir
    cfg["ref_dir"] = args.ref_dir
    cfg["mesh_dir"] = args.mesh_dir
    cfg["pc_dir"] = args.pc_dir
    cfg["kdtree_dir"] = args.kdtree_dir
    cfg["folder_to_load"] = os.path.join(
        "precomputed", "hypersim", cfg["scene_id"]
    )
    cfg["visualize_error"] = args.visualize_error
    cfg["visualize_error_threshold"] = args.visualize_error_threshold
    return cfg


def init_workspace():
    if not os.path.exists("tmp"):
        os.makedirs("tmp")


def main():
    cfg = parse_config()
    init_workspace()
    dataset_hypersim = Hypersim(cfg["data_dir"])

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

    # evaluation
    visualize_error_threshold = (
        None if not cfg["visualize_error"] else cfg["visualize_error_threshold"]
    )
    eval_hypersim(
        cfg,
        lines,
        dataset_hypersim,
        cfg["scene_id"],
        cam_id=cfg["cam_id"],
        vis_err_th=visualize_error_threshold,
    )

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
