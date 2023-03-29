import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm

import limap.base as _base
import limap.evaluation as _eval
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis
from runners.hypersim.Hypersim import Hypersim

import matplotlib as mpl
import matplotlib.pyplot as plt

# hypersim
MPAU = 0.02539999969303608

def plot_curve(fname, thresholds, data):
    plt.plot(thresholds, data)
    plt.savefig(fname)

def report_error_to_GT(evaluator, lines):
    lengths = np.array([line.length() for line in lines])
    sum_length = lengths.sum()
    thresholds = [0.001, 0.005, 0.01]
    list_recall, list_precision = [], []
    for threshold in thresholds:
        ratios = np.array([evaluator.ComputeInlierRatio(line, threshold) for line in lines])
        length_recall = (lengths * ratios).sum()
        list_recall.append(length_recall)
        precision = 100 * (ratios > 0).astype(int).sum() / ratios.shape[0]
        list_precision.append(precision)
    print("R: recall, P: precision")
    for idx, threshold in enumerate(thresholds):
        print("R / P at {0}mm: {1:.2f} / {2:.2f}".format(int(threshold * 1000), list_recall[idx], list_precision[idx]))
    return evaluator

def read_ply(fname):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points = np.stack([x, y, z], axis=1)
    print("number of points: {0}".format(points.shape[0]))
    return points

def write_ply(fname, points):
    from plyfile import PlyData, PlyElement
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(fname)

def report_error_to_mesh(mesh_fname, lines):
    evaluator = _eval.MeshEvaluator(mesh_fname, MPAU)
    return report_error_to_GT(evaluator, lines)

def report_error_to_point_cloud(points, lines, kdtree_dir=None):
    evaluator = _eval.PointCloudEvaluator(points)
    if kdtree_dir is None:
        evaluator.Build()
        evaluator.Save('tmp/kdtree.bin')
    else:
        evaluator.Load(kdtree_dir)
    return report_error_to_GT(evaluator, lines)

def eval_hypersim(cfg, lines, dataset_hypersim, scene_id, cam_id=0):
    # set scene id
    dataset_hypersim.set_scene_id(scene_id)
    dataset_hypersim.set_max_dim(cfg["max_image_dim"])
    dataset_hypersim.load_cameras(cam_id=cam_id)

    # generate image indexes
    index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
    index_list = dataset_hypersim.filter_index_list(index_list, cam_id=cam_id)

    if cfg["mesh_dir"] is not None:
        # eval w.r.t mesh
        evaluator = report_error_to_mesh(cfg["mesh_dir"], lines)
    elif cfg["pc_dir"] is not None:
        points = read_ply(cfg["pc_dir"])
        evaluator = report_error_to_point_cloud(points, lines, kdtree_dir=cfg["kdtree_dir"])
    else:
        # eval w.r.t point cloud
        points = dataset_hypersim.get_point_cloud_from_list(index_list, cam_id=cam_id)
        evaluator = report_error_to_point_cloud(points, lines, kdtree_dir=cfg["kdtree_dir"])
    if cfg["visualize"]:
        thresholds = np.arange(1, 11, 1) * 0.001
        for threshold in tqdm(thresholds.tolist()):
            inlier_lines = evaluator.ComputeInlierSegs(lines, threshold)
            inlier_lines_np = np.array([line.as_array() for line in inlier_lines])
            limapio.save_obj("tmp/inliers_th_{0:.4f}.obj".format(threshold), inlier_lines_np)
            outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)
            outlier_lines_np = np.array([line.as_array() for line in outlier_lines])
            limap.save_obj("tmp/outliers_th_{0:.4f}.obj".format(threshold), outlier_lines_np)

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='eval 3d lines on hypersim')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/eval/hypersim.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/eval/default.yaml', help='default config file')
    arg_parser.add_argument('-i', '--input_dir', type=str, required=True, help='input line file [*.npy, *.obj, [folder-to-linetracks]]')
    arg_parser.add_argument('-r', '--ref_dir', type=str, default=None, help='reference pseudo gt lines [*.npy, *.obj]')
    arg_parser.add_argument('--mesh_dir', type=str, default=None, help='path to the .obj file')
    arg_parser.add_argument('--pc_dir', type=str, default=None, help='path to the .ply file')
    arg_parser.add_argument('--kdtree_dir', type=str, default=None, help='path to the saved index for kd tree')
    arg_parser.add_argument('--noeval', action='store_true', help='if enabled, the evaluation is not performed')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-sid'] = '--scene_id'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_dir"] =  args.input_dir
    cfg["ref_dir"] = args.ref_dir
    cfg["mesh_dir"] = args.mesh_dir
    cfg["pc_dir"] = args.pc_dir
    cfg["kdtree_dir"] = args.kdtree_dir
    cfg["folder_to_load"] = os.path.join("precomputed", "hypersim", cfg["scene_id"])
    cfg["noeval"] = args.noeval
    # print(cfg)
    return cfg

def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

def main():
    cfg = parse_config()
    init_workspace()
    dataset_hypersim = Hypersim(cfg["data_dir"])

    # read lines
    lines, linetracks = limapio.read_lines_from_input(cfg["input_dir"])
    if linetracks is not None:
        lines = [track.line for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
        linetracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
    if cfg["noeval"]:
        return
    eval_hypersim(cfg, lines, dataset_hypersim, cfg["scene_id"], cam_id=cfg["cam_id"])

    # report track quality
    if linetracks is not None:
        sup_image_counts = np.array([track.count_images() for track in linetracks])
        sup_line_counts = np.array([track.count_lines() for track in linetracks])
        print("supporting images / lines: ({0:.2f} / {1:.2f})".format(sup_image_counts.mean(), sup_line_counts.mean()))

if __name__ == '__main__':
    main()


