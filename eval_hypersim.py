import os, sys
import numpy as np
from tqdm import tqdm
import core.utils as utils
import core.visualize as vis
from core.dataset import Hypersim

import limap.base as _base
import limap.evaluation as _eval

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
    thresholds = np.arange(1, 11, 1) * 0.001
    list_survived, list_fracs_05, list_fracs_02, list_fracs_00 = [], [], [], []
    for threshold in thresholds.tolist():
        ratios = np.array([evaluator.ComputeInlierRatio(line, threshold) for line in lines])
        lengths_survived = (lengths * ratios).sum()
        list_survived.append(lengths_survived)
        val = ratios >= 0.5
        fracs_05 = 100 * val.sum() / val.shape[0]
        list_fracs_05.append(fracs_05)
        val = ratios >= 0.2
        fracs_02 = 100 * val.sum() / val.shape[0]
        list_fracs_02.append(fracs_02)
        val = ratios > 0
        fracs_00 = 100 * val.sum() / val.shape[0]
        list_fracs_00.append(fracs_00)
    list_survived = np.array(list_survived)
    list_survived_ratio = 100 * list_survived / sum_length
    print("th, recall, precision, p50, p20, p0")
    for idx, threshold in enumerate(thresholds):
        print("{0}mm, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}".format(int(threshold * 1000), list_survived[idx], list_survived_ratio[idx], list_fracs_05[idx], list_fracs_02[idx], list_fracs_00[idx]))
    return evaluator

def report_error_to_ref_lines(evaluator, lines):
    lengths = np.array([line.length() for line in lines])
    sum_length_tested = lengths.sum()
    sum_length_ref = evaluator.SumLength()
    thresholds = (np.arange(1, 11, 1) * 0.001).tolist()
    list_recall_ref, list_recall_tested = [], []
    for threshold in thresholds:
        recall_ref = evaluator.ComputeRecallRef(lines, threshold)
        list_recall_ref.append(recall_ref)
        recall_tested = evaluator.ComputeRecallTested(lines, threshold)
        list_recall_tested.append(recall_tested)
    print("th, ref, ref(%), test, test(%)")
    for idx, threshold in enumerate(thresholds):
        print("{0}mm, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(int(threshold * 1000), list_recall_ref[idx], 100 * list_recall_ref[idx] / sum_length_ref, list_recall_tested[idx], 100 * list_recall_tested[idx] / sum_length_tested))
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

def eval_hypersim(cfg, lines, dataset_hypersim, scene_id, cam_id=0, ref_lines=None):
    # set scene id
    dataset_hypersim.set_scene_id(scene_id)
    dataset_hypersim.set_max_dim(cfg["max_image_dim"])
    dataset_hypersim.load_cameras(cam_id=cam_id)

    # generate image indexes
    index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
    index_list = dataset_hypersim.filter_index_list(index_list, cam_id=cam_id)

    # eval w.r.t psuedo gt lines
    if ref_lines is not None:
        evaluator = _eval.RefLineEvaluator(ref_lines)
        report_error_to_ref_lines(evaluator, lines)
        return
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
            vis.save_obj("tmp/inliers_th_{0:.4f}.obj".format(threshold), inlier_lines_np)
            outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)
            outlier_lines_np = np.array([line.as_array() for line in outlier_lines])
            vis.save_obj("tmp/outliers_th_{0:.4f}.obj".format(threshold), outlier_lines_np)

def read_txt_Line3Dpp(fname):
    linetracks = []
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    line_counts = []
    line_track_id_list = []
    line_counters = 0
    for txt_line in txt_lines:
        txt_line = txt_line.strip('\n').split(' ')
        counter = 0
        n_lines = int(txt_line[counter])
        counter += 1
        line_counters += n_lines
        # get line 3d
        line3d_list = []
        for idx in range(n_lines):
            infos = [float(k) for k in txt_line[counter:(counter+6)]]
            line3d = _base.Line3d(infos[:3], infos[3:])
            counter += 6
            line3d_list.append(line3d)
        line3d = line3d_list[0]
        n_supports = int(txt_line[counter])
        counter += 1
        # collect supports
        img_id_list, line_id_list, line2d_list = [], [], []
        for supp_id in range(n_supports):
            img_id = int(txt_line[counter])
            counter += 1
            line_id = int(txt_line[counter])
            counter += 1
            infos = [float(k) for k in txt_line[counter:(counter+4)]]
            line2d = _base.Line2d(infos[:2], infos[2:])
            counter += 4
            img_id_list.append(img_id)
            line_id_list.append(line_id)
            line2d_list.append(line2d)
        track = _base.LineTrack(line3d, img_id_list, line_id_list, line2d_list)
        linetracks.append(track)
        for idx in range(n_lines):
            line_counts.append(track.count_images())
            line_track_id_list.append(len(linetracks) - 1)

    # construct matrix
    mergemat = np.zeros((len(linetracks), line_counters))
    for idx, track_id in enumerate(line_track_id_list):
        mergemat[track_id, idx] = 1
    return linetracks, line_track_id_list, line_counts, mergemat

def read_lines_from_input(input_dir, n_visible_views=4):
    if not os.path.exists(input_dir):
        raise ValueError("Error! Input file/dir {0} not found.".format(input_dir))

    # linetracks folder
    if not os.path.isfile(input_dir):
        n_tracks = vis.count_linetracks_from_folder(input_dir)
        linetracks = [_base.LineTrack() for idx in range(n_tracks)]
        linetracks = vis.load_linetracks_from_folder(linetracks, input_dir)
        VisTrack = vis.TrackVisualizer(linetracks)
        VisTrack.report()
        lines = []
        for track in linetracks:
            if track.count_images() < n_visible_views:
                continue
            lines.append(track.line)
        return lines, linetracks, None

    # npy file
    if input_dir.endswith('.npy'):
        lines_np, counts = vis.load_npy(input_dir)
        lines = []
        for line_np, count in zip(lines_np.tolist(), counts.tolist()):
            if count < n_visible_views:
                continue
            line_np = np.array(line_np)
            lines.append(_base.Line3d(line_np[0,:], line_np[1,:]))
        return lines, None, None

    # obj file: n_visible_views is pre-determined in the obj file
    if input_dir.endswith('.obj'):
        linetracks = None
        # check if txt file exists
        txt_file = input_dir[:-4] + '.txt'
        mergemat = None
        if os.path.exists(txt_file):
            linetracks, line_track_id_list, counts, mergemat = read_txt_Line3Dpp(txt_file)
            VisTrack = vis.TrackVisualizer(linetracks)
            VisTrack.report()
        lines_np = vis.load_obj(input_dir)
        lines = []
        if mergemat is not None:
            valid_line_id_list, valid_track_id_list = [], []
        for line_id, (line_np, count) in enumerate(zip(lines_np.tolist(), counts)):
            line_np = np.array(line_np)
            if count < n_visible_views:
                continue
            lines.append(_base.Line3d(line_np[0,:], line_np[1,:]))
            if mergemat is not None:
                valid_line_id_list.append(line_id)
                valid_track_id_list.append(line_track_id_list[line_id])
        if mergemat is not None:
            valid_track_id_list = list(set(valid_track_id_list))
            mergemat = mergemat[np.ix_(valid_track_id_list, valid_line_id_list)]
        return lines, linetracks, mergemat

    # line3dpp format
    if input_dir.endswith('.txt'):
        linetracks, _ = read_txt_Line3Dpp(input_dir)
        VisTrack = vis.TrackVisualizer(linetracks)
        VisTrack.report()
        lines = []
        for track in linetracks:
            if track.count_images() < n_visible_views:
                continue
            lines.append(track.line)
        return lines, linetracks, None
    raise ValueError("Error! File {0} not supported. should be txt, obj, or folder to the linetracks.".format(input_dir))

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='eval 3d lines on hypersim')
    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/eval/hypersim_evaluation.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/eval/default_evaluation.yaml', help='default config file')
    arg_parser.add_argument('-i', '--input_dir', type=str, required=True, help='input line file [*.npy, *.obj, [folder-to-linetracks]]')
    arg_parser.add_argument('-r', '--ref_dir', type=str, default=None, help='reference pseudo gt lines [*.npy, *.obj]')
    arg_parser.add_argument('--mesh_dir', type=str, default=None, help='path to the .obj file')
    arg_parser.add_argument('--pc_dir', type=str, default=None, help='path to the .ply file')
    arg_parser.add_argument('--kdtree_dir', type=str, default=None, help='path to the saved index for kd tree')
    arg_parser.add_argument('--noeval', action='store_true', help='if enabled, the evaluation is not performed')

    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-sid'] = '--scene_id'
    cfg = utils.update_config(cfg, unknown, shortcuts)
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
    lines, linetracks, mergemat = read_lines_from_input(cfg["input_dir"], n_visible_views=cfg["n_visible_views"])
    if cfg["noeval"]:
        return
    ref_lines = None
    if cfg["ref_dir"] is not None:
        ref_lines, _, _ = read_lines_from_input(cfg["ref_dir"], n_visible_views=3)

    eval_hypersim(cfg, lines, dataset_hypersim, cfg["scene_id"], cam_id=cfg["cam_id"], ref_lines=ref_lines)

if __name__ == '__main__':
    main()


