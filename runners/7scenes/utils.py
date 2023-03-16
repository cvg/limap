import numpy as np
import os
import PIL
import pycolmap
from pathlib import Path
import limap.base as _base
import limap.pointsfm as _psfm
import limap.util.io as limapio

from hloc import extract_features, localize_sfm, match_features, pairs_from_covisibility, triangulation
from hloc_utils import create_reference_sfm, create_query_list_with_intrinsics, evaluate, correct_sfm_with_gt_depth

class DepthReader(_base.BaseDepthReader):
    def __init__(self, filename, depth_folder):
        super(DepthReader, self).__init__(filename)
        self.depth_folder = depth_folder

    def read(self, filename):
        depth = PIL.Image.open(Path(self.depth_folder) / filename)
        depth = np.array(depth).astype('float64')
        depth = depth / 1000.  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.inf
        return depth

def read_scene_7scenes(cfg, root_path, model_path, image_path, n_neighbors=20):
    metainfos_filename = 'infos_7scenes.npy'
    output_dir = 'tmp' if cfg['output_dir'] is None else cfg['output_dir']
    limapio.check_makedirs(output_dir)
    if cfg['skip_exists'] and os.path.exists(os.path.join(output_dir, metainfos_filename)):
        cfg['info_path'] = os.path.join(output_dir, metainfos_filename)
    if cfg['info_path'] is None:
        imagecols, neighbors, ranges = _psfm.read_infos_colmap(cfg['sfm'], root_path, model_path, image_path, n_neighbors=n_neighbors)
        with open(os.path.join(output_dir, metainfos_filename), 'wb') as f:
            np.savez(f, imagecols_np=imagecols.as_dict(), neighbors=neighbors, ranges=ranges)
    else:
        with open(cfg['info_path'], 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imagecols_np, neighbors, ranges = data['imagecols_np'].item(), data['neighbors'].item(), data['ranges']
            imagecols = _base.ImageCollection(imagecols_np)
    return imagecols, neighbors, ranges
    
def get_result_filenames(cfg, use_dense_depth=False):
    ransac_cfg = cfg['ransac']
    ransac_postfix = ''
    if ransac_cfg['method'] != None:
        if ransac_cfg['method'] in ['ransac', 'hybrid']:
            ransac_postfix = '_{}'.format(ransac_cfg['method'])
        elif ransac_cfg['method'] == 'solver':
            ransac_postfix = '_sfransac'
        else:
            raise ValueError('Unsupported ransac method: {}'.format(ransac_cfg['method']))
        ransac_postfix += '_{}'.format(ransac_cfg['thres'] if ransac_cfg['method'] != 'hybrid' else '{}-{}'.format(ransac_cfg['thres_point'], ransac_cfg['thres_line']))
    results_point = 'results_{}_point.txt'.format('dense' if use_dense_depth else 'sparse')
    results_joint = 'results_{}_joint_{}{}{}{}{}.txt'.format(
            'dense' if use_dense_depth else 'sparse',
            '{}_'.format(cfg['2d_matcher']),
            '{}_'.format(cfg['reprojection_filter']) if cfg['reprojection_filter'] is not None else '',
            'filtered_' if cfg['2d_matcher'] == 'superglue_endpoints' and cfg['epipolar_filter'] else '',
            cfg['line_cost_func'],
            ransac_postfix)
    if cfg['2d_matcher'] == 'gluestick':
        results_point = results_point.replace('point', 'point_gluestick')
        results_joint = results_joint.replace('gluestick', 'gluestickp+l')
    return results_point, results_joint

def run_hloc_7scenes(cfg, dataset, scene, results_file, test_list, num_covis=30, use_dense_depth=False, logger=None):
    results_dir = results_file.parent
    gt_dir = dataset / f'7scenes_sfm_triangulated/{scene}/triangulated'

    ref_sfm_sift = results_dir / 'sfm_sift'
    ref_sfm = results_dir / 'sfm_superpoint+superglue'
    query_list = results_dir / 'query_list_with_intrinsics.txt'
    sfm_pairs = results_dir / f'pairs-db-covis{num_covis}.txt'
    depth_dir = dataset / f'depth/7scenes_{scene}/train/depth'
    retrieval_path = dataset / '7scenes_densevlad_retrieval_top_10' / f'{scene}_top10.txt'
    feature_conf = {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {'name': 'superpoint', 'nms_radius': 3, 'max_keypoints': 4096},
        'preprocessing': {'globs': ['*.color.png'], 'grayscale': True, 'resize_max': 1024}
    }
    if cfg['localization']['2d_matcher'] == 'gluestick':
        matcher_conf = match_features.confs['gluestick']
    else:
        matcher_conf = match_features.confs['superglue']
        matcher_conf['model']['sinkhorn_iterations'] = 5

    # feature extraction
    features = extract_features.main(
            feature_conf, dataset / scene, results_dir, as_half=True)

    train_ids, query_ids = create_reference_sfm(gt_dir, ref_sfm_sift, test_list)
    create_query_list_with_intrinsics(gt_dir, query_list, test_list)
    if not sfm_pairs.exists():
        pairs_from_covisibility.main(
                ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
            matcher_conf, sfm_pairs, feature_conf['output'], results_dir)
    loc_matches = match_features.main(
            matcher_conf, retrieval_path, feature_conf['output'], results_dir)
    if not ref_sfm.exists():
        triangulation.main(
                ref_sfm, ref_sfm_sift, dataset / scene, sfm_pairs, features, sfm_matches)

    if use_dense_depth:
        assert depth_dir is not None
        ref_sfm_fix = results_dir / 'sfm_superpoint+superglue+depth'
        if not cfg['skip_exists'] or not ref_sfm_fix.exists():
            correct_sfm_with_gt_depth(ref_sfm, depth_dir, ref_sfm_fix)
        ref_sfm = ref_sfm_fix

    ref_sfm = pycolmap.Reconstruction(ref_sfm)

    if not (cfg['skip_exists'] or cfg['localization']['hloc']['skip_exists']) or not os.path.exists(results_file):
        # point only localization
        if logger: logger.info('Running Point-only localization...')
        localize_sfm.main(
            ref_sfm, query_list, retrieval_path, features, loc_matches, results_file, covisibility_clustering=False, prepend_camera_name=True)
        if logger: logger.info(f'Coarse pose saved at {results_file}')
        evaluate(gt_dir, results_file, test_list)
    else:
        if logger: logger.info(f'Point-only localization skipped.')

    # Read coarse poses
    poses = {}
    with open(results_file, 'r') as f:
        lines = []
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            poses[name] = _base.CameraPose(q, t)
    if logger: logger.info(f'Coarse pose read from {results_file}')
    hloc_log_file = f'{results_file}_logs.pkl'

    return poses, hloc_log_file, {'train': train_ids, 'query': query_ids}