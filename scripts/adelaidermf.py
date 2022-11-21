import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

import limap.base as _base
import limap.pointsfm as _sfm
import limap.triangulation as _tri
import limap.vplib as _vplib
import limap.util.io as limapio
import limap.visualize as limapvis

data_path = os.path.expanduser('~/data/AdelaideRMF/adelaidermf_10_hartley')
colmap_path = os.path.join(data_path)
lineinfo_path = os.path.join(data_path, "line_matches.txt")
CONST_IOU_THRESHOLD = -1.0

v2 = np.array([0.56076211, 0.12800691, 0.81802206])
V2_FLAG = True

def read_lineinfo(fname):
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    txt_lines = [k.strip('\n') for k in txt_lines]
    counter = 0

    # load all lines
    all_lines = []
    n_images = int(txt_lines[counter])
    counter += 1
    for img_id in range(n_images):
        k = txt_lines[counter].split(' ')
        assert img_id == int(k[0])
        n_lines = int(k[1])
        counter += 1
        lines = []
        for line_id in range(n_lines):
            coors = txt_lines[counter].split(' ')
            coors = [int(k) for k in coors]
            x1, y1, x2, y2 = coors
            line = _base.Line2d(np.array([x1, y1]), np.array([x2, y2]))
            lines.append(line)
            counter += 1
        all_lines.append(lines)

    # load matches
    matches = []
    k = txt_lines[counter].split(' ')
    assert 'M' == k[0]
    n_matches = int(k[1])
    counter += 1
    for match_id in range(n_matches):
        indexes = txt_lines[counter].split(' ')
        indexes = [int(k) for k in indexes]
        matches.append(indexes)
    return all_lines, matches

def debug(imname_list, all_lines):
    imname = imname_list[0]
    lines = all_lines[0]
    import cv2
    img = utils.read_image(imname, max_image_dim=1200, set_gray=False)
    import matplotlib.pyplot as plt
    n_lines = len(lines)
    plt.imshow(img)
    for line_id in range(n_lines):
        line = lines[line_id]
        x1, y1 = line.start[0], line.start[1]
        x2, y2 = line.end[0], line.end[1]
        plt.plot(x1, y1, x2, y2, marker='o')
        plt.draw()
        plt.pause(0.5)
    plt.close()

def main():
    cfg = {}
    cfg["min_triangulation_angle"] = 1.0
    cfg["neighbor_type"] = "dice"
    cfg["ranges"] = {}
    cfg["ranges"]["range_robust"] = [0.05, 0.95]
    cfg["ranges"]["k_stretch"] = 3.0
    imagecols, neighbors, ranges = _sfm.read_infos_colmap(cfg, colmap_path, "sparse/0", "images")
    imagecols.set_max_image_dim(1200)
    imname_list = imagecols.get_image_name_list()
    view1 = imagecols.camview(1)
    view2 = imagecols.camview(2)
    all_lines, matches = read_lineinfo(lineinfo_path)

    # vpdetection
    vpcfg = _vplib.JLinkageConfig()
    vpcfg.inlier_threshold = 1.5
    vpcfg.min_num_supports = 5
    detector = _vplib.JLinkage(vpcfg)
    vpresults = []
    vpresults.append(detector.AssociateVPs(all_lines[0]))
    vpresults.append(detector.AssociateVPs(all_lines[1]))

    # visualize vpdetection
    for img_id, (imname, lines, vpres) in enumerate(zip(imname_list, all_lines, vpresults)):
        img = imagecols.read_image(img_id + 1, set_gray=False)
        img = limapvis.vis_vpresult(img, lines, vpres)
        cv2.imwrite("tmp/vis_{0}.png".format(img_id + 1), img)

    # Method 0: triangulation endpoints
    tri_lines_0 = []
    for (line1, line2) in zip(all_lines[0], all_lines[1]):
        IoU = _tri.compute_epipolar_IoU(line1, view1, line2, view2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        line = _tri.triangulate_endpoints(line1, view1, line2, view2)
        tri_lines_0.append(line)

    # Method 1: triangulation
    tri_lines_1, tri_lines_1_inv = [], []
    for (line1, line2) in zip(all_lines[0], all_lines[1]):
        IoU = _tri.compute_epipolar_IoU(line1, view1, line2, view2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        line = _tri.triangulate(line1, view1, line2, view2)
        tri_lines_1.append(line)
        line_inv = _tri.triangulate(line2, view2, line1, view1)
        tri_lines_1_inv.append(line_inv)

    # Method 2: triangulation with direction
    tri_lines_2 = []
    for line_id, (line1, line2) in enumerate(zip(all_lines[0], all_lines[1])):
        IoU = _tri.compute_epipolar_IoU(line1, view1, line2, view2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        direc_list = []
        if vpresults[0].HasVP(line_id):
            direc = _tri.get_direction_from_VP(vpresults[0].GetVP(line_id), view1)
            if V2_FLAG and vpresults[0].labels[line_id] == 1:
                direc = v2
            direc_list.append(np.array(direc))
        if vpresults[1].HasVP(line_id):
            direc = _tri.get_direction_from_VP(vpresults[1].GetVP(line_id), view2)
            if V2_FLAG and vpresults[1].labels[line_id] == 1:
                direc = v2
            direc_list.append(np.array(direc))
        if len(direc_list) == 0:
            line = _tri.triangulate(line1, view1, line2, view2)
            tri_lines_2.append(line)
        else:
            direc = np.array(direc_list).mean(0)
            direc /= np.linalg.norm(direc)
            line = _tri.triangulate_with_direction(line1, view1, line2, view2, direc)
            tri_lines_2.append(line)

    # visualize triangulation
    limapio.save_obj("tmp/lines_test_0.obj", tri_lines_0)
    limapio.save_obj("tmp/lines_test_1.obj", tri_lines_1)
    limapio.save_obj("tmp/lines_test_2.obj", tri_lines_2)

    import pdb
    pdb.set_trace()
    limapvis.open3d_vis_3d_lines(tri_lines_1)

if __name__ == '__main__':
    main()

