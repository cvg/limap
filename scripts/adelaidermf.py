import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import core.visualize as vis
import core.utils as utils
import cv2

import limap.base as _base
import limap.sfm as _sfm
import limap.triangulation as _tri
import limap.vpdetection as _vpdet
import pdb

data_path = os.path.expanduser('~/data/AdelaideRMF/adelaidermf_10_hartley')
colmap_path = os.path.join(data_path, "dense")
lineinfo_path = os.path.join(data_path, "line_matches.txt")
CONST_IOU_THRESHOLD = 0.1

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
            lines.append(_base.Line2d(np.array([x1, y1]), np.array([x2, y2])))
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
    imname_list, cameras, _ = _sfm.ReadInfos(colmap_path, "sparse", "images", max_image_dim=1200)
    model = _sfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, "sparse", "images")
    ranges = model.ComputeRanges([0.05, 0.95], 3.0)
    namelist = [os.path.basename(k) for k in imname_list]
    cam1 = cameras[namelist.index('img1.jpg')]
    cam2 = cameras[namelist.index('img2.jpg')]
    new_imname_list = []
    new_imname_list.append(imname_list[namelist.index('img1.jpg')])
    new_imname_list.append(imname_list[namelist.index('img2.jpg')])
    imname_list = new_imname_list
    all_lines, matches = read_lineinfo(lineinfo_path)

    # debug
    # debug(imname_list, all_lines)

    # vpdetection
    vpcfg = _vpdet.VPDetectorConfig()
    vpcfg.inlier_threshold = 1.5
    vpcfg.min_num_supports = 5
    detector = _vpdet.VPDetector(vpcfg)
    vpresults = []
    vpresults.append(detector.AssociateVPs(all_lines[0]))
    vpresults.append(detector.AssociateVPs(all_lines[1]))

    # visualize vpdetection
    for img_id, (imname, lines, vpres) in enumerate(zip(imname_list, all_lines, vpresults)):
        img = utils.read_image(imname, max_image_dim=1200, set_gray=False)
        img = vis.vis_vpresult(img, lines, vpres)
        cv2.imwrite("tmp/vis_{0}.png".format(img_id), img)

    # Method 0: triangulation endpoints
    tri_lines_0 = []
    for (line1, line2) in zip(all_lines[0], all_lines[1]):
        IoU = _tri.compute_epipolar_IoU(line1, cam1, line2, cam2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        line = _tri.triangulate_endpoints(line1, cam1, line2, cam2)
        tri_lines_0.append(line)

    # Method 1: triangulation
    tri_lines_1 = []
    for (line1, line2) in zip(all_lines[0], all_lines[1]):
        IoU = _tri.compute_epipolar_IoU(line1, cam1, line2, cam2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        line = _tri.triangulate(line1, cam1, line2, cam2)
        tri_lines_1.append(line)

    # Method 2: triangulation with direction
    tri_lines_2 = []
    for line_id, (line1, line2) in enumerate(zip(all_lines[0], all_lines[1])):
        IoU = _tri.compute_epipolar_IoU(line1, cam1, line2, cam2)
        if IoU < CONST_IOU_THRESHOLD:
            continue
        direc_list = []
        if vpresults[0].HasVP(line_id):
            direc = _tri.get_direction_from_VP(vpresults[0].GetVP(line_id), cam1)
            if V2_FLAG and vpresults[0].labels[line_id] == 1:
                direc = v2
            direc_list.append(np.array(direc))
        if vpresults[1].HasVP(line_id):
            direc = _tri.get_direction_from_VP(vpresults[1].GetVP(line_id), cam2)
            if V2_FLAG and vpresults[1].labels[line_id] == 1:
                direc = v2
            direc_list.append(np.array(direc))
        if len(direc_list) == 0:
            line = _tri.triangulate(line1, cam1, line2, cam2)
            tri_lines_2.append(line)
        else:
            direc = np.array(direc_list).mean(0)
            direc /= np.linalg.norm(direc)
            line = _tri.triangulate_with_direction(line1, cam1, line2, cam2, direc)
            tri_lines_2.append(line)

    # visualize triangulation
    tri_lines_0_np = np.array([line.as_array() for line in tri_lines_0])
    vis.save_obj("tmp/lines_test_0.obj", tri_lines_0_np)
    tri_lines_1_np = np.array([line.as_array() for line in tri_lines_1])
    vis.save_obj("tmp/lines_test_1.obj", tri_lines_1_np)
    tri_lines_2_np = np.array([line.as_array() for line in tri_lines_2])
    vis.save_obj("tmp/lines_test_2.obj", tri_lines_2_np)
    pdb.set_trace()
    tri_lines_np = tri_lines_2_np
    vis.vis_3d_lines(tri_lines_np, ranges=ranges)

if __name__ == '__main__':
    main()

