import os, sys
import shutil
import copy
import random
import cv2
import math
import numpy as np
from tqdm import tqdm

def random_color():
    r = int(255 * np.random.rand())
    g = int(255 * np.random.rand())
    b = 255 - r
    return b, g, r

def draw_points(image, points, color=None, thickness=1):
    for p in points:
        c = random_color() if color is None else color
        pos_x, pos_y = int(round(p[0])), int(round(p[1]))
        cv2.circle(image, (pos_x, pos_y), thickness * 2, c, -1)
    return image

def draw_segments(image, segments, color=None, thickness=1, endpoints=True):
    for s in segments:
        c = random_color() if color is None else color
        p1 = (int(s[0]), int(s[1]))
        p2 = (int(s[2]), int(s[3]))
        cv2.line(image, p1, p2, c, thickness)
        if endpoints:
            cv2.circle(image, p1, thickness * 2, c, -1)
            cv2.circle(image, p2, thickness * 2, c, -1)
    return image

def draw_salient_segments(image, segments, saliency, color1=(0, 255, 0), color2=(0, 0, 255), thickness=1,
                          endpoints=True):
    assert len(segments) == len(saliency)
    max_saliency, min_saliency = np.max(saliency), np.min(saliency)
    for s, s_saliency in zip(segments, saliency):
        r = (s_saliency - min_saliency) / (max_saliency - min_saliency)
        c = (int(color1[0] * r + color2[0] * (1 - r)),
             int(color1[1] * r + color2[1] * (1 - r)),
             int(color1[2] * r + color2[2] * (1 - r)))
        p1 = (int(s[0]), int(s[1]))
        p2 = (int(s[2]), int(s[3]))
        cv2.line(image, p1, p2, c, thickness)
        if endpoints:
            cv2.circle(image, p1, thickness * 2, c, -1)
            cv2.circle(image, p2, thickness * 2, c, -1)
    return image

def draw_multiscale_segments(img, segs, color=None, endpoints=True, thickness=2):
    for s in segs:
        mycolor = random_color() if color is None else color
        octave, l = s[0]
        l = l * np.sqrt(2) ** octave
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), mycolor, thickness)
        if endpoints:
            cv2.circle(img, (int(l[0]), int(l[1])), int(thickness * 1.5), mycolor, -1)
            cv2.circle(img, (int(l[2]), int(l[3])), int(thickness * 1.5), mycolor, -1)
    return img

def draw_multiscale_matches(img_left, img_right, segs_left, segs_right, matches):
    assert img_left.ndim == 2
    h, w = img_left.shape

    # store the matching results of the first and second images into a single image
    left_color_img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    right_color_img = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    r1, g1, b1 = [], [], []  # the line colors

    for pair in range(len(matches)):
        r, g, b = random_color()
        r1.append(r)
        g1.append(g)
        b1.append(b)
        line_id_l, line_id_r = matches[pair]

        octave, l = segs_left[line_id_l][0]
        l = l * np.sqrt(2) ** octave
        cv2.line(left_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

        octave, l = segs_right[line_id_r][0]
        l = l * np.sqrt(2) ** octave
        cv2.line(right_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (r1[pair], g1[pair], b1[pair]), 3)

    result_img = np.hstack([left_color_img, right_color_img])
    result_img_tmp = result_img.copy()
    for pair in range(27, len(matches)):
        line_id_l, line_id_r = matches[pair]
        octave_left, seg_left = segs_left[line_id_l][0]
        octave_right, seg_right = segs_right[line_id_r][0]
        seg_left = seg_left[:4] * (np.sqrt(2) ** octave_left)
        seg_right = seg_right[:4] * (np.sqrt(2) ** octave_right)

        start_ptn = (int(seg_left[0]), int(seg_left[1]))
        end_ptn = (int(seg_right[0] + w), int(seg_right[1]))
        cv2.line(result_img_tmp, start_ptn, end_ptn, (r1[pair], g1[pair], b1[pair]), 2, cv2.LINE_AA)

    result_img = cv2.addWeighted(result_img, 0.5, result_img_tmp, 0.5, 0.0)
    return result_img

def draw_singlescale_matches(img_left, img_right, matched_segs, mask=None):
    assert img_left.ndim == 2
    h, w = img_left.shape

    # store the matching results of the first and second images into a single image
    left_color_img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    right_color_img = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    colors = []
    if mask is None:
        for l, r in matched_segs:
            color = random_color()
            colors.append(color)
            cv2.line(left_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, 3)
            cv2.line(right_color_img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 3)
    else:
        for correct, (l, r) in zip(mask, matched_segs):
            color = (0, 255, 0) if correct else (0, 0, 255)
            cv2.line(left_color_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, 3)
            cv2.line(right_color_img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 3)

    result_img = np.hstack([left_color_img, right_color_img])
    result_img_tmp = result_img.copy()
    if mask is None:
        for i in range(len(matched_segs)):
            l, r = matched_segs[i]
            start_ptn = (int(l[0]), int(l[1]))
            end_ptn = (int(r[0] + w), int(r[1]))
            cv2.line(result_img_tmp, start_ptn, end_ptn, colors[i], 2, cv2.LINE_AA)
    else:
        for correct, (l, r) in zip(mask, matched_segs):
            color = (0, 255, 0) if correct else (0, 0, 255)
            start_ptn = (int(l[0]), int(l[1]))
            end_ptn = (int(r[0] + w), int(r[1]))
            cv2.line(result_img_tmp, start_ptn, end_ptn, color, 2, cv2.LINE_AA)

    result_img = cv2.addWeighted(result_img, 0.5, result_img_tmp, 0.5, 0.0)
    return result_img

def crop_to_patch(img, center, patch_size=50):
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h < patch_size or img_w < patch_size:
        raise ValueError("Error! Image size should be larger than the given patch size!")

    # get range
    cx, cy = center[0], center[1]
    start_x = int(cx - patch_size / 2.0)
    if start_x < 0:
        start_x = 0
    if start_x + patch_size > img_w:
        start_x = img_w - patch_size
    end_x = start_x + patch_size
    start_y = int(cy - patch_size / 2.0)
    if start_y < 0:
        start_y = 0
    if start_y + patch_size > img_h:
        start_y = img_h - patch_size
    end_y = start_y + patch_size

    # crop
    if len(img.shape) == 2:
        newimg = img[start_y:end_y, start_x:end_x]
        return newimg
    elif len(img.shape) == 3:
        newimg = img[start_y:end_y, start_x:end_x,:]
        return newimg
    else:
        raise NotImplementedError

def cat_to_bigimage(imgs, shape, pad=20):
    '''
    make a big image with 2d image collections
    all images should have the same size
    '''
    img = imgs[0][0]
    img_h, img_w = img.shape[0], img.shape[1]
    channel = img.shape[2]
    n_rows, n_cols = shape[0], shape[1]

    final_img_h = img_h * n_rows + pad * (n_rows - 1)
    final_img_w = img_w * n_cols + pad * (n_cols - 1)
    bigimg = (np.ones((final_img_h, final_img_w, channel)) * 255).astype(np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            row_start = (img_h + pad) * row
            row_end = row_start + img_h
            col_start = (img_w + pad) * col
            col_end = col_start + img_w
            img = imgs[row][col]
            bigimg[row_start:row_end, col_start:col_end,:] = img
    return bigimg

def make_bigimage(imgs, pad=20):
    n_images = len(imgs)
    if n_images <= 5:
        return cat_to_bigimage([imgs], (1, n_images), pad=pad)

    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    imgs_collections = []
    imgs_now = []
    for image in imgs:
        imgs_now.append(image)
        if len(imgs_now) == n_cols:
            imgs_collections.append(imgs_now)
            imgs_now = []
    blank_image = (np.ones((imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2])) * 255).astype(np.uint8)
    while len(imgs_now) < n_cols:
        imgs_now.append(blank_image)
    imgs_collections.append(imgs_now)
    return cat_to_bigimage(imgs_collections, (n_rows, n_cols), pad=pad)

def test_point_inside_ranges(point, ranges):
    point = np.array(point)
    if ~np.all(point > ranges[0]) or ~np.all(point < ranges[1]):
        return False
    return True

def test_line_inside_ranges(line, ranges):
    if not test_point_inside_ranges(line.start, ranges):
        return False
    if not test_point_inside_ranges(line.end, ranges):
        return False
    return True

def filter_ranges(lines_np, counts_np, ranges):
    new_lines_np, new_counts_np = [], []
    for idx in range(lines_np.shape[0]):
        if test_line_inside_ranges(lines_np[idx], ranges):
            new_lines_np.append(lines_np[idx])
            new_counts_np.append(counts_np[idx])
    return np.array(new_lines_np), np.array(new_counts_np)

def report_dist_reprojection(line3d, line2d, camview, prefix=None):
    import limap.base as _base
    line2d_proj = line3d.projection(camview)
    angle = _base.compute_distance_2d(line2d, line2d_proj, _base.LineDistType.ANGULAR)
    perp_dist = _base.compute_distance_2d(line2d, line2d_proj, _base.LineDistType.PERPENDICULAR_ONEWAY)
    overlap = _base.compute_distance_2d(line2d_proj, line2d, _base.LineDistType.OVERLAP)
    sensitivity = line3d.sensitivity(camview)
    if prefix is None:
        print("angle = {0:.4f}, perp = {1:.4f}, overlap = {2:.4f}, sensi = {3:.4f}".format(angle, perp_dist, overlap, sensitivity))
    else:
        print("{4}: angle = {0:.4f}, perp = {1:.4f}, overlap = {2:.4f}, sensi = {3:.4f}".format(angle, perp_dist, overlap, sensitivity, prefix))

def visualize_2d_line(fname, imagecols, all_lines_2d, img_id, line_id):
    img = imagecols.read_image(img_id)
    img = draw_segments(img, [all_lines_2d[img_id][line_id].as_array().reshape(-1)], (0, 255, 0))
    cv2.imwrite(fname, img)

def visualize_line_track(imagecols, linetrack, prefix='linetrack', report=False):
    print("[VISUALIZE]: line length: {0}, num_supporting_lines: {1}".format(linetrack.line.length(), len(linetrack.image_id_list)))
    for idx, (img_id, line2d) in enumerate(zip(linetrack.image_id_list, linetrack.line2d_list)):
        img = imagecols.read_image(img_id)
        if len(img.shape) == 2:
            img = img[:,:,None].repeat(3, 2)
        report_dist_reprojection(linetrack.line, line2d, imagecols.camview(img_id), prefix="Reprojecting to line {0} (img {1}, line {2})".format(idx, img_id, linetrack.line_id_list[idx]))
        line2d_proj = linetrack.line.projection(imagecols.camview(img_id))
        img = draw_segments(img, [line2d_proj.as_array().reshape(-1)], (255, 0, 0), thickness=1)
        img = draw_segments(img, [line2d.as_array().reshape(-1)], (0, 0, 255), thickness=1)
        fname = os.path.join('tmp', '{0}.{1}.{2}.png'.format(prefix, idx, os.path.basename(imagecols.camimage(img_id).image_name())[:-4]))
        cv2.imwrite(fname, img)

def vis_vpresult(img, lines, vpres, show_original=False):
    import seaborn as sns
    import cv2
    n_vps = vpres.count_vps()
    colors = sns.color_palette("husl", n_colors=n_vps)
    colors = (np.array(colors) * 255).astype(np.uint8).tolist()
    if n_vps == 1:
        colors = [[255, 0, 0]]
    for line_id, line in enumerate(lines):
        if not vpres.HasVP(line_id):
            if not show_original:
                continue
            c = [255, 255, 255]
        else:
            c = colors[vpres.labels[line_id]]
        cv2.line(img, (int(line.start[0]), int(line.start[1])), (int(line.end[0]), int(line.end[1])), c, 2)
    return img

def features_to_RGB(*Fs, skip=1):
    """Copied from pixloc repo. [LINK] https://github.com/cvg/pixloc/blob/master/pixloc/visualization/viz_2d.py"""
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    flatten = []
    shapes = []
    for F in Fs:
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F = F.reshape(-1, c)
        flatten.append(F)
        shapes.append((h, w))
    flatten = np.concatenate(flatten, axis=0)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(normalize(flatten[::skip]))
        flatten = normalize(pca.transform(normalize(flatten)))
    else:
        flatten = normalize(pca.fit_transform(normalize(flatten)))
    flatten = (flatten + 1) / 2

    Fs = []
    for h, w in shapes:
        F, flatten = np.split(flatten, [h*w], axis=0)
        F = F.reshape((h, w, 3))
        Fs.append(F)
    assert flatten.shape[0] == 0
    return Fs

