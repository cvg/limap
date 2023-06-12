import os
import numpy as np
from tqdm import tqdm
import limap.visualize as limapvis

def visualize_heatmap_intersections(prefix, imname_list, image_ids, p_heatmaps, ht_intersections, max_image_dim=None):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap="viridis")

    path = os.path.dirname(prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    for img_id, heatmap, intersections in zip(image_ids, p_heatmaps, ht_intersections):
        imname = imname_list[img_id]

        # visualize image
        img = utils.read_image(imname, max_image_dim=max_image_dim, set_gray=False)
        img = limapvis.draw_points(img, intersections, (255, 0, 0), 2)
        fname_out = prefix + '_img{0}.png'.format(img_id)
        cv2.imwrite(fname_out, img)

        # visualize heatmap
        heatmap_img = (scalarMap.to_rgba(heatmap)[:,:,:3] * 255).astype(np.uint8)
        heatmap_img = limapvis.draw_points(heatmap_img, intersections, (255, 0, 0), 2)
        fname_out_heatmap = prefix + '_heatmap{0}.png'.format(img_id)
        cv2.imwrite(fname_out_heatmap, heatmap_img)

def visualize_fconsis_intersections(prefix, imname_list, image_ids, fc_intersections, max_image_dim=None, n_samples_vis=-1):
    if n_samples_vis != -1:
        fc_intersections = fc_intersections[:n_samples_vis]
    path = os.path.dirname(prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    for sample_id, intersections in enumerate(tqdm(fc_intersections)):
        imgs = []
        for data in intersections:
            img_id, point = image_ids[data[0]], data[1]
            img = utils.read_image(imname_list[img_id], max_image_dim=max_image_dim, set_gray=False)
            limapvis.draw_points(img, [point], (0, 0, 255), 1)
            img = limapvis.crop_to_patch(img, point, patch_size=100)
            imgs.append(img)
        bigimg = limapvis.make_bigimage(imgs, pad=20)
        fname_out = prefix + '_sample{0}.png'.format(sample_id)
        cv2.imwrite(fname_out, bigimg)

def unit_test_add_noise_to_track(track):
    # for unit test
    tmptrack = _base.LineTrack(track)
    start = track.line.start + (np.random.rand(3) - 0.5) * 1e-1
    end = track.line.end + (np.random.rand(3) - 0.5) * 1e-1
    tmpline = _base.Line3d(start, end)
    tmptrack.line = tmpline
    return tmptrack


