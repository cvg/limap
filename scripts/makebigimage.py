import os, sys
import numpy as np
import cv2
from tqdm import tqdm

def make_bigimage(imgs, shape, pad=20):
    '''
    all images should have the same size
    '''
    img = imgs[0][0]
    img_h, img_w = img.shape[0], img.shape[1]
    channel = img.shape[2]
    n_rows, n_cols = shape[0], shape[1]

    final_img_h = img_h * n_rows + pad * (n_rows - 1)
    final_img_w = img_w * n_cols + pad * (n_cols - 1)
    bigimg = np.zeros((final_img_h, final_img_w, channel)).astype(np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            row_start = (img_h + pad) * row
            row_end = row_start + img_h
            col_start = (img_w + pad) * col
            col_end = col_start + img_w
            img = imgs[row][col]
            bigimg[row_start:row_end, col_start:col_end,:] = img
    return bigimg

def main():
    folder = 'tmp/vis_refinement/track0/'
    id_list = [0, 8, 45, 63]
    n_states = 18

    for state_id in tqdm(range(n_states)):
        imgs_row1 = []
        for img_id in id_list:
            fname_heatmap = os.path.join(folder, 'state{0}_heatmap{1}.png'.format(state_id, img_id))
            imgs_row1.append(cv2.imread(fname_heatmap))
        imgs_row2 = []
        for img_id in id_list:
            fname_img = os.path.join(folder, 'state{0}_img{1}.png'.format(state_id, img_id))
            imgs_row2.append(cv2.imread(fname_img))
        imgs = []
        imgs.append(imgs_row1)
        imgs.append(imgs_row2)
        bigimg = make_bigimage(imgs, (2, 3))
        fname = 'tmp/vis_refinement/state_{0}.png'.format(state_id)
        cv2.imwrite(fname, bigimg)

if __name__ == "__main__":
    main()

