import os
import numpy as np
import cv2
import torch
import limap.util.io as limapio
from ..base_detector import BaseDetector, BaseDetectorOptions

class L2D2Extractor(BaseDetector):
    def __init__(self, options = BaseDetectorOptions(), device=None):
        super(L2D2Extractor, self).__init__(options)
        self.mini_batch = 20
        self.device = 'cuda' if device is None else device
        if self.weight_path is None:
            ckpt = os.path.join(os.path.dirname(__file__),
                            'checkpoint_line_descriptor.th')
        else:
            ckpt = os.path.join(self.weight_path, "line2d", "L2D2", 'checkpoint_line_descriptor.th')
        if not os.path.isfile(ckpt):
            self.download_model(ckpt)
        import sys
        sys.path.append(os.path.dirname(__file__))
        self.model = torch.load(ckpt).to(self.device)
        self.model.eval()

    def download_model(self, path):
        import subprocess
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://github.com/hichem-abdellali/L2D2/blob/main/IN_OUT_DATA/INPUT_NETWEIGHT/checkpoint_line_descriptor.th?raw=true"
        cmd = ["wget", link, "-O", path]
        print("Downloading L2D2 model...")
        subprocess.run(cmd, check=True)

    def get_module_name(self):
        return "l2d2"

    def get_descinfo_fname(self, descinfo_folder, img_id):
        fname = os.path.join(descinfo_folder, "descinfo_{0}.npz".format(img_id))
        return fname

    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        limapio.check_makedirs(descinfo_folder)
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        limapio.save_npz(fname, descinfo)

    def read_descinfo(self, descinfo_folder, img_id):
        fname = self.get_descinfo_fname(descinfo_folder, img_id)
        descinfo = limapio.read_npz(fname)
        return descinfo

    def extract(self, camview, segs):
        img = camview.read_image(set_gray=self.set_gray)
        descinfo = self.compute_descinfo(img, segs)
        return descinfo

    def get_patch(self, img, line):
        """ Extract a 48x32 patch around a line [2, 2]. """
        h, w = img.shape

        # Keep a consistent endpoint ordering
        if line[1, 1] < line[0, 1]:
            line = line[[1, 0]]

        # Get the rotation angle
        angle = np.arctan2(line[1, 0] - line[0, 0], line[1, 1] - line[0, 1])

        # Compute the affine transform to center and rotate the line
        midpoint = line.mean(axis=0)
        T_midpoint_to_origin = np.array([[1., 0., -midpoint[0]],
                                       [0., 1., -midpoint[1]],
                                       [0., 0., 1.]])
        T_rot = np.array([[np.cos(angle), -np.sin(angle), 0.],
                          [np.sin(angle), np.cos(angle), 0.],
                          [0., 0., 1.]])
        T_origin_to_center = np.array([[1., 0., w // 2],
                                       [0., 1., h // 2],
                                       [0., 0., 1.]])
        A = T_origin_to_center @ T_rot @ T_midpoint_to_origin

        # Translate and rotate the image
        patch = cv2.warpAffine(img, A[:2], (w, h))

        # Crop and resize the patch
        length = np.linalg.norm(line[0] - line[1])
        new_h = max(int(np.round(length)), 5)  # use a minimum height of 5 for short segments
        new_w = new_h * 32 // 48
        patch = patch[h // 2 - new_h // 2: h // 2 + new_h // 2,
                      w // 2 - new_w // 2: w // 2 + new_w // 2]
        patch = cv2.resize(patch, (32, 48))
        return patch

    def compute_descinfo(self, img, segs):
        """ A desc_info is composed of the following tuple / np arrays:
            - the line descriptors [N, 128]
        """
        # Extract patches and compute a line descriptor for each patch
        lines = segs.reshape(-1, 2, 2)
        if len(lines) == 0:
            return {'line_descriptors': np.empty((0, 128))}

        patches, line_desc = [], []
        for i, l in enumerate(lines):
            patches.append(self.get_patch(img, l))

            if ((i + 1) % self.mini_batch == 0
                or i == len(lines) - 1):
                # Extract the descriptors
                patches = torch.tensor(np.array(patches), dtype=torch.float,
                    device=self.device)[:, None] / 255.
                patches = (patches - 0.492967568115862) / 0.272086182765434
                with torch.no_grad():
                    line_desc.append(self.model(patches))
                patches = []
        line_desc = torch.cat(line_desc, dim=0)  # [n_lines, 128]
        return {'line_descriptors': line_desc.cpu().numpy()}

