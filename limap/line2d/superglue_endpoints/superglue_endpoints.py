import os, sys
import numpy as np
import torch

from ...point2d.superpoint.superpoint import SuperPoint
from ...point2d.superglue.superglue import SuperGlue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_detector import BaseDetector
from base_matcher import BaseMatcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.util.io as limapio


class EndpointsExtractor(BaseDetector):
    def __init__(self, set_gray=True, max_num_2d_segs=3000, device=None):
        super(EndpointsExtractor, self).__init__(
            set_gray=set_gray, max_num_2d_segs=max_num_2d_segs)
        self.device = "cuda" if device is None else device
        self.sp = SuperPoint({}).eval().to(self.device)

    def get_module_name(self):
        return "superglue_endpoints"

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
    
    def compute_descinfo(self, img, segs):
        """ A desc_info is composed of the following tuple / np arrays:
            - the original image shape (h, w)
            - the 2D endpoints of the lines in shape [N*2, 2] (xy convention)
            - the line score of shape [N]
            - the descriptor of each endpoints of shape [256, N*2]
        """
        lines = segs[:, :4].reshape(-1, 2)
        scores = segs[:, -1]
        torch_img = {'image': torch.tensor(img.astype(np.float32) / 255,
                                           dtype=torch.float,
                                           device=self.device)[None, None]}
        torch_endpoints = torch.tensor(lines.reshape(1, -1, 2),
                                       dtype=torch.float, device=self.device)
        with torch.no_grad():
            endpoint_descs = self.sp.sample_descriptors(
                torch_img, torch_endpoints)['descriptors'][0].cpu().numpy()
        return {'image_shape': img.shape, 'lines': lines,
                'lines_score': scores, 'endpoints_desc': endpoint_descs}


class SuperGlueEndpointsMatcher(BaseMatcher):
    def __init__(self, extractor, weights='outdoor', n_neighbors=20,
                 topk=10, n_jobs=1, device=None):
        super(SuperGlueEndpointsMatcher, self).__init__(
            extractor, n_neighbors=n_neighbors, topk=topk, n_jobs=n_jobs)
        self.device = "cuda" if device is None else device
        self.sg = SuperGlue({'weights': weights}).eval().to(self.device)

    def get_module_name(self):
        return "superglue_endpoints"

    def match_pair(self, descinfo1, descinfo2):
        if self.topk == 0:
            return self.match_segs_with_descinfo(descinfo1, descinfo2)
        else:
            return self.match_segs_with_descinfo_topk(descinfo1, descinfo2,
                                                      topk=self.topk)

    def match_segs_with_descinfo(self, descinfo1, descinfo2):
        # Setup the inputs for SuperGlue
        inputs = {
            'image_shape0': descinfo1['image_shape'],
            'image_shape1': descinfo2['image_shape'],
            'keypoints0': torch.tensor(
                descinfo1['lines'][None], dtype=torch.float,
                device=self.device),
            'keypoints1': torch.tensor(
                descinfo2['lines'][None], dtype=torch.float,
                device=self.device),
            'scores0': torch.tensor(
                descinfo1['lines_score'].repeat(2)[None], dtype=torch.float,
                device=self.device),
            'scores1': torch.tensor(
                descinfo2['lines_score'].repeat(2)[None], dtype=torch.float,
                device=self.device),
            'descriptors0': torch.tensor(
                descinfo1['endpoints_desc'][None], dtype=torch.float,
                device=self.device),
            'descriptors1': torch.tensor(
                descinfo2['endpoints_desc'][None], dtype=torch.float,
                device=self.device),
        }
        
        with torch.no_grad():
            # Run the point matching
            out = self.sg(inputs)
            
            # Retrieve the best matching score of the line endpoints
            n_lines1 = len(descinfo1['lines']) // 2
            n_lines2 = len(descinfo2['lines']) // 2
            scores = out['scores'].reshape(nlines1, 2, nlines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0])
            
            # Run the Sinkhorn algorithm and get the line matches
            scores = self.sg._solve_optimal_transport(scores)
            matches = self.sg._get_matches(scores)[0].cpu().numpy()[0]
            
        # Transform matches to [n_matches, 2]
        id_list_1 = np.arange(0, matches.shape[0])[matches != -1]
        id_list_2 = matches[matches != -1]
        matches_t = np.stack([id_list_1, id_list_2], 1)
        return matches_t

    def match_segs_with_descinfo_topk(self, descinfo1, descinfo2, topk=10):
        # Setup the inputs for SuperGlue
        inputs = {
            'image_shape0': descinfo1['image_shape'],
            'image_shape1': descinfo2['image_shape'],
            'keypoints0': torch.tensor(
                descinfo1['lines'][None], dtype=torch.float,
                device=self.device),
            'keypoints1': torch.tensor(
                descinfo2['lines'][None], dtype=torch.float,
                device=self.device),
            'scores0': torch.tensor(
                descinfo1['lines_score'].repeat(2)[None], dtype=torch.float,
                device=self.device),
            'scores1': torch.tensor(
                descinfo2['lines_score'].repeat(2)[None], dtype=torch.float,
                device=self.device),
            'descriptors0': torch.tensor(
                descinfo1['endpoints_desc'][None], dtype=torch.float,
                device=self.device),
            'descriptors1': torch.tensor(
                descinfo2['endpoints_desc'][None], dtype=torch.float,
                device=self.device),
        }
        
        with torch.no_grad():
            # Run the point matching
            out = self.sg(inputs)
            
            # Retrieve the best matching score of the line endpoints
            n_lines1 = len(descinfo1['lines']) // 2
            n_lines2 = len(descinfo2['lines']) // 2
            scores = out['scores'].reshape(n_lines1, 2, n_lines2, 2)
            scores = 0.5 * torch.maximum(
                scores[:, 0, :, 0] + scores[:, 1, :, 1],
                scores[:, 0, :, 1] + scores[:, 1, :, 0])
            
            # For each line in img1, retrieve the topk matches in img2
            matches = torch.argsort(scores, dim=1)[:, -topk:].cpu().numpy()
            
        # Transform matches to [n_matches, 2]
        n_lines = matches.shape[0]
        topk = matches.shape[1]
        matches_t = np.stack([np.arange(n_lines).repeat(topk),
                              matches.flatten()], axis=1)
        return matches_t
