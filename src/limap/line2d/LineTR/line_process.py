import math

import numpy as np
import torch


def filter_by_length(lines, min_length, max_sublines):
    length = lines["length_klines"]
    valid_idx = length > min_length

    klines = lines["klines"][valid_idx]
    length = lines["length_klines"][valid_idx]
    angles = lines["angles"][valid_idx]

    # re-ordering by line length
    index = np.argsort(length)
    index = index[::-1]
    klines = klines[index[:max_sublines]]
    length = length[index[:max_sublines]]

    angles = get_angles(klines)
    return {"klines": klines, "length_klines": length, "angles": angles}


def get_line_dist(line):
    sp = line[0]
    ep = line[1]
    return np.sqrt(np.sum((ep - sp) ** 2))


def get_angles(lines):
    line_exists = len(lines) > 0
    if not line_exists:
        angles = []
        return angles

    sp = lines[:, 0]
    ep = lines[:, 1]
    angles = np.arctan2((ep[:, 0] - sp[:, 0]), (ep[:, 1] - sp[:, 1]))
    for i, angle in enumerate(angles):
        if angle < 0:
            angles[i] += np.pi
    angles = np.asarray([np.cos(2 * angles), np.sin(2 * angles)]).T
    return angles


def point_on_line(line, dist_px):
    assert dist_px >= 0, "distance should be positive!"
    assert (
        get_line_dist(line) >= dist_px
    ), "distance should be smaller than line length!"

    sp, ep = line
    vec = ep - sp
    if vec[0] != 0:
        m = vec[1] / vec[0]
        x = np.sqrt(dist_px**2 / (1 + m**2))
        y = m * x
    else:
        x = 0
        y = dist_px if ep[1] - sp[1] > 0 else -dist_px

    return (x, y) + sp


def remove_borders(
    lines, border: int, height: int, width: int, valid_mask_given=None
):
    """Removes keylines too close to the border"""
    klines = lines["klines"]
    valid_mask_h0 = (klines[:, 0, 0] >= border) & (
        klines[:, 0, 0] < (width - border)
    )
    valid_mask_w0 = (klines[:, 0, 1] >= border) & (
        klines[:, 0, 1] < (height - border)
    )

    valid_mask_h1 = (klines[:, 1, 0] >= border) & (
        klines[:, 1, 0] < (width - border)
    )
    valid_mask_w1 = (klines[:, 1, 1] >= border) & (
        klines[:, 1, 1] < (height - border)
    )

    valid_mask0 = valid_mask_h0 & valid_mask_w0
    valid_mask1 = valid_mask_h1 & valid_mask_w1
    valid_mask = valid_mask0 & valid_mask1

    eps = 0.001
    klines[:, :, 0] = klines[:, :, 0].clip(max=width - eps - border)
    klines[:, :, 1] = klines[:, :, 1].clip(max=height - eps - border)

    if isinstance(valid_mask_given, np.ndarray):
        sp = np.floor(klines[:, 0]).astype(int)
        ep = np.floor(klines[:, 1]).astype(int)
        valid_mask_given = (
            valid_mask_given[sp[:, 1], sp[:, 0]]
            + valid_mask_given[ep[:, 1], ep[:, 0]]
        ).astype(bool)
        valid_mask = valid_mask & valid_mask_given

    lines = {k: v[valid_mask] for k, v in lines.items()}

    return lines


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def line_tokenizer(
    klines, token_distance, max_tokens, pred_superpoint, image_shape
):
    height, width = image_shape
    slines_all, num_slines_all = [], []
    tokens_all, masks_all = [], []
    response_all, angle_all = [], []

    for i, (kline, klength) in enumerate(
        zip(klines["klines"], klines["length_klines"])
    ):
        # line tokens for a keyline
        tokens = []
        num_tokens = int(math.ceil(klength / token_distance))
        for i_token in range(num_tokens - 1):
            dist = i_token * token_distance
            token = point_on_line(kline, dist)
            tokens.append(token)
        token = kline[1]
        token[0] = token[0].clip(max=width - 0.6)
        token[1] = token[1].clip(max=height - 0.6)
        tokens.append(token)

        # keyline-to-sublines
        sp, ep = kline
        num_sublines = int(math.ceil(num_tokens / max_tokens))
        sublines = np.zeros((num_sublines, 2, 2))
        sublines[0, 0] = sp
        sublines[-1, 1] = ep
        for i_sline in range(num_sublines - 1):
            mid_token = tokens[(i_sline + 1) * max_tokens - 1]
            sublines[i_sline, 1] = mid_token
            sublines[i_sline + 1, 0] = mid_token
        slines_all.extend(sublines)
        num_slines_all.append(num_sublines)

        # line tokens & masks for sublines
        tokens_sline = np.zeros((num_sublines, max_tokens, 2))
        masks_sline = np.zeros((num_sublines, max_tokens + 1, 1))
        masks_sline[:, 0] = 1
        for i_sline in range(num_sublines):
            i_start = i_sline * max_tokens
            i_end = i_start + max_tokens
            tmp_tokens = np.asarray(tokens[i_start:i_end])
            tokens_sline[i_sline, : len(tmp_tokens)] = tmp_tokens
            masks_sline[i_sline, 1 : len(tmp_tokens) + 1] = 1
        tokens_all.extend(tokens_sline)
        masks_all.extend(masks_sline)

        # response & angle for sublines
        resp_sline = np.zeros((num_sublines, 1))
        ang_sline = np.zeros((num_sublines, 2))
        max_length = token_distance * max_tokens
        for i_sline in range(num_sublines):
            resp_sline[i_sline] = get_line_dist(sublines[i_sline]) / max_length
            ang_sline[i_sline] = klines["angles"][i]
        response_all.extend(resp_sline)
        angle_all.extend(ang_sline)
    device = pred_superpoint["dense_descriptor"].device
    # device = 'cpu'
    slines = (
        torch.from_numpy(np.vstack(slines_all).reshape(-1, 2, 2))
        .float()
        .to(device)
    )
    tokens = (
        torch.from_numpy(np.vstack(tokens_all).reshape(-1, max_tokens, 2))
        .float()
        .to(device)
    )
    masks = (
        torch.from_numpy(np.vstack(masks_all).reshape(-1, max_tokens + 1, 1))
        .float()
        .to(device)
    )
    responses = (
        torch.from_numpy(np.vstack(response_all).reshape(-1, 1))
        .float()
        .to(device)
    )
    angles = (
        torch.from_numpy(np.vstack(angle_all).reshape(-1, 2)).float().to(device)
    )

    # adjacency matrix (mat_keylines2sublines)
    klines2slines = torch.zeros((len(klines["klines"]), len(slines))).to(device)
    st_idx = 0
    for i, num_sline in enumerate(num_slines_all):
        klines2slines[i, st_idx : st_idx + num_sline] = 1 / num_sline
        st_idx += num_sline

    # descriptors & scores for line tokens
    dense_descriptor = pred_superpoint["dense_descriptor"]
    descriptors = sample_descriptors(tokens[None], dense_descriptor, 8)[
        0
    ].reshape(256, tokens.shape[0], tokens.shape[1])
    descriptors = descriptors.permute(1, 2, 0)

    dense_score = pred_superpoint["dense_score"].transpose(1, 2)
    pos_score = [torch.round(tokens).long().reshape(-1, 2)]
    pos_score[0][:, 0] = pos_score[0][:, 0].clip(max=dense_score.shape[1] - 1)
    pos_score[0][:, 1] = pos_score[0][:, 1].clip(max=dense_score.shape[2] - 1)
    scores = [s[tuple(k.t())] for s, k in zip(dense_score, pos_score)]
    scores = scores[0].reshape(len(slines), max_tokens, 1)

    # numpy2tensor
    klines["klines"] = (
        torch.from_numpy(klines["klines"]).float().to(device)[None]
    )
    klines["length_klines"] = (
        torch.from_numpy(klines["length_klines"]).float().to(device)[None]
    )
    klines["angles"] = (
        torch.from_numpy(klines["angles"]).float().to(device)[None]
    )

    klines["sublines"] = slines[None]
    klines["pnt_sublines"] = tokens[None]
    klines["mask_sublines"] = masks[None]
    klines["resp_sublines"] = responses[None]
    klines["angle_sublines"] = angles[None]
    klines["desc_sublines"] = descriptors[None]
    klines["score_sublines"] = scores[None]
    klines["mat_klines2sublines"] = klines2slines[None]

    return klines


def get_dist_matrix(desc0, desc1):
    distance = np.einsum("bdn,bdm->bnm", desc0, desc1)  # -1 ~ 1
    distance = (2.0 - 2.0 * distance).clip(min=0)
    return distance


def change_cv2_T_np(klines_cv):
    klines_sp, klines_ep, length, angle = [], [], [], []

    for line in klines_cv:
        sp_x = line.startPointX
        sp_y = line.startPointY
        ep_x = line.endPointX
        ep_y = line.endPointY
        kline_sp = []
        if sp_x < ep_x:
            kline_sp = [sp_x, sp_y]
            kline_ep = [ep_x, ep_y]
        else:
            kline_sp = [ep_x, ep_y]
            kline_ep = [sp_x, sp_y]

        # linelength = math.sqrt((kline_ep[0]-kline_sp[0])**2 +(kline_ep[1]-kline_sp[1])**2)
        linelength = line.lineLength * (2**line.octave)

        klines_sp.append(kline_sp)
        klines_ep.append(kline_ep)
        length.append(linelength)

    klines_sp = np.asarray(klines_sp)
    klines_ep = np.asarray(klines_ep)
    klines = np.stack((klines_sp, klines_ep), axis=1)
    length = np.asarray(length)
    angles = get_angles(klines)
    return {"klines": klines, "length_klines": length, "angles": angles}


def preprocess(klines_cv, image_shape, pred_superpoint, mask=None, conf=None):
    if conf is None:
        conf = {}
    default_conf = {
        "min_length": 16,
        "max_sublines": 256,
        "token_distance": 8,
        "max_tokens": 21,
        "remove_borders": 0,
    }
    conf = {**default_conf, **conf}

    """ Pre-process for line tokenization """
    # cv2np
    klines = change_cv2_T_np(klines_cv)

    # remove_borders
    height, width = image_shape
    border = conf["remove_borders"]
    klines = remove_borders(klines, border, height, width, mask)

    # filter_by_length
    klines = filter_by_length(
        klines, conf["min_length"], conf["max_sublines"]
    )  # 15 msec
    num_klines = len(klines["klines"])
    if num_klines == 0:
        return klines

    # line_tokenizer
    klines = line_tokenizer(
        klines,
        conf["token_distance"],
        conf["max_tokens"],
        pred_superpoint,
        image_shape,
    )
    return klines
