"""Pieces shared by the line-only and the joint GlueStick matchers."""

import torch

from limap.util.model_weights import download_weights, resolve_weight_path

_WEIGHTS_URL = (
    "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/"
    "checkpoint_GlueStick_MD.tar"
)


def load_gluestick(weight_path, device):
    """Build the GlueStick matcher network with its released weights."""
    from gluestick.models.gluestick import GlueStick

    net = GlueStick({}).eval().to(device)
    ckpt = resolve_weight_path(
        weight_path,
        "line2d",
        "GlueStick",
        "weights/checkpoint_GlueStick_MD.tar",
    )
    download_weights(_WEIGHTS_URL, ckpt)
    state = torch.load(ckpt, map_location="cpu")["model"]
    state = {k[8:]: v for (k, v) in state.items() if k.startswith("matcher")}
    net.load_state_dict(state, strict=True)
    return net


def build_inputs(descinfo1, descinfo2, device):
    """Assemble the network input dict from two wireframe descriptions."""

    inputs = {
        "image_size0": tuple(descinfo1["image_shape"]),
        "image_size1": tuple(descinfo2["image_shape"]),
    }
    for suffix, descinfo in (("0", descinfo1), ("1", descinfo2)):

        def tensor(key, dtype=torch.float, descinfo=descinfo):
            return torch.tensor(descinfo[key][None], dtype=dtype, device=device)

        inputs[f"keypoints{suffix}"] = tensor("junctions")
        inputs[f"keypoint_scores{suffix}"] = tensor("junc_scores")
        inputs[f"descriptors{suffix}"] = tensor("junc_desc")
        inputs[f"lines{suffix}"] = tensor("lines")
        inputs[f"line_scores{suffix}"] = tensor("line_scores")
        inputs[f"lines_junc_idx{suffix}"] = tensor("lines_junc_idx", torch.long)
    return inputs


def run_gluestick(net, descinfo1, descinfo2, device):
    """Match two wireframe descriptions, in the layout the matchers expect."""
    inputs = build_inputs(descinfo1, descinfo2, device)
    with torch.no_grad():
        out = net(inputs)
    return (
        out["matches0"].cpu().numpy()[0],
        out["match_scores0"].cpu().numpy()[0],
        out["line_matches0"].cpu().numpy()[0],
        # Absent when an image has no keypoints at all, which is also the case
        # where there is nothing to rank.
        out["raw_line_scores"][0] if "raw_line_scores" in out else None,
    )
