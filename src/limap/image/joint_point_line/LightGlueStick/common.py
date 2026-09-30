"""Pieces shared by the line-only and the joint LightGlueStick matchers."""

import numpy as np
import torch

from limap.util.model_weights import download_weights, resolve_weight_path

_WEIGHTS_URL = (
    "https://github.com/aubingazhib/LightGlueStick/releases/download/v1.0.0/"
    "lightgluestick.tar"
)


def load_lightgluestick(weight_path, device, options=None):
    """Build the LightGlueStick matcher network with its released weights."""
    from lightgluestick.lightgluestick import LightGlueStick

    if options is None:
        # Imported here so that the registry, which owns the dataclass, stays
        # importable without torch.
        from ...line.register_matcher import LightGlueStickOptions

        options = LightGlueStickOptions()
    ckpt = resolve_weight_path(
        weight_path, "line2d", "LightGlueStick", "lightgluestick.tar"
    )
    download_weights(_WEIGHTS_URL, ckpt)
    # Upstream loads the checkpoint itself, from the path in its config.
    net = LightGlueStick(
        {
            "weights": str(ckpt),
            "depth_confidence": options.depth_confidence,
        }
    )
    net = net.eval().to(device)
    # eye_mask is a plain attribute rather than a buffer, so moving the module
    # leaves it on whichever device was visible when the class was imported.
    net.eye_mask = net.eye_mask.to(device)
    return net


def build_inputs(descinfo1, descinfo2, device):
    """Assemble the network input dict from two wireframe descriptions.

    Those junction descriptors are sampled the way GlueStick does it, where
    upstream uses the corrected grid sampling -- a ~0.97 cosine difference on
    real features, taken so that both matchers read the same description.
    """

    inputs = {}
    for suffix, descinfo in (("0", descinfo1), ("1", descinfo2)):

        def tensor(key, dtype=torch.float, descinfo=descinfo):
            return torch.tensor(descinfo[key][None], dtype=dtype, device=device)

        height, width = descinfo["image_shape"][:2]
        inputs[f"view{suffix}"] = {
            "image_size": torch.tensor(
                [[width, height]], dtype=torch.float, device=device
            )
        }
        inputs[f"keypoints{suffix}"] = tensor("junctions")
        inputs[f"keypoint_scores{suffix}"] = tensor("junc_scores")
        # LightGlue's layout, one descriptor per row where GlueStick has one
        # per column.
        inputs[f"descriptors{suffix}"] = tensor("junc_desc").transpose(-1, -2)
        inputs[f"lines{suffix}"] = tensor("lines")
        inputs[f"line_scores{suffix}"] = tensor("line_scores")
        inputs[f"lines_junc_idx{suffix}"] = tensor("lines_junc_idx", torch.long)
    return inputs


def _fit_eye_mask(net, n_junctions):
    """Grow the identity LightGlueStick builds its line masks from.

    It allocates one at construction, sized from ``max_num_lines``, and then
    indexes it by junction -- so a wireframe with more junctions than that
    would run off the end.
    """
    if net.eye_mask.shape[-1] < n_junctions:
        net.eye_mask = torch.eye(
            n_junctions, dtype=torch.float32, device=net.eye_mask.device
        )[None]


def run_lightgluestick(net, descinfo1, descinfo2, device):
    """Match two wireframe descriptions, in the layout the matchers expect."""
    n_lines1, n_lines2 = len(descinfo1["lines"]), len(descinfo2["lines"])
    if n_lines1 == 0 or n_lines2 == 0:
        # LightGlueStick reads the junction count off lines_junc_idx.max(),
        # which an image with no lines does not have.
        n_junctions1 = len(descinfo1["junctions"])
        return (
            np.full(n_junctions1, -1),
            np.zeros(n_junctions1),
            np.full(n_lines1, -1),
            None,
        )

    inputs = build_inputs(descinfo1, descinfo2, device)
    _fit_eye_mask(
        net,
        1
        + max(
            int(inputs["lines_junc_idx0"].max()),
            int(inputs["lines_junc_idx1"].max()),
        ),
    )
    with torch.no_grad():
        out = net(inputs)
    return (
        out["matches0"].cpu().numpy()[0],
        out["matching_scores0"].cpu().numpy()[0],
        out["line_matches0"].cpu().numpy()[0],
        # Absent when an image has no keypoints at all, which is also the case
        # where there is nothing to rank.
        out["raw_line_scores"][0] if "raw_line_scores" in out else None,
    )
