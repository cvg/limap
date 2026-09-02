"""Location and download of pretrained model weights.

Weights default to a per-user cache directory rather than to the installed
package. ``__file__`` points inside site-packages for a non-editable install,
which is often read-only and is wiped on upgrade.

Override the root with the ``weight_path`` option of a detector / matcher, or
globally with the ``LIMAP_WEIGHTS_PATH`` environment variable.
"""

import os
from pathlib import Path

from pycolmap import logging

__all__ = ["weights_root", "resolve_weight_path", "download_weights"]


def weights_root() -> Path:
    """Default root directory for cached weights."""
    env = os.environ.get("LIMAP_WEIGHTS_PATH")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "limap"


def resolve_weight_path(weight_path, *parts) -> Path:
    """Full path of a checkpoint, under ``weight_path`` or the default root.

    Args:
        weight_path: root to use, or None for :func:`weights_root`
        parts: per-module layout, e.g. ("line2d", "DeepLSD", "deeplsd_md.tar")
    """
    root = weights_root() if weight_path is None else Path(weight_path)
    return root.joinpath(*parts)


def download_weights(url: str, path) -> Path:
    """Fetch ``url`` into ``path``, unless it is already there.

    Downloads through torch rather than shelling out to ``wget``, which is not
    a declared dependency and is missing on many systems.
    """
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    from torch.hub import download_url_to_file

    logging.info(f"Downloading model weights to {path} ...")
    download_url_to_file(url, str(path))
    return path
