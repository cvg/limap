import collections.abc as collections
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from hloc import extract_features
from hloc.utils.io import list_h5_names
from pycolmap import logging
from tqdm import tqdm

from .superpoint import SuperPoint

string_classes = str


# Copy from legacy hloc code
def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f"input must be tensor, dict or list; found {type(input_)}"
        )


@torch.no_grad()
def run_superpoint(
    conf: Dict,
    image_dir: Path,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
    keypoints=None,
) -> Path:
    logging.info(
        "[SuperPoint] Extracting local features with configuration:"
        f"\n{pprint.pformat(conf)}"
    )

    dataset = extract_features.ImageDataset(
        image_dir, conf["preprocessing"], image_list
    )

    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path)
        if feature_path.exists() and not overwrite
        else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logging.info("[SuperPoint] Skipping the extraction.")
        return feature_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperPoint(conf["model"]).eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for img_id, data in enumerate(tqdm(loader)):
        name = dataset.names[img_id]
        if name in skip_names:
            continue

        data_tmp = map_tensor(data, lambda x: x.to(device))
        if keypoints is None or keypoints == []:
            pred = model(data_tmp)
        else:
            keypoints_tmp = (
                torch.from_numpy(keypoints[img_id]).float().to(device)
            )
            pred = model.sample_descriptors(data_tmp, [keypoints_tmp])
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    raise ValueError(
                        "[SuperPoint] Out of disk space: storing features \
                            on disk can take "
                        "significant space, did you enable the as_half flag?"
                    ) from None
                    del grp, fd[name]
                raise error

        del pred

    logging.info("[SuperPoint] Finished exporting features.")
    return feature_path
