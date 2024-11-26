"""
This file contains some useful functions for train / val.
"""

import os

import numpy as np
import torch
from pycolmap import logging


#################
## image utils ##
#################
def convert_image(input_tensor, axis):
    """Convert single channel images to 3-channel images."""
    image_lst = [input_tensor for _ in range(3)]
    outputs = np.concatenate(image_lst, axis)
    return outputs


######################
## checkpoint utils ##
######################
def get_latest_checkpoint(checkpoint_root, checkpoint_name, device=None):
    """Get the latest checkpoint or by filename."""
    if device is None:
        device = torch.device("cuda")
    # Load specific checkpoint
    if checkpoint_name is not None:
        checkpoint = torch.load(
            os.path.join(checkpoint_root, checkpoint_name), map_location=device
        )
    # Load the latest checkpoint
    else:
        lastest_checkpoint = sorted(
            os.listdir(os.path.join(checkpoint_root, "*.tar"))
        )[-1]
        checkpoint = torch.load(
            os.path.join(checkpoint_root, lastest_checkpoint),
            map_location=device,
        )
    return checkpoint


def remove_old_checkpoints(checkpoint_root, max_ckpt=15):
    """Remove the outdated checkpoints."""
    # Get sorted list of checkpoints
    checkpoint_list = sorted(
        [
            _
            for _ in os.listdir(os.path.join(checkpoint_root))
            if _.endswith(".tar")
        ]
    )

    # Get the checkpoints to be removed
    if len(checkpoint_list) > max_ckpt:
        remove_list = checkpoint_list[:-max_ckpt]
        for _ in remove_list:
            full_name = os.path.join(checkpoint_root, _)
            os.remove(full_name)
            logging.info("[Debug] Remove outdated checkpoint %s" % (full_name))


################
## HDF5 utils ##
################
def parse_h5_data(h5_data):
    """Parse h5 dataset."""
    output_data = {}
    for key in h5_data.keys():
        output_data[key] = np.array(h5_data[key])

    return output_data
