"""
Main file to launch training and testing experiments.
"""

import yaml
import os
import argparse
import numpy as np
import torch

# Pytorch configurations
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

def load_config(config_path):
    """ Load configurations from a given yaml file. """
    # Check file exists
    if not os.path.exists(config_path):
        raise ValueError("[Error] The provided config path is not valid.")

    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config(path, model_cfg=None, dataset_cfg=None):
    """ Update configuration file from the resume path. """
    # Check we need to update or completely override.
    model_cfg = {} if model_cfg is None else model_cfg
    dataset_cfg = {} if dataset_cfg is None else dataset_cfg

    # Load saved configs
    with open(os.path.join(path, "model_cfg.yaml"), "r") as f:
        model_cfg_saved = yaml.safe_load(f)
        model_cfg.update(model_cfg_saved)
    with open(os.path.join(path, "dataset_cfg.yaml"), "r") as f:
        dataset_cfg_saved = yaml.safe_load(f)
        dataset_cfg.update(dataset_cfg_saved)

    # Update the saved yaml file
    if not model_cfg == model_cfg_saved:
        with open(os.path.join(path, "model_cfg.yaml"), "w") as f:
            yaml.dump(model_cfg, f)
    if not dataset_cfg == dataset_cfg_saved:
        with open(os.path.join(path, "dataset_cfg.yaml"), "w") as f:
            yaml.dump(dataset_cfg, f)

    return model_cfg, dataset_cfg


def record_config(model_cfg, dataset_cfg, output_path):
    """ Record dataset config to the log path. """
    # Record model config
    with open(os.path.join(output_path, "model_cfg.yaml"), "w") as f:
            yaml.safe_dump(model_cfg, f)

    # Record dataset config
    with open(os.path.join(output_path, "dataset_cfg.yaml"), "w") as f:
            yaml.safe_dump(dataset_cfg, f)

