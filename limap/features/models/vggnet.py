import torch
import torch.nn as nn
import torchvision.models as models

from .base_model import BaseModel
from .s2dnet import vgg16_layers


class VGGNet(BaseModel):
    default_conf = {
        "hypercolumn_layers": ["conv1_2", "conv3_3"],  # "conv5_3"],
        "checkpointing": None,
        "output_dim": 128,
        "pretrained": "imagenet",
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _init(self, conf=default_conf):
        assert conf.pretrained in ["s2dnet", "imagenet", None]

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [
            self.layer_to_index[n] for n in conf.hypercolumn_layers
        ]
        num_layers = self.hypercolumn_indices[-1] + 1

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=True)
        layers = list(vgg16.features.children())[:num_layers]
        self.encoder = nn.ModuleList(layers)

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

    def _forward(self, data):
        image = data  # data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        # torch.true_divide((image - mean[:, None, None]),)
        del mean, std
        feature_map = image
        feature_maps = []
        start = 0
        for idx in self.hypercolumn_indices:
            if self.conf.checkpointing:
                blocks = list(range(start, idx + 2, self.conf.checkpointing))
                if blocks[-1] != idx + 1:
                    blocks.append(idx + 1)
                for start_, end_ in zip(blocks[:-1], blocks[1:]):
                    feature_map = torch.utils.checkpoint.checkpoint(
                        nn.Sequential(*self.encoder[start_:end_]), feature_map
                    )
            else:
                for i in range(start, idx + 1):
                    feature_map = self.encoder[i](feature_map)
            feature_maps.append(feature_map)
            start = idx + 1

        # feature_maps = self.adaptation_layers(feature_maps)
        return {"feature_maps": feature_maps}

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
