# [NOTE] modified from the pixel-perfect-sfm project

import sys
import time

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as tvf
from _limap import _features
from pycolmap import logging
from torchvision import transforms

from .models.s2dnet import S2DNet
from .models.vggnet import VGGNet

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=RGB_mean, std=RGB_std)]
)

to_grayscale = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ]
)


# Interface
class Extractor(torch.nn.Module):
    """Dense feature extractor.

    Args:
        device: torch device name (cuda, cpu, cuda:0, ...).

    Attributes:
        model (torch.NN.Module): Torch model used for extraction (else None).
        num_levels (int): Number of levels returned by this network.
        channels (List[int]): output channels for each level of featuremaps.

    """

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.num_levels = 0
        self.model = None
        self.channels = []
        self.l2_normalize = False
        self.offset = np.array([[0.0, 0.0]])

    def forward(self, pil_img: PIL.Image):
        return self.extract_featuremaps(self.adapt_image(pil_img))

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        """
        Extract list of featuremaps where first is finest and last is coarsest.

        Args:
            image_batch: [BxHxWxC] Tensor.

        Returns:
            List of self.num_levels featuremaps as \
            torch.Tensor[HxWxC] on device.
        """
        raise NotImplementedError()

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        """
        Preprocess images. The output is later forwarded to extract_featuremaps.

        Args:
            pil_img: RGB PIL-Image, not preprocessed.

        Returns:
            Preprocessed and corrected image as torch.Tensor[1xHxWxC].
        """
        raise NotImplementedError()


class S2DNetExtractor(Extractor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(device)
        self.model = S2DNet().to(device)
        self.num_levels = 1
        self.channels = [128]  # ,128] #[128,128]
        self.l2_normalize = True
        if "output_channels" in kwargs:
            self.channels = [min(kwargs["output_channels"], self.channels[0])]

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        maps = self.model(image_batch)["feature_maps"]

        for i, channels in enumerate(self.channels):
            if channels != 128:
                maps[i] = maps[i][:, :channels]
        early = maps[0]
        return [early]  # ,middle]

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        return tvf.to_tensor(pil_img).unsqueeze(0)


class DSIFTExtractor(Extractor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(device)
        self.model = None
        self.num_levels = 1
        self.channels = [128]
        self.l2_normalize = True  # already normalized
        self.step = 1
        self.bin_size = 4
        sys.path.append("lib/sift-flow-gpu")
        from sift_flow_torch import SiftFlowTorch

        self.model = SiftFlowTorch()

        self.pad = torch.nn.ZeroPad2d(int(round(1.5 * self.bin_size)))

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        cpu_image = image_batch.squeeze().cpu().numpy()
        numpy_fmap = _features.extract_dsift(
            cpu_image, self.step, self.bin_size
        )
        res = self.pad(
            torch.Tensor(numpy_fmap)
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        logging.info(res.shape, image_batch.shape)
        return [res]

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        t = time.time()
        res = to_grayscale(pil_img).unsqueeze(0)
        logging.info(time.time() - t)
        return res


class VGGNetExtractor(Extractor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(device)
        self.model = VGGNet().to(device)
        self.num_levels = 1  # 2
        self.channels = [64]  # [128,128]
        self.l2_normalize = True
        if "output_channels" in kwargs:
            self.channels = [min(kwargs["output_channels"], self.channels[0])]

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        maps = self.model(image_batch)["feature_maps"]

        for i, channels in enumerate(self.channels):
            if channels != 64:
                maps[i] = maps[i][:, :channels]
        early = maps[0]
        middle = maps[1]
        return [early, middle]

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        return tvf.to_tensor(pil_img).unsqueeze(0)


class ImageExtractor(Extractor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(device)
        self.model = None
        self.num_levels = 1
        self.channels = [3]

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        return [image_batch]

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        return tvf.to_tensor(pil_img).unsqueeze(0)


class GrayscaleExtractor(Extractor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(device)
        self.model = None
        self.num_levels = 1
        self.channels = [1]

    def extract_featuremaps(self, image_batch: torch.Tensor) -> list:
        return [image_batch]

    def adapt_image(self, pil_img: PIL.Image) -> torch.Tensor:
        return to_grayscale(pil_img).unsqueeze(0)


def load_extractor(
    extractor_name: str, device: str, *args, **kwargs
) -> Extractor:
    """
    Load extractor by name onto device.

    Args:
        extractor_name: Name of extractor.
        device: Device where extractor should be loaded on.

    Returns:
        Extractor: Extractor model

    Raises:
        NotImplementedError: When no extractor matching extractor_name is found.
    """
    if extractor_name == "s2dnet":
        return S2DNetExtractor(device, *args, **kwargs)
    if extractor_name == "vggnet":
        return VGGNetExtractor(device, *args, **kwargs)
    if extractor_name == "image":
        return ImageExtractor(device, *args, **kwargs)
    if extractor_name == "grayscale":
        return GrayscaleExtractor(device, *args, **kwargs)
    if extractor_name == "dsift":
        return DSIFTExtractor(device, *args, **kwargs)
    raise NotImplementedError(extractor_name)
