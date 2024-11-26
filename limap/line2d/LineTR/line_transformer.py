from copy import deepcopy
from pathlib import Path

import torch
from einops import repeat
from pycolmap import logging
from torch import nn

from .line_attention import FeedForward, MultiHeadAttention
from .line_process import *


def MLP(channels: list, do_bn=True):  # channels [3, 32, 64, 128, 256, 256]
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keylines(klines, kplines, image_shape):
    """Normalize keylines locations based on image shape"""
    if len(image_shape) == 2:
        height, width = image_shape
    else:
        _, _, height, width = image_shape  # height: 480, width: 640
    one = klines.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7

    normalized_keylines = torch.zeros_like(klines)
    normalized_keylines[:, :, 0] = (
        klines[:, :, 0] - center[:, None, :]
    ) / scaling[:, None, :]
    normalized_keylines[:, :, 1] = (
        klines[:, :, 1] - center[:, None, :]
    ) / scaling[:, None, :]

    normalized_kplines = (kplines - center[:, None, None, :]) / scaling[
        :, None, None, :
    ]
    return normalized_keylines, normalized_kplines


class LinePositionalEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + layers + [feature_dim])
        nn.init.constant_(
            self.encoder[-1].bias, 0.0
        )  # Fills the input Tensor with the value

    def forward(self, klines, responses, angles):
        mid_points = (klines[:, :, 0] + klines[:, :, 1]) / 2.0
        inputs = [
            mid_points.transpose(1, 2),
            responses.transpose(1, 2),
            angles.transpose(1, 2),
        ]
        return self.encoder(torch.cat(inputs, dim=1))  # [3, 5, 128]


class WordPositionalEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

        self.feature_dim = feature_dim

    def forward(self, words_in_line, scores_in_line):
        num_batch = words_in_line.size(0)
        num_sublines = words_in_line.size(1)
        num_words = words_in_line.size(2)
        d_input = words_in_line.size(3) + scores_in_line.size(3)

        inputs = torch.cat(
            [words_in_line, scores_in_line], dim=-1
        )  # [1, 128, 21, 2+1]
        inputs = inputs.transpose(-2, -1).reshape(
            num_batch * num_sublines, d_input, num_words
        )  # [128, 2+1, 21]

        pos_enc = self.encoder(inputs)
        pos_enc = pos_enc.transpose(-1, -2).reshape(
            num_batch, num_sublines, num_words, self.feature_dim
        )  # [batch, 128, 2+1, 256]

        return pos_enc


class LineDescriptiveEncoder(nn.Module):
    """Line Descriptive Network using the transformer"""

    def __init__(
        self,
        d_feature: int,
        n_heads: int,
        n_att_layers: int,
        d_inner: int,
        dropout=0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_feature)
        self.pos_ffn = FeedForward(d_feature, d_inner, dropout=dropout)

    def forward(self, desc, slf_attn_mask=None):
        n_batch = desc.size(0)
        n_sublines = desc.size(1)
        n_words = desc.size(2)
        d_feature = desc.size(3)

        desc, enc_slf_attn = self.slf_attn(desc, desc, desc, mask=slf_attn_mask)
        desc = self.pos_ffn(desc)

        return desc, enc_slf_attn


class KeylineEncoder(nn.Module):
    """Line Descriptive Networks & Positional Embedding for Tokens and Lines"""

    def __init__(
        self,
        feature_dim,
        mlp_layers,
        n_heads,
        n_att_layers,
        d_inner,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.line_position_enc = LinePositionalEncoder(feature_dim, mlp_layers)
        self.word_position_enc = WordPositionalEncoder(feature_dim, mlp_layers)

        self.desc_layers = nn.ModuleList(
            [
                LineDescriptiveEncoder(
                    feature_dim, n_heads, n_att_layers, d_inner, dropout=dropout
                )
                for _ in range(n_att_layers)
            ]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, feature_dim))

    def forward(
        self, klines, resp, angle, pnt, desc, score, mask, return_attns=False
    ):
        enc_slf_attn_list = []
        n_batches = klines.size(0)
        n_klines = klines.size(1)
        d_feature = self.feature_dim

        # combine positional embedding for line itself
        klines_pos = self.line_position_enc(klines, resp, angle)

        # combine positional embedding on the points on lines
        desc = desc + self.word_position_enc(pnt, score)

        # CLS token
        cls_tokens = repeat(
            self.cls_token, "() () w d -> b l w d", b=n_batches, l=n_klines
        )
        desc = torch.cat((cls_tokens, desc), dim=2)

        for desc_layer in self.desc_layers:
            enc_output, enc_slf_attn = desc_layer(desc, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # Positional embedding for Line Signature Networks
        sentence = klines_pos + enc_output[:, :, 0, :].transpose(1, 2)

        return sentence


def attention(query, key, value):
    dim = query.shape[1]
    scores = (
        torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    )  # [3, 64, 4, 512] -> [3, 4, 512, 512]
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0  # d_model : 256, num_heads : 4
        self.dim = d_model // num_heads  # dim: 64
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)  # 256x256
        self.proj = nn.ModuleList(
            [deepcopy(self.merge) for _ in range(3)]
        )  # 256x256가 3개. query, key, value

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = (
            layer(x).view(
                batch_dim, self.dim, self.num_heads, -1
            )  # [3, 64, 4, 512]
            for layer, x in zip(self.proj, (query, key, value))
        )
        x, prob = attention(query, key, value)
        return (
            self.merge(
                x.contiguous().view(batch_dim, self.dim * self.num_heads, -1)
            ),
            prob,
        )


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP(
            [feature_dim * 2, feature_dim * 2, feature_dim]
        )  # (512, 512, 256)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message, prob = self.attn(
            x, source, source
        )  # query (x), key, value (source);  message: [3, 256, 512]
        return (
            self.mlp(torch.cat([x, message], dim=1)),
            prob,
        )  # ([3, 256, 512], [3, 256, 512]) -> [3, 512, 512] -> [3, 256, 512]


class SelfAttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionalPropagation(feature_dim, 4)
                for _ in range(len(layer_names))
            ]
        )
        self.names = layer_names

    def forward(self, kline_desc):
        d_desc_kline = kline_desc.size(2)

        for layer in self.layers:
            # for layer, name in zip(self.layers, self.names):
            delta, _ = layer(kline_desc, kline_desc)
            kline_desc = kline_desc + delta

        return kline_desc


class LineTransformer(nn.Module):
    """Line-Transformer Networks including Line Descriptive Nets and Line Signature Nets"""

    default_config = {
        "mode": "test",
        "image_shape": [480, 640],
        "min_length": 16,
        "token_distance": 8,
        "max_tokens": 21,
        "remove_borders": 8,
        "max_keylines": -1,
        "descriptor_dim": 256,
        "keyline_encoder": [32, 64, 128, 256],
        "n_heads": 4,
        "n_line_descriptive_layers": 1,
        "d_inner": 1024,  # d_inner at the Feed_Forward Layer
        "weight_path": None,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.image_shape = self.config["image_shape"]

        ## Line Descriptive Network
        self.klenc = KeylineEncoder(
            self.config["descriptor_dim"],
            self.config["keyline_encoder"],
            self.config["n_heads"],
            self.config["n_line_descriptive_layers"],
            self.config["d_inner"],
        )

        ## Line Signature Network
        self.selfattn = SelfAttentionalLayer(
            self.config["descriptor_dim"],
            ["self", "self", "self", "self", "self", "self", "self"],
        )

        self.final_proj = nn.Conv1d(
            self.config["descriptor_dim"],
            self.config["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )

        if self.config["mode"] == "test":
            if self.config["weight_path"] is None:
                path = Path(__file__).parent / "weights/LineTR_weight.pth"
            else:
                import os

                path = (
                    Path(
                        os.path.join(
                            self.config["weight_path"], "line2d", "LineTR"
                        )
                    )
                    / "weights/LineTR_weight.pth"
                )
            if not path.is_file():
                self.download_model(path)
            self.load_state_dict(torch.load(str(path)))
            logging.info("Loaded Line-Transformer model")

    def download_model(self, path):
        import subprocess

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = "https://github.com/yosungho/LineTR/blob/main/models/weights/LineTR_weight.pth?raw=true"
        cmd = ["wget", link, "-O", str(path)]
        logging.info("Downloading LineTR model...")
        subprocess.run(cmd, check=True)

    def forward(self, data):
        if len(data["klines"]) == 0:
            return self.default_ret()

        klines = data["sublines"]
        resp = data["resp_sublines"]
        angle = data["angle_sublines"]
        pnt_sublines = data["pnt_sublines"]
        desc_sublines = data["desc_sublines"]
        score_sublines = data["score_sublines"]
        mask_sublines = data["mask_sublines"]

        # Keyline normalization.
        klines, pnt_sublines = normalize_keylines(
            klines, pnt_sublines, self.image_shape
        )

        # line descriptive networks
        line_desc_ = self.klenc(
            klines,
            resp,
            angle,
            pnt_sublines,
            desc_sublines,
            score_sublines,
            mask_sublines,
        )

        # line signature networks
        line_desc_ = self.selfattn(line_desc_)
        line_desc_ = self.final_proj(line_desc_)
        line_desc_ = torch.nn.functional.normalize(line_desc_, p=2, dim=1)
        data.update(
            {
                "line_descriptors": line_desc_,
                "valid_lines": torch.ones(
                    data["klines"].shape[:2],
                    dtype=torch.bool,
                    device=klines.device,
                ),
            }
        )
        return data

    def preprocess(
        self, klines_cv, image_shape, pred_superpoint, valid_mask=None
    ):
        """Pre-process for line tokenization"""

        # change line formats
        klines = {
            "klines": klines_cv,
            "length_klines": np.linalg.norm(
                (klines_cv[:, 0] - klines_cv[:, 1]), axis=-1
            ),
            "angles": get_angles(klines_cv),
        }

        # remove_borders
        height, width = self.config["image_shape"] = image_shape
        border = self.config["remove_borders"]

        if valid_mask is None:
            valid_mask = np.ones((height, width))

        # klines = remove_borders(klines, border, height, width, valid_mask)

        # # filter_by_length
        # klines = filter_by_length(klines, self.config['min_length'], self.config['max_keylines'])  # 15 msec
        # num_klines = len(klines['lines'])
        # if num_klines == 0:
        #     klines['mat_klines2sublines'] = torch.empty((1, 0, 0))
        #     return klines

        # line_tokenizer
        klines = line_tokenizer(
            klines,
            self.config["token_distance"],
            self.config["max_tokens"],
            pred_superpoint,
            image_shape[-2:],
        )

        return klines

    def subline2keyline(
        self, distance_sublines, mat_klines2sublines0, mat_klines2sublines1
    ):
        """Convert sublines' distance matrix to keylines' distance matrix"""
        distance_klines = (
            mat_klines2sublines0 @ distance_sublines @ mat_klines2sublines1.T
        )[None]
        return distance_klines

    def default_ret(self):
        """Default return dictionary for the exception"""
        pred = {}
        pred["klines"] = torch.empty((1, 0, 2, 2))
        pred["sublines"] = torch.empty((1, 0, 2, 2))
        pred["line_desc"] = torch.empty((1, 256, 0))
        pred["mat_klines2sublines"] = torch.empty((1, 0, 0))
        return pred
