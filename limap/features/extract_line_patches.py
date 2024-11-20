import numpy as np
from _limap import _features


def write_patch(fname, patch, dtype="float16"):
    # write out a PatchInfo_f object
    array = patch.array
    if dtype == "float16":
        array = array.astype(np.float16)
    elif dtype == "float32":
        array = array.astype(np.float32)
    with open(fname, "wb") as f:
        np.savez(
            f, array=array, R=patch.R, tvec=patch.tvec, img_hw=patch.img_hw
        )


def load_patch(fname, dtype="float16"):
    # return a PatchInfo_f object
    patch_info_name = f"PatchInfo_f{dtype[-2:]}"
    with open(fname, "rb") as f:
        data = np.load(f, allow_pickle=True)
        patch = getattr(_features, patch_info_name)(
            data["array"], data["R"], data["tvec"], data["img_hw"]
        )
    return patch


def get_line_patch_extractor(cfg, channels):
    lpe_options = _features.LinePatchExtractorOptions(cfg)
    patch_extractor_name = f"LinePatchExtractor_f64_c{channels}"
    extractor = getattr(_features, patch_extractor_name)(lpe_options)
    return extractor


def extract_line_patch_one_image(cfg, track, img_id, camview, feature):
    """
    Returns:
    _features.PatchInfo_fx
    """
    lpe_options = _features.LinePatchExtractorOptions(cfg)
    patch_extractor_name = f"LinePatchExtractor_f64_c{feature.shape[2]}"
    extractor = getattr(_features, patch_extractor_name)(lpe_options)
    patch = extractor.ExtractOneImage(track, img_id, camview, feature)
    return patch


def extract_line_patches(cfg, track, p_camviews, p_features):
    """
    Returns:
    list of _features.PatchInfo_fx
    """
    lpe_options = _features.LinePatchExtractorOptions(cfg)
    patch_extractor_name = f"LinePatchExtractor_f64_c{p_features[0].shape[2]}"
    extractor = getattr(_features, patch_extractor_name)(lpe_options)
    patches = extractor.Extract(track, p_camviews, p_features)
    return patches
