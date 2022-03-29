import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
import core.utils as utils
import core.visualize as vis

def extract_sold2(cfg):
    import core.detector.SOLD2 as sold2
    imname_list = cfg["imname_list"]
    folder_to_save = cfg["output_folder"]
    heatmap_dir = os.path.join(folder_to_save, 'sold2_heatmaps')
    all_2d_segs, descinfos = sold2.sold2_detect_2d_segs_on_images(imname_list, max_image_dim=cfg["max_image_dim"], heatmap_dir=heatmap_dir)
    with open(os.path.join(folder_to_save, 'sold2_all_2d_segs.npy'), 'wb') as f: np.savez(f, imname_list=imname_list, all_2d_segs=all_2d_segs)
    vis.save_datalist_to_folder(os.path.join(folder_to_save, 'sold2_descinfos'), 'descinfo', imname_list, descinfos)

def extract_lsd(cfg):
    # we still need to run sold2 get its corresponding features
    import core.detector.LSD as lsd
    import core.detector.SOLD2 as sold2
    imname_list = cfg["imname_list"]
    folder_to_save = cfg["output_folder"]
    all_2d_segs = lsd.lsd_detect_2d_segs_on_images(imname_list, max_image_dim=cfg["max_image_dim"])
    with open(os.path.join(folder_to_save, 'lsd_all_2d_segs.npy'), 'wb') as f: np.savez(f, imname_list=imname_list, all_2d_segs=all_2d_segs)
    descinfos = sold2.sold2_compute_descinfos(imname_list, all_2d_segs, max_image_dim=cfg["max_image_dim"])
    vis.save_datalist_to_folder(os.path.join(folder_to_save, 'lsd_descinfos'), 'descinfo', imname_list, descinfos)

def extract_s2dnet(cfg):
    import limap.features as _features
    import torch
    imname_list = cfg["imname_list"]
    folder_to_save = cfg["output_folder"]
    s2dnet_dir = os.path.join(folder_to_save, 'features_s2dnet')
    if not os.path.exists(s2dnet_dir):
        os.makedirs(s2dnet_dir)
    with open(os.path.join(s2dnet_dir, 'imname_list.npy'), 'wb') as f:
        np.savez(f, imname_list=imname_list)
    extractor = _features.S2DNetExtractor("cuda")
    for idx, imname in enumerate(tqdm(imname_list)):
        img = utils.read_image(imname, max_image_dim=cfg["max_image_dim"], set_gray=False)
        input_image = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)[None,...]
        input_image = input_image.to("cuda")
        featuremap = extractor.extract_featuremaps(input_image)
        featuremap = featuremap[0][0].detach().cpu().numpy()
        if cfg["dtype"] == "float16":
            featuremap = featuremap.astype(np.float16)
        fname_out = os.path.join(s2dnet_dir, "feature_{0}.npy".format(idx))
        with open(fname_out, 'wb') as f:
            np.save(fname_out, featuremap)

def extract(cfg):
    if cfg["mode"] == 'sold2':
        extract_sold2(cfg)
    elif cfg["mode"] == 'lsd':
        extract_lsd(cfg)
    elif cfg["mode"] == 's2dnet':
        extract_s2dnet(cfg)
    else:
        raise NotImplementedError

def load_imname_list(fname):
    if fname.endswith('.txt'):
        with open(fname, 'r') as f:
            lines = f.readlines()
        n_images = int(lines[0].strip('\n'))
        assert n_images == len(lines) - 1
        imname_list = [line.strip('\n').split()[1] for line in lines[1:]]
        return imname_list
    elif fname.endswith('.npy'):
        with open(fname, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            imname_list = data["imname_list"]
        return imname_list
    else:
        raise NotImplementedError

def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='extract 2d lines and features')
    arg_parser.add_argument('-n', '--names', type=str, required=True, help='imname_list.npy or txt')
    arg_parser.add_argument('-o', '--output_folder', type=str, default='tmp', help='output folder')
    arg_parser.add_argument('--mode', type=str, required=True, help='mode [sold2, lsd, s2dnet]')
    arg_parser.add_argument('--max_image_dim', type=int, default=-1, help='max image dimension')
    arg_parser.add_argument('--dtype', type=str, default='float16', help='dtype for the feature map [float32, float16]')

    arg_parser.add_argument('-c', '--config_file', type=str, default='cfgs/triangulation/hypersim_triangulation.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str, default='cfgs/triangulation/default_triangulation.yaml', help='default config file')
    args, unknown = arg_parser.parse_known_args()
    cfg = utils.load_config(args.config_file, default_path=args.default_config_file)

    cfg["imname_list"] = load_imname_list(args.names)
    cfg["mode"] = args.mode
    cfg["output_folder"] = args.output_folder
    cfg["dtype"] = args.dtype
    if args.max_image_dim != -1:
        cfg["max_mage_dim"] = args.max_image_dim
    if not os.path.exists(cfg["output_folder"]):
        os.makedirs(cfg["output_folder"])
    cmd = 'cp {0} {1}'.format(args.names, cfg["output_folder"])
    print(cmd)
    os.system(cmd)
    return cfg

if __name__ == '__main__':
    cfg = parse_config()
    extract(cfg)

