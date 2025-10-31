import copy

import yaml


def update_recursive(dict1, dictinfo):
    for k, v in dictinfo.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def update_recursive_deepcopy(dict1, dictinfo):
    dict1_copy = copy.deepcopy(dict1)
    update_recursive(dict1_copy, dictinfo)
    return dict1_copy


def load_config(config_file, default_path=None):
    with open(config_file) as f:
        cfg_loaded = yaml.load(f, Loader=yaml.Loader)

    base_config_file = cfg_loaded.get("base_config_file")
    if base_config_file is not None:
        cfg = load_config(base_config_file)
    elif (default_path is not None) and (config_file != default_path):
        cfg = load_config(default_path)
    else:
        cfg = dict()
    update_recursive(cfg, cfg_loaded)
    return cfg


def update_config(cfg, unknown, shortcuts):
    def get_val_from_keys(cfg, keys):
        v = cfg
        for key in keys:
            v = v[key]
        return v

    for idx, arg in enumerate(unknown):
        if arg in shortcuts:
            unknown[idx] = shortcuts[arg]

    for idx, arg in enumerate(unknown):
        # test if it is a key
        if not arg.startswith("--"):
            continue

        # process value
        keys = arg.replace("--", "").split(".")
        val = get_val_from_keys(cfg, keys)
        argtype = type(val)
        if argtype is bool:
            # test if it is a store action
            if idx == len(unknown) - 1 or unknown[idx + 1].startswith("--"):
                v = True
            else:
                v = unknown[idx + 1].lower() == "true"
        else:
            v = unknown[idx + 1]
            if val is not None:
                if argtype is list:
                    if v.startswith("["):
                        v = eval(v)
                    else:
                        for i in range(idx + 2, len(unknown)):
                            if unknown[i].startswith("--"):
                                break
                            v += "," + unknown[i]
                        v = eval("[" + v + "]")
                v = argtype(v)

        if isinstance(v, str) and (v.lower() == "none" or v.lower() == "null"):
            v = None

        # modify value
        n_lvls = len(keys)
        if n_lvls == 1:
            cfg[keys[0]] = v
        elif n_lvls == 2:
            cfg[keys[0]][keys[1]] = v
        elif n_lvls == 3:
            cfg[keys[0]][keys[1]][keys[2]] = v
        elif n_lvls == 4:
            cfg[keys[0]][keys[1]][keys[2]][keys[3]] = v
        elif n_lvls == 5:
            cfg[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = v
        else:
            raise ValueError("number of levels are too high to handle!!")
    return cfg
