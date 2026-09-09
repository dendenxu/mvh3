from omegaconf import OmegaConf, DictConfig
from utils.console import *


def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    base = OmegaConf.create({})

    parents = cfg.get("parents", OmegaConf.create([]))
    parents = [parents] if not OmegaConf.is_list(parents) else parents

    cfg_dir = dirname(cfg_path)
    cwd = os.getcwd()

    for p in parents:
        # parent path can be absolute, or relative to:
        #   - this cfg file's dir, or
        #   - current working dir

        parent_path = join(cfg_dir, p)
        parent_path = join(cwd, p) if not exists(parent_path) else parent_path
        if not exists(parent_path):
            # raise FileNotFoundError(f"Parent config file not found: {parent_path}")
            log(red(f"Parent config file not found: {parent_path}, ignoring and continuing"))
        parent_cfg = load_cfg(parent_path)
        base = OmegaConf.merge(base, parent_cfg)  # later parents override earlier

    merged = OmegaConf.merge(base, cfg)  # child overrides parents

    return merged


def warn_on_new_keys(base_cfg, new_cfg, current_path=""):
    """
    Recursively checks if keys in new_cfg exist in base_cfg.
    Prints a warning if a key is new.
    """
    # Ensure we are working with DictConfig or dict for iteration
    if not isinstance(new_cfg, (dict, DictConfig)):
        return
    for key in new_cfg:
        # Construct the full path for clearer warnings (e.g., "server.port")
        key_path = f"{current_path}.{key}" if current_path else key
        if key not in base_cfg:
            log(yellow(f"WARNING: Key '{magenta(key_path)}' is present in new config but missing in old config."))
        else:
            # If both values are configs/dicts, recurse deeper
            base_val = base_cfg[key]
            new_val = new_cfg[key]
            if isinstance(base_val, (dict, DictConfig)) and isinstance(new_val, (dict, DictConfig)):
                warn_on_new_keys(base_val, new_val, current_path=key_path)
