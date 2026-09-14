from omegaconf import OmegaConf

from dataset.static import StaticDataset
from dataset.mvgame import MultiViewDataset
from dataset.aggregator import DatasetAggregator
from dataset.presampled import PresampledDataset
from dataset.multiview import MultiViewRealDataset
from dataset.dynamic import DynamicDataset, StaticDynamicDataset

DATASET_REGISTRY = {
    "mvgame": MultiViewDataset,
    "mvgame_raw": MultiViewRealDataset,  # raw mvgame seq (video/<c:06d>.mp4 + parquet pose); no view-aug, full image-aug
    "static": StaticDataset,
    "static_dynamic": StaticDynamicDataset,
    "egoexo4d": MultiViewRealDataset,
    "waymo_e2e": MultiViewRealDataset,
    "waymo_perception": MultiViewRealDataset,
    "dynamic": DynamicDataset,
    "presampled": PresampledDataset,  # self-contained shape-ordered presampled parquet (deterministic)
}


def create_dataset(dataset_cfg, config):
    """Factory for creating datasets from config. Supports single and concat modes.

    Note: dataset classes that implement pose preloading (StaticDataset,
    MultiViewRealDataset, and subclasses like DynamicDataset) run the main-
    process pose preload inside their own `__init__` — they read `num_workers`
    from the cfg via **kwargs. No explicit preload call needed here.
    """
    cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dtype = cfg.pop("type", "mvgame")

    if dtype == "concat":
        sub_cfgs = cfg.pop("datasets")
        weights = cfg.pop("weights", None)

        # Shared fields (height, width, num_workers, etc.) from parent config
        # are inherited by sub-datasets. num_workers in particular needs to
        # flow into each sub-dataset so its __init__ preload matches what the
        # DataLoader will use at runtime.
        shared = {k: v for k, v in cfg.items() if k not in ("type",)}
        datasets = []
        for sc in sub_cfgs:
            merged = {**shared, **sc}
            sub_type = merged.pop("type", "mvgame")
            cls = DATASET_REGISTRY[sub_type]
            datasets.append(cls(config=config, **merged))
        return DatasetAggregator(datasets, weights)
    else:
        cls = DATASET_REGISTRY[dtype]
        return cls(config=config, **cfg)
