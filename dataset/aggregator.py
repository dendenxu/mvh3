# Simple dataset aggregator for co-training
# Per-sample weighted random choice of sub-dataset
import math
import random
from torch.utils.data import Dataset

from utils.console import log
from utils.console import green
from utils.console import blue
from utils.console import yellow
from utils.console import red
from utils.distributed import is_main_process
from utils.distributed import is_node_main


class DatasetAggregator(Dataset):
    """Weighted random dataset selector.

    Each __getitem__ call picks a sub-dataset via weighted random choice
    (seeded from idx for reproducibility), then delegates to that dataset's
    __getitem__.

    Mixed tensor shapes across ranks are supported: `gather_mixed_batch`
    (utils/distributed.py) uses per-rank broadcasts rather than a single
    all_gather_into_tensor, so different ranks can produce batches with
    different gen_size, mv_size, channel count, even different tensor keys.
    Co-training datasets can therefore have different shapes.

    Weighting (when weights=None):
        weight = effective_samples ** sampling_weight_power
        Default sampling_weight_power = 0.8 (mild rebalancing, closer to
        size-proportional — limits small-dataset over-exposure). Lower values
        (0.6, 0.5 = sqrt, 0.4) boost small datasets more / further suppress
        large ones. Each sub-dataset may set its own power via the
        `sampling_weight_power` kwarg.

    Args:
        datasets: list of Dataset instances (each must have n_seqs and effective_samples properties)
        weights: explicit sampling weights per dataset (overrides auto-computation)
    """

    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        if weights is None:
            eff = [ds.effective_samples for ds in datasets]
            # sampling_weight_power controls how effective_samples maps to weight:
            #   default 0.8 → eff^0.8 (mild rebalancing, near size-proportional)
            #   0.6 / 0.5   → stronger small-dataset boost (0.5 = sqrt)
            #   0.25        → sqrt(sqrt(eff)) = eff^0.25 (reduced weight for lower-quality data)
            # Datasets with effective_samples == 0 (empty after length prefilter)
            # get weight=0 and are never picked by random.choices below.
            weights = [
                (e ** getattr(ds, 'sampling_weight_power', 0.8)) if e > 0 else 0.0
                for ds, e in zip(datasets, eff)
            ]
            if is_node_main():
                total_w = sum(weights)
                if total_w == 0:
                    log(red('[Aggregator] FATAL: all sub-datasets are empty '
                            '(effective_samples=0). Check the per-dataset '
                            'filter warnings above.'))
                for ds, e, w in zip(datasets, eff, weights):
                    pct = (w / total_w * 100) if total_w > 0 else 0.0
                    swp = getattr(ds, 'sampling_weight_power', 0.8)
                    tag = f' [p={swp}]' if swp != 0.8 else ''
                    name = getattr(ds, 'data_path', type(ds).__name__)
                    # Show just the filename for readability
                    if isinstance(name, str) and '/' in name:
                        name = name.rsplit('/', 1)[-1]
                    if w == 0:
                        log(red(f'  Aggregator: {name}{tag} — DROPPED '
                                f'(0 viable rows after length prefilter)'))
                    else:
                        log(f'  Aggregator: {blue(name)}{tag} — {ds.n_seqs} scenes, '
                            f'{green(e)} effective samples, weight={yellow(f"{pct:.1f}%")}')
        total = sum(weights)
        if total == 0:
            # Avoid divide-by-zero — caller will surface the fatal log above.
            self.weights = [0.0 for _ in weights]
        else:
            self.weights = [w / total for w in weights]

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    @property
    def n_seqs(self):
        return sum(ds.n_seqs for ds in self.datasets)

    def __getitem__(self, idx):
        # Seed choice from idx so it's reproducible and independent of global RNG state.
        # Same idx always picks the same dataset — over many idx values, the
        # distribution matches `weights`. The sub-dataset's own getitem_impl
        # will call set_seed(idx // len(shard)) afterwards.
        rng = random.Random(idx)
        ds_idx = rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        return self.datasets[ds_idx][idx]

    @staticmethod
    def collate_fn(samples):
        """Aggregator collate: same per-sample `view_as_batch` flatten as
        DynamicDataset.collate_fn. Whichever sub-dataset produced this sample
        (Dynamic / Static / Mvgame / Multiview), the flatten is gated on
        `cpu['view_as_batch']` so samples without the flag pass through
        unchanged.
        """
        return view_as_batch_collate(samples)


def view_as_batch_collate(samples):
    """Module-level collate (picklable for DataLoader workers).

    For view_as_batch samples (DynamicDataset's full-res strip path) each sample
    bundles `mv` views that the model consumes as `mv` independent single-view
    batch elements, so we fold that `mv` axis into the batch dim:

      * model-input tensors (frames/projs/...) are default_collate'd to
        [B, mv, ...] then flattened to [B*mv, ...];
      * the cpu metadata dict is collated HERE rather than handed to
        default_collate (which would stack per-view arrays to [B, mv] and
        transpose per-view str lists view-major — the exact misalignment that
        used to force per-consumer unwrap hacks downstream). One field-name-
        agnostic rule: a cpu value that is a length-mv sequence is per-view and
        is folded into the batch in sample-major order (matching
        frames.flatten(0, 1) — s0v0, s0v1, …, s1v0, …); a scalar / dict /
        otherwise-shaped value has no mv axis and is collated normally.
        "length == mv" is exactly what "mv is the batch dim" means.

      str/bytes/dict/Tensor are excluded from the per-view test on purpose:
      they are either shared scalars (e.g. the single `video_path` string, set
      to video_paths_list[0]) or shared dicts (the `pack`/`aug_kwargs` dicts) or
      already-batchable by default_collate, and none of them carry the mv axis we
      are trying to fold. (Note the per-view caption list is emitted under
      `prompts` as a length-mv list, so it is correctly folded, not excluded.)
      CAVEAT: the rule is
      structural, not name-based — a *shared* cpu field that happened to be a
      non-str/dict/Tensor sequence of length exactly mv would be misclassified
      as per-view and wrongly flattened. No such field exists today (every
      length-mv cpu value emitted by DynamicDataset is genuinely per-view), but
      a future shared length-mv list would need an explicit exclusion here.

    Non view_as_batch samples pass straight through default_collate unchanged,
    so for every dataset that never sets the flag this collate is byte-identical
    to plain default_collate (the early return below is hit before any folding).
    """
    import torch
    from torch.utils.data._utils.collate import default_collate

    if not any(s.get('cpu', {}).get('view_as_batch') for s in samples):
        return default_collate(samples)

    # Fold factor comes from `orig_mv` (the real per-view count), NOT batch['mv']:
    # the view_as_batch producer sets batch['mv']=1 (each view is a single-view
    # element) and stashes the true view count in orig_mv. Read from samples[0]
    # and applied to every sample — assumes one collate's samples share an mv,
    # which holds for the canonical batch_size=1 per GPU.
    mv = int(samples[0]['cpu'].get('orig_mv', 1))

    # Tensor half: collate everything except cpu, then fold mv into the batch.
    batched = default_collate([{k: v for k, v in s.items() if k != 'cpu'}
                               for s in samples])
    for k in ('frames', 'projs', 'projs_inv', 'Ks', 'Rs', 'Ts', 'prompt_embeds'):
        t = batched.get(k)
        if isinstance(t, torch.Tensor) and t.ndim >= 2:
            batched[k] = t.flatten(0, 1)

    # cpu half: fold per-view (length-mv) fields into the batch; collate the rest
    # (scalars, dicts, the shared pack) normally.
    def is_per_view(v):
        return (not isinstance(v, (str, bytes, dict, torch.Tensor))
                and hasattr(v, '__len__') and len(v) == mv)
    cpu_out = {}
    for key in samples[0]['cpu']:
        vals = [s['cpu'][key] for s in samples]
        cpu_out[key] = ([x for v in vals for x in v] if is_per_view(vals[0])
                        else default_collate(vals))
    batched['cpu'] = cpu_out
    return batched
