from pathlib import Path
import json

import torch
import yaml

from dataset.curriculum import short_mono_windows
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout


def test_all_views_and_tails_retained():
    counts = [297, 77, 10, 150]
    windows = short_mono_windows(counts)
    for view, count in enumerate(counts):
        covered = []
        for window in windows:
            if window.view == view:
                assert 0 < window.frame_count <= 77
                covered.extend(range(window.start, window.stop))
        assert covered == list(range(count))


def test_both_stages_use_the_same_full_sources():
    configs = Path(__file__).resolve().parents[1] / "configs"
    from omegaconf import OmegaConf
    from utils.config import load_config
    short = load_config(configs / "stage1_short_mono.yaml")
    full = load_config(configs / "stage2_long_multiview.yaml")
    assert OmegaConf.to_container(short.dataset) == OmegaConf.to_container(full.dataset)
    assert len(short.dataset.datasets) == 19
    assert short.h3.stage == 1 and full.h3.stage == 2
    assert short.model == full.model
    assert short.ar_lr == full.ar_lr == 1e-5
    assert short.h3.short_frames == 77


def test_masks_obey_teacher_forcing_and_view_isolation():
    kind = torch.tensor([CONDITION, CONDITION, CONDITION, CLEAN, CLEAN, NOISY, NOISY, CLEAN])
    chunk = torch.tensor([-1, 0, 1, 0, 1, 0, 1, 0])
    scope = torch.tensor([-1, 0, 0, 0, 0, 0, 0, 1])
    mask = TokenLayout(kind, chunk, scope, cross_view=False).dense()
    assert mask[0, 0] and mask[0].sum() == 1
    assert mask[5, 0] and mask[5, 1] and mask[5, 5]
    assert not mask[5, 2] and not mask[5, 3] and not mask[5, 4] and not mask[5, 6]
    assert mask[6, 3] and not mask[6, 4] and not mask[6, 5]
    assert not mask[3, 7] and not mask[7, 3]
    assert TokenLayout(kind, chunk, scope, cross_view=True).dense()[3, 7]
    assert mask.any(dim=-1).all()
