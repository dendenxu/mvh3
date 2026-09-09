from pathlib import Path
import json

import torch
import yaml

from mvh3.curriculum import short_mono_windows
from mvh3.masking import CLEAN, CONDITION, NOISY, TokenLayout


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
    short = yaml.safe_load((configs / "stage1_short_mono.yaml").read_text())
    full = yaml.safe_load((configs / "stage2_long_multiview.yaml").read_text())
    assert short["data"]["reference"] == full["data"]["reference"]
    assert short["data"]["subset"] is None and full["data"]["subset"] is None
    assert len(json.loads((configs / short["data"]["reference"]).read_text())["datasets"]) == 19
    assert short["data"]["spatial_views"] == 1 and short["data"]["views_as_batch"]
    assert short["model"] == full["model"]
    assert short["model"]["added_trainable_parameters"] == 0
    assert short["transition"]["preserve_optimizer"] and short["transition"]["preserve_global_step"]


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
