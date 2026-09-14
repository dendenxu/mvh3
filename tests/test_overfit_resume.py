"""Diagnostic convergence must never turn failed reproducibility into acceptance."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from overfit import restored_state_checks, resume_evaluation_decision
from test_ema import setup, update
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ema import ShardedEMA


@pytest.mark.parametrize("corrupt", [None, "raw", "adamw", "ema"])
def test_diagnostic_resume_checks_every_saved_state(tmp_path, corrupt):
    model, optimizer, cfg = setup()
    ema = ShardedEMA.from_config(model, cfg)
    update(model, optimizer, ema)
    checkpoint = save_checkpoint(model, optimizer, cfg, 1, 1, {}, tmp_path, ema=ema)
    saved = deepcopy(load_checkpoint(model, optimizer, cfg, checkpoint, ema=ema))
    if corrupt == "raw":
        with torch.no_grad():
            model.weight[0, 0] += .01
    elif corrupt == "adamw":
        optimizer.state[model.weight]["exp_avg"][0, 0] += .01
    elif corrupt == "ema":
        ema.weights["weight"][0, 0] += .01
    checks = restored_state_checks(model, optimizer, ema, saved)
    assert checks == {key: key != corrupt for key in ("raw", "adamw", "ema")}
    decision = resume_evaluation_decision(.0001, .00105, "Bounded convergence probe", [checks])
    assert decision["status"] == "failed"
    assert decision["continue_training"] == (corrupt is None)


def test_diagnostic_continuation_preserves_strict_gate_and_requires_every_rank():
    exact = [dict(raw=True, adamw=True, ema=True) for _ in range(8)]
    strict = resume_evaluation_decision(.0001, .00105)
    assert strict["status"] == "failed" and not strict["continue_training"]
    assert resume_evaluation_decision(.0001, .0009)["continue_training"]
    diagnostic = resume_evaluation_decision(.0001, .00105, "Convergence only", exact)
    assert diagnostic["status"] == "failed" and diagnostic["continue_training"]
    assert diagnostic["relative_tolerance"] == strict["relative_tolerance"] == .001
    assert not resume_evaluation_decision(.0001, .00105, " ", exact)["continue_training"]
    exact[7]["adamw"] = False
    assert not resume_evaluation_decision(.0001, .00105, "Convergence only", exact)["continue_training"]
    assert not resume_evaluation_decision(.0001, .0009, "Convergence only", exact)["continue_training"]


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -.01])
def test_diagnostic_resume_never_accepts_invalid_evaluations(error):
    exact = [dict(raw=True, adamw=True, ema=True)]
    decision = resume_evaluation_decision(.0001, error, "Convergence only", exact)
    assert decision["status"] == "failed" and not decision["continue_training"]
