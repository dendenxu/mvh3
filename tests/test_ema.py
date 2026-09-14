from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ema import ShardedEMA, inference_weight_kind


def setup():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    model.register_parameter("empty_shard", torch.nn.Parameter(torch.empty(0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    cfg = OmegaConf.create(dict(ema_weight=0.9, ema_warmup=False, fs_size=1, sp_size=1, h3={}))
    return model, optimizer, cfg


def update(model, optimizer, ema):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(torch.randn(5, 3)) - torch.randn(5, 2)).square().mean()
    loss.backward()
    optimizer.step()
    ema.update(model)


def test_ema_matches_weighted_optimizer_trajectory_and_preserves_frozen_weights():
    model, optimizer, cfg = setup()
    ema = ShardedEMA.from_config(model, cfg)
    original_bias = model.bias.detach().clone()
    trajectory = [model.weight.detach().double().clone()]
    for _ in range(7):
        update(model, optimizer, ema)
        trajectory.append(model.weight.detach().double().clone())
    expected = trajectory[0] * 0.9 ** 7
    expected += sum(value * (0.1 * 0.9 ** (7 - i)) for i, value in enumerate(trajectory[1:], 1))
    torch.testing.assert_close(ema.weights["weight"].double(), expected, atol=2e-7, rtol=1e-6)
    assert "bias" not in ema.weights and ema.weights["empty_shard"].numel() == 0
    assert torch.equal(model.bias, original_bias)
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in ema.weights.values())


def test_ema_swap_restores_training_state_after_inference_exception():
    model, optimizer, cfg = setup()
    ema = ShardedEMA.from_config(model, cfg)
    update(model, optimizer, ema)
    raw = model.weight.detach().clone()
    gradient = model.weight.grad.clone()
    moments = optimizer.state[model.weight]["exp_avg"].clone()
    with pytest.raises(RuntimeError, match="decode failed"):
        with ema.average_parameters(model):
            assert torch.equal(model.weight, ema.weights["weight"])
            with pytest.raises(RuntimeError, match="Cannot update EMA"):
                ema.update(model)
            raise RuntimeError("decode failed")
    assert torch.equal(model.weight, raw)
    assert torch.equal(model.weight.grad, gradient)
    assert torch.equal(optimizer.state[model.weight]["exp_avg"], moments)
    assert not ema.swapped


def test_checkpoint_resume_preserves_raw_optimizer_and_ema_trajectory(tmp_path):
    torch.manual_seed(41)
    model, optimizer, cfg = setup()
    initial = deepcopy(model)
    ema = ShardedEMA.from_config(model, cfg)
    for _ in range(3):
        update(model, optimizer, ema)
    checkpoint = save_checkpoint(model, optimizer, cfg, 3, 1, {}, tmp_path, ema=ema)
    saved_ema = {name: value.clone() for name, value in ema.weights.items()}
    for _ in range(4):
        update(model, optimizer, ema)
    resumed = deepcopy(initial)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.03)
    resumed_ema = ShardedEMA.from_config(resumed, cfg)
    load_checkpoint(resumed, resumed_optimizer, cfg, checkpoint, ema=resumed_ema)
    assert resumed_ema.num_updates == 3
    for _ in range(4):
        update(resumed, resumed_optimizer, resumed_ema)
    assert resumed_ema.num_updates == ema.num_updates == 7
    for name, value in model.named_parameters():
        assert torch.equal(value, dict(resumed.named_parameters())[name])
    for name, value in ema.weights.items():
        assert torch.equal(value, resumed_ema.weights[name])
    cfg.inference_weights = "ema"
    load_checkpoint(initial, None, cfg, checkpoint, restore_random=False, weights=inference_weight_kind(cfg))
    assert torch.equal(initial.weight, saved_ema["weight"])
    with pytest.raises(ValueError, match="optimizer resume requires raw"):
        load_checkpoint(resumed, resumed_optimizer, cfg, checkpoint, weights="ema")


def test_ema_schedule_and_shard_changes_fail_before_loading():
    model, optimizer, cfg = setup()
    ema = ShardedEMA.from_config(model, cfg)
    update(model, optimizer, ema)
    saved = deepcopy(ema.state_dict())
    saved["decay"] = 0.99
    with pytest.raises(ValueError, match="schedule"):
        ema.load_state_dict(saved)
    with torch.no_grad():
        model.weight.set_(torch.zeros(2))
    with pytest.raises(ValueError, match="local FSDP shards"):
        ema.update(model)
    assert ema.num_updates == 1


def test_ema_warmup_is_continuous_after_restore():
    model, optimizer, cfg = setup()
    cfg.ema_warmup = True
    ema = ShardedEMA.from_config(model, cfg)
    for _ in range(5):
        update(model, optimizer, ema)
    restored = ShardedEMA.from_config(model, cfg)
    restored.load_state_dict(ema.state_dict())
    update(model, optimizer, ema)
    restored.update(model)
    assert restored.num_updates == ema.num_updates == 6
    assert restored.current_decay == ema.current_decay < cfg.ema_weight
    assert torch.equal(restored.weights["weight"], ema.weights["weight"])


def test_checkpoint_requires_matching_ema_updates(tmp_path):
    model, optimizer, cfg = setup()
    ema = ShardedEMA.from_config(model, cfg)
    with pytest.raises(ValueError, match="include the configured EMA"):
        save_checkpoint(model, optimizer, cfg, 0, 1, {}, tmp_path)
    with pytest.raises(ValueError, match="matching EMA updates"):
        save_checkpoint(model, optimizer, cfg, 1, 1, {}, tmp_path, ema=ema)


def test_unsaturated_decay_change_matches_using_new_cap_from_start(tmp_path):
    torch.manual_seed(58)
    model, optimizer, cfg = setup()
    cfg.ema_weight, cfg.ema_warmup = 0.999, True
    ema = ShardedEMA.from_config(model, cfg)
    target = deepcopy(model)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.03)
    target_cfg = deepcopy(cfg)
    target_cfg.ema_weight = 0.995
    target_ema = ShardedEMA.from_config(target, target_cfg)
    for _ in range(16):
        state = torch.get_rng_state()
        update(model, optimizer, ema)
        torch.set_rng_state(state)
        update(target, target_optimizer, target_ema)
    for name in ema.weights:
        assert torch.equal(ema.weights[name], target_ema.weights[name])
    checkpoint = save_checkpoint(model, optimizer, cfg, 16, 1, {}, tmp_path, ema=ema)
    restored = ShardedEMA.from_config(model, target_cfg)
    state = load_checkpoint(model, optimizer, target_cfg, checkpoint, ema=restored)
    assert state["ema_decay_change"] == dict(previous=.999, current=.995, step=16, history_identical=True)
    assert restored.decay == .995 and restored.num_updates == 16
    for _ in range(4):
        state = torch.get_rng_state()
        update(model, optimizer, restored)
        torch.set_rng_state(state)
        update(target, target_optimizer, target_ema)
    for name in restored.weights:
        assert torch.equal(restored.weights[name], target_ema.weights[name])
    assert torch.equal(model.weight, target.weight)


def test_decay_change_cannot_relabel_a_different_history_or_recipe(tmp_path):
    model, optimizer, cfg = setup()
    cfg.ema_warmup = True
    ema = ShardedEMA.from_config(model, cfg)
    for _ in range(5):
        update(model, optimizer, ema)
    checkpoint = save_checkpoint(model, optimizer, cfg, 5, 1, {}, tmp_path, ema=ema)
    changed = deepcopy(cfg)
    changed.ema_weight = .2
    raw = model.weight.detach().clone()
    with pytest.raises(ValueError, match="settings differ"):
        load_checkpoint(model, optimizer, changed, checkpoint, ema=ShardedEMA.from_config(model, changed))
    changed.ema_weight = .8
    changed.h3.condition_noise = .02
    with pytest.raises(ValueError, match="settings differ"):
        load_checkpoint(model, optimizer, changed, checkpoint, ema=ShardedEMA.from_config(model, changed))
    assert torch.equal(model.weight, raw)


def test_explicit_fixed_decay_transition_retains_weights_moments_and_count(tmp_path):
    model, optimizer, cfg = setup()
    cfg.ema_weight, cfg.ema_warmup = .995, True
    ema = ShardedEMA.from_config(model, cfg)
    for _ in range(5):
        update(model, optimizer, ema)
    checkpoint = save_checkpoint(model, optimizer, cfg, 5, 1, {}, tmp_path, ema=ema)
    raw = model.weight.detach().clone()
    moments = deepcopy(optimizer.state_dict())
    averaged = deepcopy(ema.state_dict())
    fixed = deepcopy(cfg)
    fixed.ema_warmup = False
    restored = ShardedEMA.from_config(model, fixed)
    with pytest.raises(ValueError, match="settings differ"):
        load_checkpoint(model, optimizer, fixed, checkpoint, ema=restored)
    state = load_checkpoint(model, optimizer, fixed, checkpoint, ema=restored,
                            ema_schedule_change="User selected fixed 0.995 without warmup")
    assert restored.num_updates == 5 and restored.current_decay == .995 and not restored.warmup
    assert torch.equal(model.weight, raw)
    torch.testing.assert_close(optimizer.state_dict(), moments, rtol=0, atol=0)
    torch.testing.assert_close(restored.weights, averaged["weights"], rtol=0, atol=0)
    assert state["ema_schedule_change"]["previous"] == dict(decay=.995, warmup=True)
    assert state["ema_schedule_change"]["current"] == dict(decay=.995, warmup=False)
    assert state["ema_schedule_change"]["saved_history_preserved"]
    assert not state["ema_schedule_change"]["history_identical_to_new_schedule"]
    expected = restored.weights["weight"].double().clone()
    for _ in range(3):
        update(model, optimizer, restored)
        expected = .995 * expected + .005 * model.weight.detach().double()
        assert restored.current_decay == .995
        torch.testing.assert_close(restored.weights["weight"].double(), expected, atol=2e-7, rtol=1e-6)
    updated = save_checkpoint(model, optimizer, fixed, 8, 1, {}, tmp_path, ema=restored)
    reloaded = ShardedEMA.from_config(model, fixed)
    load_checkpoint(model, optimizer, fixed, updated, ema=reloaded)
    torch.testing.assert_close(restored.state_dict(), reloaded.state_dict(), rtol=0, atol=0)


def test_explicit_schedule_transition_still_rejects_other_recipe_changes(tmp_path):
    model, optimizer, cfg = setup()
    cfg.ema_warmup = True
    ema = ShardedEMA.from_config(model, cfg)
    update(model, optimizer, ema)
    checkpoint = save_checkpoint(model, optimizer, cfg, 1, 1, {}, tmp_path, ema=ema)
    fixed = deepcopy(cfg)
    fixed.ema_warmup = False
    restored = ShardedEMA.from_config(model, fixed)
    raw, averaged = model.weight.detach().clone(), deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match="explicit reason"):
        load_checkpoint(model, optimizer, fixed, checkpoint, ema=restored, ema_schedule_change=" ")
    fixed.h3.condition_noise = .02
    with pytest.raises(ValueError, match="settings differ"):
        load_checkpoint(model, optimizer, fixed, checkpoint, ema=restored, ema_schedule_change="Fixed decay")
    assert torch.equal(model.weight, raw)
    torch.testing.assert_close(restored.state_dict(), averaged, rtol=0, atol=0)
