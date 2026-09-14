from pathlib import Path

import torch

from fixtures_h3 import feature_document
from h3.modules.masking import CONDITION
from h3.packing import patchify
from h3.utils.scheduler import MiniMaxH3Scheduler
from model.diffusion import WorldViewsObjective
from utils.config import load_config, validate_config


def native_recipe():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    cfg.h3.checkpoint, cfg.h3.vae = "/unused/h3", "/unused/vae"
    return validate_config(cfg)


def conditioned_document(frames=22):
    document = feature_document(frames=frames)
    view = document["views"][0]
    view["condition"] = {key: view[key][:1].clone()
                         for key in ("pose", "projection", "inverse", "frames", "valid")}
    view["condition"]["latent"] = view["latent"][:, :, :1].clone()
    return document


def test_native_recipe_and_shifted_training_noise():
    cfg = native_recipe()
    assert cfg.model.timestep_shift == cfg.timestep_shift == 12
    assert cfg.guidance_scale == 1 and cfg.sampling_solver == "h3_euler"
    assert cfg.sampling_steps == 50 and cfg.dataset.model_fps == 24
    torch.manual_seed(912)
    uniform = torch.rand(200)
    torch.manual_seed(912)
    sigma, weight, high = WorldViewsObjective(cfg).sample_sigmas(200, "cpu")
    torch.testing.assert_close(sigma, 12 * uniform / (1 + 11 * uniform), rtol=0, atol=0)
    assert weight.eq(1).all() and not high


def test_fixed_noise_evaluation_preserves_native_flow_direction():
    objective, document = WorldViewsObjective(native_recipe()), conditioned_document()
    torch.manual_seed(70000)
    first = objective.pack(document, "cpu", evaluation_sigma=.75)
    torch.manual_seed(70000)
    second = objective.pack(document, "cpu", evaluation_sigma=.75)
    inputs, target, weights, records, _ = first
    assert torch.equal(inputs["hidden_states"], second[0]["hidden_states"])
    assert torch.equal(target, second[1])
    assert weights[weights > 0].eq(1).all()
    record = records[0]
    noisy = patchify(record["noisy"])
    velocity = target[:, record["start"]:record["stop"]]
    clean = patchify(document["views"][0]["latent"])
    torch.testing.assert_close(noisy + .75 * velocity, clean, rtol=1e-5, atol=1e-6)
    times = inputs["timestep"][inputs["timestep_indices"]]
    assert times[inputs["text_indices"]].eq(.25).all()


def test_condition_noise_stays_fixed_across_denoising_and_cache_writes():
    cfg, document = native_recipe(), conditioned_document()
    objective = WorldViewsObjective(cfg)
    conditions = objective.condition_latents(document, "cpu")
    clean = document["views"][0]["condition"]["latent"]
    assert not torch.equal(conditions[0], clean)
    for sigma, cache in ((.8, False), (.2, False), (0., True)):
        state = dict(chunk=0, sigma=sigma, current=[document["views"][0]["latent"]],
                     conditions=conditions, cached=True, update_cache=cache)
        inputs, _, _, _, _ = objective.pack(document, "cpu", inference=state)
        layout = inputs["attention_mask"]
        video = inputs["video_indices"]
        select = layout.kind[video] == CONDITION
        assert torch.equal(inputs["hidden_states"][:, select], patchify(conditions[0]))
        times = inputs["timestep"][inputs["timestep_indices"]][video][select]
        torch.testing.assert_close(times, torch.full_like(times, 1 - min(sigma, .001)))
    assert torch.equal(document["views"][0]["condition"]["latent"], clean)


def test_native_euler_reaches_clean_with_data_ward_velocity():
    scheduler = MiniMaxH3Scheduler(shift=12)
    scheduler.set_timesteps(50)
    assert len(scheduler.timesteps) == 49 and len(scheduler.sigmas) == 50
    clean, noise = torch.randn(7, 24), torch.randn(7, 24)
    sample = noise.clone()
    for timestep in scheduler.timesteps:
        sample = scheduler.step(clean - noise, timestep, sample, return_dict=False)[0]
    torch.testing.assert_close(sample, clean, rtol=1e-5, atol=1e-6)
