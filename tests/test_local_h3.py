"""Numerical oracles for expanding Diffusers into local PyTorch implementations."""

import sys
import json
import subprocess
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file

from h3.scheduler import MiniMaxH3Scheduler
from h3.modules.vae import AutoencoderKLMiniMaxH3


def original_transformer_checkpoint(directory):
    """Build a small source-format fixture with distinguishable fused Q/K/V rows."""
    from fixtures_h3 import tiny_model

    from h3.checkpoint import get_transformer_key_plan, MINIMAX_H3_FP32_SOURCE_PREFIXES

    config = dict(tiny_model().config)
    plan = get_transformer_key_plan(config)
    source = {}
    for name, destinations in plan.items():
        shape = list(destinations[0][1]) if destinations else [config["rope_freq_dim"]]
        if len(destinations) == 3:
            shape[0] *= 3
        dtype = torch.float32 if name.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16
        source[name] = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 127).to(dtype)
    checkpoint = directory / "transformer"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in source}})
    )
    save_file(source, checkpoint / "weights.safetensors")
    return config, source


def test_original_transformer_load_preserves_dtypes_qkv_and_rng(tmp_path):
    from h3.modules.model import MiniMaxH3Transformer3DModel

    config, source = original_transformer_checkpoint(tmp_path)
    rng = torch.get_rng_state().clone()
    model = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path, progress=None)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not value.is_meta and not value.requires_grad for value in model.parameters())
    assert all(not value.is_meta for value in model.buffers())
    assert model.proj_in.weight.dtype == torch.float32
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert torch.equal(model.proj_in.weight, source["video_patch_proj.weight"])

    # Source Q/K/V rows are interleaved per head, unlike the local projections.
    fused = source["blocks.0.attn.qkv_proj.weight"].reshape(
        config["num_attention_heads"], 3, config["attention_head_dim"], config["hidden_size"]
    )
    for index, projection in enumerate(("to_q", "to_k", "to_v")):
        expected = fused[:, index].flatten(0, 1)
        assert torch.equal(getattr(model.transformer_blocks[0].attn, projection).weight, expected)


def test_original_transformer_load_rejects_incomplete_checkpoint(tmp_path):
    from h3.modules.model import MiniMaxH3Transformer3DModel

    _, source = original_transformer_checkpoint(tmp_path)
    source.pop("video_patch_proj.weight")
    save_file(source, tmp_path / "transformer/weights.safetensors")
    with pytest.raises(ValueError, match="Incomplete original checkpoint"):
        MiniMaxH3Transformer3DModel.from_pretrained(tmp_path, progress=None)


def vae_config():
    return dict(
        latent_channels=4,
        block_out_channels=(8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=24,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )


@pytest.mark.parametrize("frames", [5, 22, 39])
def test_local_vae_matches_pinned_encoder_decoder_and_posterior(frames, tmp_path):
    from diffusers import AutoencoderKLMiniMaxH3 as Reference

    torch.manual_seed(79)
    reference = Reference(**vae_config()).eval()
    model = AutoencoderKLMiniMaxH3(**vae_config()).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert reference.state_dict().keys() == model.state_dict().keys()
    pixels = torch.randn(1, 3, frames, 16, 16)
    with torch.no_grad():
        a, b = reference.encode(pixels).latent_dist, model.encode(pixels).latent_dist
        torch.testing.assert_close(a.parameters, b.parameters, rtol=0, atol=0)
        torch.testing.assert_close(
            a.sample(torch.Generator().manual_seed(7)),
            b.sample(torch.Generator().manual_seed(7)),
            rtol=0,
            atol=0,
        )
        z = a.mode()
        if frames == 5:
            # Pinned upstream cannot decode two-token tails. Its normal overlap
            # decoder supplies the oracle after repeating the tail to 7 tokens.
            z = torch.cat((z, z[:, :, -1:].expand(-1, -1, 5, -1, -1)), dim=2)
        torch.testing.assert_close(
            reference.decode(z).sample[:, :, :frames], model.decode(b.mode()).sample, rtol=0, atol=0
        )
    (tmp_path / "config.json").write_text(json.dumps(vae_config()))
    save_file(model.state_dict(), tmp_path / "diffusion_pytorch_model.safetensors")
    restored = AutoencoderKLMiniMaxH3.from_pretrained(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.decode(b.mode()).sample, model.decode(b.mode()).sample, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_native_scheduler_matches_every_step(dtype):
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler as Reference

    a, b = Reference(shift=12), MiniMaxH3Scheduler(shift=12)
    a.set_timesteps(20)
    b.set_timesteps(20)
    assert torch.equal(a.timesteps, b.timesteps) and torch.equal(a.sigmas, b.sigmas)
    x, noise = torch.randn(2, 4, dtype=dtype), torch.randn(2, 4, dtype=dtype)
    assert torch.equal(a.scale_noise(x, 0.7, noise), b.scale_noise(x, 0.7, noise))
    for t in a.timesteps:
        velocity = torch.randn_like(x)
        expected = a.step(velocity, t, x).prev_sample
        actual = b.step(velocity, t, x).prev_sample
        assert torch.equal(expected, actual)
        x = actual


def test_core_and_training_import_without_diffusers():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
class NoDiffusers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'diffusers' or fullname.startswith('diffusers.'):
            raise RuntimeError('Runtime imported Diffusers: ' + fullname)
sys.meta_path.insert(0, NoDiffusers())
import h3.modules.model, h3.modules.vae, h3.checkpoint
import model.diffusion, trainer.diffusion, pipeline.chunked_inference
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
