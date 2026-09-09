import io

import pytest
import torch

from diffusers import MiniMaxH3Transformer3DModel
from h3 import MVH3Transformer3DModel
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout
from h3.utils.training import attention_parameters, flow_matching_loss, parameter_signature

from fixtures_h3 import tiny_model


def inputs(audio_tokens=0, batch=2):
    # Deliberately interleave video/text so modality offsets cannot be assumed.
    video_indices = torch.tensor([0, 2, 4, 6, 7, 8, 9, 10])
    text_indices = torch.tensor([1, 3, 5])
    audio_indices = torch.arange(11, 11 + audio_tokens)
    length = 11 + audio_tokens
    tags = torch.zeros(length, dtype=torch.long)
    tags[text_indices], tags[audio_indices] = 1, 2
    position_ids = torch.arange(length * 3).reshape(length, 3).float() / 7
    pose = torch.zeros(batch, 2, 10)
    pose[..., :2] = 1
    pose[:, 1, 7] = 0.15
    camera_indices = torch.full((length, ), -1, dtype=torch.long)
    camera_indices[video_indices] = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    return dict(
        hidden_states=torch.randn(batch, 8, 96),
        audio_hidden_states=torch.randn(batch, audio_tokens, 32),
        encoder_hidden_states=torch.randn(batch, 3, 32),
        timestep=torch.tensor([0.5]),
        timestep_indices=torch.zeros(length, dtype=torch.long),
        token_tags=tags,
        position_ids=position_ids,
        video_indices=video_indices,
        text_indices=text_indices,
        audio_indices=audio_indices,
        camera_pose=pose,
        camera_indices=camera_indices,
    )


def test_full_architecture_parameter_count_is_unchanged():
    with torch.device("meta"):
        upstream = MiniMaxH3Transformer3DModel()
        adapted = MVH3Transformer3DModel()
    assert parameter_signature(adapted) == parameter_signature(upstream)
    assert sum(p.numel() for p in adapted.parameters()) == 33_122_992_896
    assert sum(p.numel() for p in attention_parameters(adapted)) == 7_707_046_400


@pytest.mark.parametrize("audio_tokens", [0, 2])
def test_exact_upstream_parity_without_camera(audio_tokens):
    torch.manual_seed(4)
    upstream, model = tiny_model(MiniMaxH3Transformer3DModel), tiny_model()
    model.load_state_dict(upstream.state_dict(), strict=True)
    assert parameter_signature(model) == parameter_signature(upstream)
    assert model.state_dict().keys() == upstream.state_dict().keys()
    data = inputs(audio_tokens)
    data.pop("camera_pose")
    data.pop("camera_indices")
    with torch.no_grad():
        original, adapted = upstream(**data), model(**data)
    assert torch.equal(original.sample, adapted.sample)
    assert torch.equal(original.audio_sample, adapted.audio_sample)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_existing_attention_update_and_stage_resume(checkpointing):
    torch.manual_seed(5)
    model = tiny_model().train()
    if checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(attention_parameters(model), lr=1e-4)
    signature = parameter_signature(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    data = inputs(audio_tokens=2)
    output = model(**data)
    loss_mask = torch.zeros(2, 8, dtype=torch.bool)
    loss_mask[:, 4:] = True
    target = torch.randn_like(output.sample)
    loss = flow_matching_loss(output.sample, target, loss_mask)
    loss.backward()
    changed = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, before[name]):
            assert parameter.requires_grad, name
            changed.append(name)
    assert len(changed) == 18
    assert parameter_signature(model) == signature
    buffer = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), global_step=12, stage="short_mono"),
               buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    resumed = tiny_model()
    resumed.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(attention_parameters(resumed), lr=1e-4)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    assert checkpoint["global_step"] == 12
    assert parameter_signature(resumed) == signature
    assert len(resumed_optimizer.state) == 18
    with torch.no_grad():
        torch.testing.assert_close(resumed(**data).sample, model(**data).sample, rtol=0, atol=0)


def test_camera_changes_prediction_and_rejects_text_pose():
    model = tiny_model().eval()
    data = inputs()
    with torch.no_grad():
        a = model(**data).sample
        data["camera_pose"][:, 1, 7] += 0.5
        b = model(**data).sample
    assert not torch.allclose(a, b)
    data["camera_indices"][data["text_indices"][0]] = 0
    with pytest.raises(ValueError, match="Text and audio"):
        model(**data)


def test_future_media_and_caption_cannot_leak_across_three_blocks():
    torch.manual_seed(7)
    model = tiny_model().eval()
    data = inputs(audio_tokens=2, batch=1)
    kind = torch.full((13, ), CONDITION, dtype=torch.long)
    chunk = torch.full((13, ), -1, dtype=torch.long)
    scope = torch.zeros(13, dtype=torch.long)
    vi = data["video_indices"]
    kind[vi[:4]], kind[vi[4:]] = CLEAN, NOISY
    chunk[vi] = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    scope[data["text_indices"][0]] = -1
    chunk[data["text_indices"][1:]] = torch.tensor([0, 1])
    kind[data["audio_indices"]] = CLEAN
    chunk[data["audio_indices"]] = torch.tensor([0, 1])
    layout = TokenLayout(kind, chunk, scope)
    data["attention_mask"] = layout.dense()
    with torch.no_grad():
        original = model(**data).sample
        data["hidden_states"][:, 2:4] += 20  # Future clean video.
        data["audio_hidden_states"][:, 1] -= 30  # Future clean audio.
        data["encoder_hidden_states"][:, 2] += 10  # Future local caption; includes text-refiner path.
        changed = model(**data).sample
    torch.testing.assert_close(original[:, (0, 1, 4, 5)], changed[:, (0, 1, 4, 5)], rtol=0, atol=0)
    assert not torch.allclose(original[:, (2, 3, 6, 7)], changed[:, (2, 3, 6, 7)])


def test_stage1_batch_items_are_independent():
    torch.manual_seed(9)
    model = tiny_model().eval()
    data = inputs(batch=2)
    with torch.no_grad():
        first = model(**data).sample
        data["hidden_states"][1] += 10
        data["encoder_hidden_states"][1] -= 15
        data["camera_pose"][1, :, 7] += 1
        second = model(**data).sample
    assert torch.equal(first[0], second[0])
    assert not torch.allclose(first[1], second[1])
