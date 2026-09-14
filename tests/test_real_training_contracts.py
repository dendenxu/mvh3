import io

import torch

from h3.data import temporal_layout
from h3.modules.masking import TokenLayout


def test_arbitrary_temporal_lengths_keep_real_tail():
    for frames in (1, 2, 5, 17, 22, 39, 77, 90, 297):
        layout = temporal_layout(frames)
        assert layout.padded_frames >= frames
        assert layout.camera_frames.max() == frames - 1
        assert layout.camera_frames.dtype == torch.long
        assert layout.valid.any()
        assert (layout.camera_frames < frames).all()
    short = temporal_layout(77)
    assert short.padded_frames == 90
    assert len(short.valid) == 27 and int(short.valid.sum()) == 23
    assert int(short.camera_frames[short.valid][-1]) == 76


def test_trainable_attention_preserves_sub_bf16_updates_and_resumes():
    from fixtures_h3 import tiny_model
    from test_worldviews import recipe

    model = tiny_model().bfloat16()
    model.configure_attention(recipe())
    parameter = model.transformer_blocks[0].attn.to_q.weight
    assert parameter.requires_grad and parameter.dtype == torch.float32
    with torch.no_grad():
        parameter.fill_(1)
    optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    # The update is smaller than a BF16 unit but survives in the actual weight
    # storage used by FSDP; forward casts do not replace that FP32 storage.
    assert (parameter < 1).all()
    assert torch.equal(parameter.bfloat16(), torch.ones_like(parameter, dtype=torch.bfloat16))
    buffer = io.BytesIO()
    torch.save(dict(weight=parameter.detach(), optimizer=optimizer.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed_parameter = torch.nn.Parameter(saved["weight"].clone())
    resumed = torch.optim.AdamW([resumed_parameter], lr=1e-6, weight_decay=0)
    resumed.load_state_dict(saved["optimizer"])
    parameter.grad = torch.ones_like(parameter)
    resumed_parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    resumed.step()
    assert torch.equal(parameter, resumed_parameter)


def test_padding_cannot_relay_into_valid_tokens():
    layout = TokenLayout(
        torch.tensor([0, 1, 2, 2]),
        torch.tensor([-1, 0, 1, 1]),
        torch.tensor([-1, 0, 0, 0]),
        True,
        torch.tensor([True, True, True, False]),
    )
    mask = layout.dense()
    assert not mask[:3, 3].any()
    assert mask[3].sum() == 1 and mask[3, 3]
