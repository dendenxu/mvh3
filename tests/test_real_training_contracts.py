import io

import torch

from h3.data import temporal_layout
from h3.modules.masking import TokenLayout
from h3.utils.optim import MasterAdamW
from h3.packing import teacher_forcing_batch
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler


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


def test_fp32_master_accumulates_sub_bf16_updates_and_resumes():
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = MasterAdamW([("weight", parameter)], lr=1e-6, weight_decay=0)
    for _ in range(8):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert torch.equal(parameter, torch.ones_like(parameter))
    assert (optimizer.masters[0] < 1).all()
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    resumed_parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    resumed = MasterAdamW([("weight", resumed_parameter)], lr=1e-6, weight_decay=0)
    resumed.load_state_dict(torch.load(buffer, weights_only=True))
    parameter.grad = torch.ones_like(parameter)
    resumed_parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    resumed.step()
    assert torch.equal(optimizer.masters[0], resumed.masters[0])
    assert torch.equal(parameter, resumed_parameter)


def test_padding_cannot_relay_into_valid_tokens():
    layout = TokenLayout(torch.tensor([0, 1, 2, 2]), torch.tensor([-1, 0, 1, 1]), torch.tensor([-1, 0, 0, 0]), True,
                         torch.tensor([True, True, True, False]))
    mask = layout.dense()
    assert not mask[:3, 3].any()
    assert mask[3].sum() == 1 and mask[3, 3]


def test_context_mixing_timestep_and_geometry_match_packed_rows():
    temporal = temporal_layout(77)
    camera = torch.zeros(2, 27, 10)
    camera[..., :2] = 1
    camera[1, :, 7] = 1
    features = dict(latents=torch.ones(2, 24, 27, 2, 2),
                    camera_pose=camera,
                    rotary_frames=temporal.rotary_frames,
                    valid_frames=temporal.valid,
                    fps=16,
                    prompt_embeds=torch.ones(1, 3, 5120))
    inputs, target, mask = teacher_forcing_batch(features, "cpu", sigma=0.3, context_noise_std=0)
    times = inputs["timestep"][inputs["timestep_indices"]]
    assert torch.all(times[:3] == 0.7)
    assert times[3] == 1
    assert torch.all(times[4:3 + 54] == 0.8)
    assert torch.all(times[3 + 54:] == 0.7)
    assert inputs["camera_indices"][3:7].tolist() == [0, 1, 2, 3]
    assert inputs["camera_pose"][0, 1, 7] == 1
    assert int(mask.sum()) == 2 * 23 - 1
    assert inputs["position_ids"][5, 0] == 3 + 40 / 16
    assert target.shape == inputs["hidden_states"].shape
    clean = torch.ones_like(target[:, 54:])
    velocity = target[:, 54:]
    noise = clean - velocity
    noisy = inputs["hidden_states"][:, 54:]
    scheduler = MiniMaxH3Scheduler()
    expected = scheduler.scale_noise(clean, inputs["timestep"][1], noise)
    torch.testing.assert_close(noisy, expected)
    # Native denoising is x0 = x_t + sigma*v. This catches a reversed target sign.
    torch.testing.assert_close(noisy + 0.3 * velocity, clean)
