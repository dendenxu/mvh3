"""Camera-aligned teacher-forcing batches from actual H3 latent features."""

import math

import torch

from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout


def patchify(latents):
    views, channels, frames, height, width = latents.shape
    if channels != 24 or height % 2 or width % 2:
        raise ValueError("Expected H3 24-channel latents with even spatial dimensions")
    return latents.reshape(views, channels, frames, height // 2, 2, width // 2, 2).permute(2, 0, 3, 5, 1, 4,
                                                                                           6).reshape(1, -1, 96)


def teacher_forcing_batch(features,
                          device,
                          seed=42,
                          sigma=0.5,
                          context_noise=0.2,
                          context_noise_std=0.1,
                          cross_view=True):
    """Single packed document, with synchronized views kept separate spatially.

    The caller chooses one view per document in the short-monocular stage.
    Native rotary time is 40 units/sec, including when training videos are 16 FPS.
    """
    latents = features["latents"].to(device)
    views, _, frames, height, width = latents.shape
    if not cross_view and views != 1:
        raise ValueError("Short mono documents require exactly one view; use the batch axis for more")
    clean = patchify(latents)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    noisy = (1 - sigma) * clean + sigma * noise
    prompt = features["prompt_embeds"].to(device)
    text_count, video_count = prompt.shape[1], clean.shape[1]
    pixels_per_frame = height * width // 4
    frames_for_token = torch.arange(frames, device=device).repeat_interleave(views * pixels_per_frame)
    views_for_token = torch.arange(views, device=device).repeat_interleave(pixels_per_frame).repeat(frames)
    context_levels = (context_noise +
                      context_noise_std * torch.randn(views, generator=generator, device=device)).clamp(0, 1)
    token_context_levels = context_levels[views_for_token][None, :, None]
    context_random = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    context = clean * (1 - token_context_levels) + context_random * token_context_levels
    starts = features["rotary_frames"].to(device)
    camera_ids = frames_for_token * views + views_for_token
    camera_pose = features["camera_pose"].to(device).permute(1, 0, 2).reshape(1, frames * views, 10)
    valid_video = features["valid_frames"].to(device)[frames_for_token]
    token_chunks = (starts[frames_for_token] / 20).floor().long()
    first_condition = (frames_for_token == 0) & (views_for_token == 0)
    context[:, first_condition] = clean[:, first_condition]
    length = text_count + 2 * video_count
    text_indices = torch.arange(text_count, device=device)
    video_indices = torch.arange(text_count, length, device=device)
    audio_indices = torch.empty(0, device=device, dtype=torch.long)
    tags = torch.zeros(length, device=device, dtype=torch.long)
    tags[text_indices] = 1
    timestep_indices = torch.zeros_like(tags)
    timestep_indices[text_indices] = 1  # Native H3 text rows inherit the generated-video timestep.
    timestep_indices[text_count:text_count + video_count] = 2 + views_for_token
    timestep_indices[text_count:text_count + video_count][first_condition] = 0
    timestep_indices[text_count + video_count:] = 1
    kinds = torch.cat((torch.full_like(text_indices, CONDITION), torch.full(
        (video_count, ), CLEAN, device=device), torch.full((video_count, ), NOISY, device=device)))
    chunks = torch.cat((torch.full_like(text_indices, -1), token_chunks, token_chunks))
    scopes = torch.cat((torch.full_like(text_indices, -1), views_for_token, views_for_token))
    kinds[text_count:text_count + video_count][first_condition] = CONDITION
    chunks[text_count:text_count + video_count][first_condition] = -1
    active = torch.cat((torch.ones(text_count, device=device, dtype=torch.bool), valid_video, valid_video))
    layout = TokenLayout(kinds, chunks, scopes, cross_view, active)
    sqrt_area = math.sqrt(height * width)
    h = ((1 - height / sqrt_area) / 2 + torch.arange(height // 2, device=device).double() * (2 / sqrt_area)) * 32
    w = ((1 - width / sqrt_area) / 2 + torch.arange(width // 2, device=device).double() * (2 / sqrt_area)) * 32
    spatial = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).reshape(-1, 2).repeat(frames * views, 1)
    positions_video = torch.cat(((text_count + starts[frames_for_token] * (40 / features["fps"]))[:, None], spatial),
                                dim=-1)
    positions_text = torch.zeros(text_count, 3, device=device, dtype=torch.float64)
    positions_text[:, 0] = text_indices
    positions = torch.cat((positions_text, positions_video, positions_video))
    inputs = dict(
        hidden_states=torch.cat((context, noisy), dim=1),
        audio_hidden_states=clean.new_empty(1, 0, 32),
        # H3 uses t=1 for clean data and predicts data-ward velocity.
        encoder_hidden_states=prompt,
        timestep=torch.cat((torch.tensor([1.0, 1 - sigma], device=device), 1 - context_levels)),
        timestep_indices=timestep_indices,
        token_tags=tags,
        position_ids=positions,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        camera_pose=camera_pose,
        camera_indices=torch.cat((torch.full_like(text_indices, -1), camera_ids, camera_ids)),
        attention_mask=layout,
    )
    loss_mask = torch.cat((torch.zeros_like(valid_video), valid_video & ~first_condition))[None]
    target = torch.cat((torch.zeros_like(clean), clean - noise), dim=1)
    return inputs, target, loss_mask
