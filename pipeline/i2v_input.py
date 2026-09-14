"""Build an inference document from images, captions and calibrated camera paths."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from h3.data import temporal_layout
from h3.modules.camera import camera_projection
from model.chunks import prepare_chunk_plan
from utils.camera import prepare_camera_geometry
from utils.captions import caption_specs


def prepare_request(request, root, video, text, cfg, chunked=None):
    root = Path(root)
    chunked = cfg.h3.get("single_sequence", False) and chunked is not False
    fps = float(request.get("fps", 24))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Request FPS must be positive and finite")
    if fps != 24 and not chunked:
        raise ValueError("Released H3 i2v requests use 24 FPS")
    specifications = request["views"]
    if not specifications or not specifications[0].get("image"):
        raise ValueError("The first view needs a conditioning image")
    prompts = [v.get("prompt", request.get("prompt", "")) for v in specifications]
    if not all(prompts):
        raise ValueError("Each view needs a nonempty caption")
    images = [
        (
            torch.from_numpy(np.array(Image.open(root / v["image"]).convert("RGB"))).permute(2, 0, 1)
            if v.get("image")
            else None
        )
        for v in specifications
    ]
    conditions = [img for img in images if img is not None]
    encoded = None if chunked else text.i2v([(prompt, conditions) for prompt in prompts])
    views = []
    for index, (spec, prompt) in enumerate(zip(specifications, prompts)):
        camera = torch.from_numpy(np.load(root / spec["camera"], allow_pickle=False)).float()
        if (
            camera.ndim != 2
            or camera.shape[1] != 10
            or not torch.isfinite(camera).all()
            or not (camera[:, :2] > 0).all()
        ):
            raise ValueError("Camera arrays must be finite [frames, 10] normalized-intrinsic c2w poses")
        frames = len(camera)
        image = None
        if spec.get("image"):
            image = images[index]
            h, w = image.shape[-2:]
        else:
            h, w = int(spec["height"]), int(spec["width"])
        if min(h, w) < 32:
            raise ValueError("Images must be at least 32 x 32; intrinsics refer to their actual size")
        layout = temporal_layout(frames)
        matrices = camera_projection(camera[None])
        starts = layout.rotary_frames.long()
        subframes = (
            (starts[:, None] + torch.arange(4)).minimum(layout.camera_frames[:, None]).clamp_max(frames - 1)
        )
        projection = matrices.projection[0][subframes]
        inverse = matrices.inverse[0][subframes]
        condition = None
        if image is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.h3.condition_encode_seed))
            latent, condition_layout, _ = video.encode(image[None].float() / 255, generator)
            condition = dict(
                latent=latent,
                frames=condition_layout.rotary_frames,
                valid=condition_layout.valid,
                pose=camera[:1],
                projection=projection[:1],
                inverse=inverse[:1],
            )
        weights = F.pad(torch.ones(1, 1, h, w), (0, (-w) % 32, 0, (-h) % 32))
        weights = F.avg_pool2d(weights, 32, 32)[0, 0]
        # Only shape is supplied for the target. No future image/video is loaded or encoded.
        target = torch.zeros(1, 24, len(layout.valid), (h + 31) // 32 * 2, (w + 31) // 32 * 2)
        views.append(
            dict(
                latent=target,
                condition=condition,
                pose=camera[layout.camera_frames],
                projection=projection,
                inverse=inverse,
                frames=layout.rotary_frames,
                valid=layout.valid,
                spatial_weights=weights,
                fps=fps,
                scale=float(spec.get("scale", 1)),
                prompt=prompt,
                height=h,
                width=w,
                source_frames=frames,
                source_view=index,
                source_start=0,
            )
        )
        if encoded is not None:
            views[-1].update(text=encoded[index]["features"], text_tags=encoded[index]["tags"])
        if chunked:
            views[-1].update(
                caption_scene=spec.get("scene", request.get("scene", prompt)),
                caption_motions=spec.get("chunks", request.get("chunks")),
                caption_source_frames=frames,
            )
    document = dict(views=views, isolated=len(views) == 1, source="image-camera-request")
    if chunked:
        document = prepare_chunk_plan(document, cfg, video.device)
        pairs = [caption_specs(view, cfg) for view in document["views"]]
        encoded = iter(text.i2v([(caption, conditions) for specs in pairs for _, caption in specs]))
        for view, specs in zip(document["views"], pairs):
            values = [(chunk, next(encoded)) for chunk, _ in specs]
            view.update(
                texts=[(chunk, value["features"]) for chunk, value in values],
                text=values[0][1]["features"],
                texts_by_bd=True,
                caption_specs=specs,
                text_tag_specs={chunk: value["tags"] for chunk, value in values},
                text_tags=values[0][1]["tags"],
            )
    return prepare_camera_geometry(document, cfg)
