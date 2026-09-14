import torch
import numpy as np
from PIL import Image
from test_native_flow import native_recipe

from h3.data import temporal_layout
from h3.encoders import VideoEncoder
from pipeline.i2v_input import prepare_request
from h3.modules.camera import camera_projection


class PosteriorFixture(VideoEncoder):

    def __init__(self):
        # Exercise a GPU-resident VAE contract without requiring a test GPU.
        self.device = torch.device("cuda:7")

    def encode(self, pixels, generator=None):
        if generator is not None:
            assert generator.device.type == "cpu"
        layout = temporal_layout(len(pixels))
        latent = torch.randn(1, 24, len(layout.valid), 2, 2, generator=generator)
        return latent, layout, torch.ones(1, 1)


class TextFixture:

    def i2v(self, requests):
        return [
            dict(features=torch.zeros(1, 2, 5120), tags=torch.ones(2, dtype=torch.long)) for _ in requests
        ]


def test_condition_posterior_uses_native_cpu_seed_in_training_and_inference(tmp_path):
    cfg = native_recipe()
    cfg.cond_image_dropout_ratio = 0
    cfg.cond_image_max_frames = cfg.max_cond_latent = 1
    pose = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat(17, 1)
    geometry = camera_projection(pose[None])
    pixels = torch.zeros(17, 3, 32, 32)
    raw = dict(
        views=[
            dict(
                pixels=pixels,
                pose=pose,
                projection=geometry.projection[0],
                inverse=geometry.inverse[0],
                fps=24,
                scale=1.0,
                prompt="A room",
                source_view=0,
                source_start=0,
            )
        ],
        isolated=True,
    )
    video = PosteriorFixture()
    train = video.prepare(raw, cfg)["views"][0]["condition"]["latent"]
    Image.new("RGB", (32, 32)).save(tmp_path / "input.png")
    np.save(tmp_path / "camera.npy", pose.numpy())
    request = dict(prompt="A room", views=[dict(image="input.png", camera="camera.npy")])
    inference = prepare_request(request, tmp_path, video, TextFixture(), cfg)["views"][0]["condition"][
        "latent"
    ]
    expected = torch.randn(1, 24, 1, 2, 2, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(train, expected, rtol=0, atol=0)
    torch.testing.assert_close(inference, expected, rtol=0, atol=0)
