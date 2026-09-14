"""Pack/unpack H3 video patches while retaining native channel order."""


def patchify(latents):
    views, channels, frames, height, width = latents.shape
    if channels != 24 or height % 2 or width % 2:
        raise ValueError("Expected H3 24-channel latents with even spatial dimensions")
    return (
        latents.reshape(views, channels, frames, height // 2, 2, width // 2, 2)
        .permute(2, 0, 3, 5, 1, 4, 6)
        .reshape(1, -1, 96)
    )


def unpatchify(tokens, shape):
    """Restore the latent volume from H3's 2x2 spatial patch tokens."""
    _, channels, frames, height, width = shape
    return (
        tokens.reshape(frames, height // 2, width // 2, channels, 2, 2)
        .permute(3, 0, 1, 4, 2, 5)
        .reshape(shape)
    )
