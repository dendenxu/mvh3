"""Stage-1 splitting preserves all views and every requested frame."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MonoWindow:
    view: int
    start: int
    stop: int

    @property
    def frame_count(self):
        return self.stop - self.start


def short_mono_windows(frame_counts: list[int], max_frames: int = 77) -> list[MonoWindow]:
    """Split each view independently, including the final partial window.

    The WorldGen short maximum is 20 Wan latents = 77 sampled pixel frames.
    This limit is in source frames at 16 FPS, not a claim of 20 H3 latents.
    Tail padding and the H3 encoder's emitted temporal layout are separate.
    """
    if max_frames < 1 or not frame_counts or any(count < 1 for count in frame_counts):
        raise ValueError("Every view and window must have a positive frame count")
    return [
        MonoWindow(view, start, min(start + max_frames, count))
        for view, count in enumerate(frame_counts)
        for start in range(0, count, max_frames)
    ]
