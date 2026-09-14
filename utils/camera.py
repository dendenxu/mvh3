"""Camera units shared by live data, cached features and image-only inference."""

import math


def prepare_camera_geometry(document, cfg):
    """Undo recorded PSF when the recipe uses unscaled camera translation.

    Cached image/text/latent features do not depend on camera units. Restore
    only camera centers and the independent projection/inverse translations,
    including the conditioning image, without mutating the cached document.
    """
    if list(cfg.dataset.pose_stable_factors) != [1.0]:
        return document
    views = []
    changed = False
    for view in document["views"]:
        scale = float(view.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Camera PSF must be finite and positive")
        if scale == 1.0:
            views.append(view)
            continue

        def restore(geometry):
            result = dict(geometry)
            result["pose"] = geometry["pose"].clone()
            result["pose"][..., 7:10] *= scale
            for key in ("projection", "inverse"):
                result[key] = geometry[key].clone()
                result[key][..., :3, 3] *= scale
            return result

        restored = restore(view)
        if view.get("condition") is not None:
            restored["condition"] = restore(view["condition"])
        restored.update(scale=1.0, source_pose_stable_factor=scale)
        views.append(restored)
        changed = True
    return {**document, "views": views} if changed else document
