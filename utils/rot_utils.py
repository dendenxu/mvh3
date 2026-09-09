"""
Just like roma, but no dependency on torch
"""

import numpy as np


def rotvec_to_rotmat(rotvec):
    """
    Convert batch of rotation vectors to rotation matrices.
    Shape: (..., 3) -> (..., 3, 3)
    """
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)  # rotation angle = |rotvec|
    # Avoid division by zero for small angles
    theta_sq = theta**2

    # Rodrigues' formula needs sin(t)/t and (1-cos t)/t^2, both 0/0 at t=0.
    # Below `eps` we swap in the leading Taylor terms; above it we use the
    # exact ratios (with `+eps` in the denominator only as a hard NaN guard,
    # since that branch is masked out for tiny theta anyway).
    # sin(t)/t approx 1 - t^2/6
    # (1-cos(t))/t^2 approx 0.5 - t^2/24
    eps = 1e-8  # angle (rad) below which the exact ratios lose precision to cancellation
    mask = (theta < eps).astype(float)  # 1.0 where small-angle Taylor branch applies

    sinc = (1.0 - mask) * (np.sin(theta) / (theta + eps)) + mask * (1.0 - theta_sq / 6.0)
    one_minus_cos = (1.0 - mask) * ((1.0 - np.cos(theta)) / (theta_sq + eps)) + mask * (0.5 - theta_sq / 24.0)

    # Construct skew-symmetric matrices K
    # rotvec shape (..., 3) -> x, y, z
    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack([
        np.stack([zeros, -z, y], axis=-1),
        np.stack([z, zeros, -x], axis=-1),
        np.stack([-y, x, zeros], axis=-1)
    ], axis=-2)

    # R = I + sinc * K + one_minus_cos * K^2
    batch_shape = rotvec.shape[:-1]
    I = np.eye(3).reshape((1,) * len(batch_shape) + (3, 3))

    # K@K (batch matrix multiplication)
    KK = np.matmul(K, K)

    return I + sinc[..., np.newaxis] * K + one_minus_cos[..., np.newaxis] * KK


def rotmat_to_rotvec(rotmat):
    """
    Convert batch of rotation matrices to rotation vectors.
    Shape: (..., 3, 3) -> (..., 3)

    Uses Shepperd's algorithm: R -> unit quaternion -> rotvec. Compared
    to the direct ``rv = skew(R-R^T) * theta/(2 sin theta)`` formula, this
    path has NO ``sin(theta)`` in any denominator and is numerically stable
    everywhere, including at theta ~ pi where the direct formula amplifies
    R-noise by ``theta/(2 sin theta)`` ~ 10^6 and produces illegal
    rotvecs with ``|rv| > pi``.

    Output is guaranteed:
      - ``|rv| <= pi`` (shortest-arc, via ``q_w >= 0`` enforcement)
      - axis is unit (modulo float ulps)

    Note on sequence sign continuity: at theta = pi exactly, ``+n`` and
    ``-n`` are both valid axes for the same R (SO(3) double cover). This
    function picks one deterministically per input matrix, but adjacent
    frames in a near-pi sequence may still land on opposite hemispheres
    due to micro-noise. Pass the resulting (N, 3) sequence through
    ``canonicalize_rotvec_sequence`` to remove that residual sign flip.
    """
    R = np.asarray(rotmat)
    batch = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]

    # --- Step 1: R -> unit quaternion (x, y, z, w) via Shepperd's method.
    # Branch on trace sign; for trace <= 0, branch on largest diagonal
    # element to avoid catastrophic cancellation in sqrt.
    tr = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    q = np.empty((n, 4), dtype=R_flat.dtype)

    pos = tr > 0
    if np.any(pos):
        Rp = R_flat[pos]
        # tr = 4*w^2 - 1  =>  s := 2*sqrt(tr+1) = 4*|w|. The np.maximum(.,1e-12)
        # floor only guards a negative argument from float noise (tr+1>=0 here
        # since this branch requires tr>0), so the sqrt never NaNs.
        s = np.sqrt(np.maximum(tr[pos] + 1.0, 1e-12)) * 2.0
        inv_s = 1.0 / s
        q[pos, 3] = s * 0.25  # w = s/4 = sqrt(tr+1)/2
        q[pos, 0] = (Rp[:, 2, 1] - Rp[:, 1, 2]) * inv_s
        q[pos, 1] = (Rp[:, 0, 2] - Rp[:, 2, 0]) * inv_s
        q[pos, 2] = (Rp[:, 1, 0] - Rp[:, 0, 1]) * inv_s

    neg = ~pos
    if np.any(neg):
        Rn = R_flat[neg]
        d00, d11, d22 = Rn[:, 0, 0], Rn[:, 1, 1], Rn[:, 2, 2]
        # Pick the largest diagonal entry; the matching quaternion component is
        # the numerically largest one, so building it first avoids cancellation.
        # The mixed >=/> comparisons make c0/c1/c2 a mutually-exclusive,
        # exhaustive partition even on ties (c0 wins d00==d11 ties; c1 only
        # fires when strictly greater than d00, c2 is the remainder).
        c0 = (d00 >= d11) & (d00 >= d22)
        c1 = (d11 > d00) & (d11 >= d22)
        c2 = ~(c0 | c1)
        qn = np.empty((Rn.shape[0], 4), dtype=R_flat.dtype)
        if np.any(c0):
            Rc = Rn[c0]
            sc = np.sqrt(np.maximum(1.0 + Rc[:, 0, 0] - Rc[:, 1, 1] - Rc[:, 2, 2], 1e-12)) * 2.0
            inv = 1.0 / sc
            qn[c0, 0] = sc * 0.25
            qn[c0, 1] = (Rc[:, 0, 1] + Rc[:, 1, 0]) * inv
            qn[c0, 2] = (Rc[:, 0, 2] + Rc[:, 2, 0]) * inv
            qn[c0, 3] = (Rc[:, 2, 1] - Rc[:, 1, 2]) * inv
        if np.any(c1):
            Rc = Rn[c1]
            sc = np.sqrt(np.maximum(1.0 + Rc[:, 1, 1] - Rc[:, 0, 0] - Rc[:, 2, 2], 1e-12)) * 2.0
            inv = 1.0 / sc
            qn[c1, 0] = (Rc[:, 0, 1] + Rc[:, 1, 0]) * inv
            qn[c1, 1] = sc * 0.25
            qn[c1, 2] = (Rc[:, 1, 2] + Rc[:, 2, 1]) * inv
            qn[c1, 3] = (Rc[:, 0, 2] - Rc[:, 2, 0]) * inv
        if np.any(c2):
            Rc = Rn[c2]
            sc = np.sqrt(np.maximum(1.0 + Rc[:, 2, 2] - Rc[:, 0, 0] - Rc[:, 1, 1], 1e-12)) * 2.0
            inv = 1.0 / sc
            qn[c2, 0] = (Rc[:, 0, 2] + Rc[:, 2, 0]) * inv
            qn[c2, 1] = (Rc[:, 1, 2] + Rc[:, 2, 1]) * inv
            qn[c2, 2] = sc * 0.25
            qn[c2, 3] = (Rc[:, 1, 0] - Rc[:, 0, 1]) * inv
        q[neg] = qn

    # --- Step 2: unit quaternion -> rotvec.
    # half_angle = atan2(|q_xyz|, |q_w|) is well-defined and in [0, pi/2];
    # so theta = 2 * half_angle is in [0, pi] -- |rv| <= pi by construction.
    # Sign flip via sign(q_w) chooses the shorter rotation when q_w < 0.
    xyz = q[:, :3]
    w = q[:, 3]
    norm_xyz = np.linalg.norm(xyz, axis=-1)
    half_angle = np.arctan2(norm_xyz, np.abs(w))
    # rotvec = (theta/sin(ha)) * xyz with theta = 2*ha, i.e. scale = 2*ha/sin(ha).
    # That ratio is 0/0 at ha=0, so below the threshold we use its Taylor series.
    # 1e-4 rad: there the next dropped term is O(ha^4) ~ 1e-16, at float64 eps,
    # so the 2-term Taylor is already exact to machine precision.
    small = half_angle < 1e-4
    # Nested np.where is purely a NaN/divide-warning guard, NOT extra logic:
    # numpy evaluates BOTH branches of the outer where, so the exact-formula
    # branch must not divide by sin(0) in the small-angle lanes. The innermost
    # `where(small, 1.0, sin)` forces those lanes' denominator to 1.0, and the
    # middle `where(small, 1.0, ...)` overwrites their (now-finite) result with
    # 1.0 -- both are discarded by the outer where, which keeps the Taylor value.
    scale = np.where(
        small,
        2.0 + (half_angle ** 2) / 3.0,   # 2*ha / sin(ha) ~ 2 + ha^2/3 + O(ha^4)
        np.where(small, 1.0, (2.0 * half_angle) / np.where(small, 1.0, np.sin(half_angle))),
    )
    # q and -q encode the same rotation (double cover); flipping by sign(q_w)
    # is equivalent to having forced q_w >= 0, yielding the shortest-arc rotvec.
    sign = np.where(w < 0, -1.0, 1.0)
    rv_flat = xyz * (scale * sign)[:, None]
    return rv_flat.reshape(*batch, 3)


def canonicalize_rotvec_sequence(rotvec, axis=-2, pi_tol=1e-3):
    """
    Make a sequence of rotvecs sign-continuous along `axis` by flipping
    frames whose dot with the previous frame is negative AND whose
    magnitude is within ``pi_tol`` of pi. The double-cover ambiguity
    (+theta n ≡ -theta n for the same rotation) is only geometrically
    safe at theta = pi exactly; off-pi frames must NOT be flipped or
    the underlying rotation changes (the flipped rep then describes
    the inverse rotation about the opposite axis).

    ``pi_tol`` (rad) is the half-width of the near-pi band that is
    eligible for flipping; 1e-3 keeps it tight enough that genuinely
    off-pi frames (where +n and -n are different rotations) are never
    touched, while still catching the micro-noise jitter that straddles
    pi. Only the *current* frame's norm is range-checked: a flip just
    negates the vector, so its magnitude (and thus the rotation) is
    unchanged -- the previous frame need not also be near pi.

    Shape: (..., N, 3) along default axis=-2. Returns a new array.
    """
    # .copy() so we never mutate the caller's array; moveaxis alone returns a view.
    rv = np.moveaxis(rotvec, axis, -2).copy()
    N = rv.shape[-2]
    norms = np.linalg.norm(rv, axis=-1)
    # Sequential (not vectorized) over frames on purpose: a flip at i-1 is
    # written back into rv before frame i is compared against it, so the
    # canonical sign propagates forward along the whole run. norms is computed
    # once up front because flipping never changes a vector's magnitude.
    for i in range(1, N):
        dots = np.sum(rv[..., i - 1, :] * rv[..., i, :], axis=-1)
        near_pi = np.abs(norms[..., i] - np.pi) < pi_tol
        flip = (dots < 0) & near_pi
        rv[..., i, :] = np.where(flip[..., None], -rv[..., i, :], rv[..., i, :])
    return np.moveaxis(rv, -2, axis)
