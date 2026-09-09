"""WorldViews color transfer functions used by the image utilities."""

import torch


@torch.jit.script
def linear2srgb(linear: torch.Tensor):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    linear = linear.clip(0, 1)
    tensor_linear = linear * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        (linear + 1e-7) ** (1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = linear <= srgb_linear_thres
    return torch.where(is_linear, tensor_linear, tensor_nonlinear)


@torch.jit.script
def srgb2linear(srgb: torch.Tensor):
    linear_srgb_thres = 0.04045
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    srgb = srgb.clip(0, 1)
    tensor_linear = srgb / srgb_linear_coeff
    tensor_nonlinear = (
        (srgb + srgb_exponential_coeff - 1) / srgb_exponential_coeff
    ) ** srgb_exponent

    is_linear = srgb <= linear_srgb_thres
    return torch.where(is_linear, tensor_linear, tensor_nonlinear)
