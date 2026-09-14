#!/usr/bin/env python3
"""Stress compiled SM90 sparse backward against SDPA, including allocator reuse."""

import runtime_env  # noqa: F401
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from h3.modules.attention import compiled_flex_attention
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 3560, 4096])
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    rows = []
    for size in args.tokens:
        torch.manual_seed(321 + size)
        index = torch.arange(size, device=args.device)
        kind = torch.where(index < size // 2, CLEAN, NOISY)
        chunk = (index % (size // 2)) // max(1, size // 8)
        scope = (index // max(1, size // 4)) % 2
        kind[:17], chunk[:17], scope[:17] = CONDITION, -1, -1
        active = index < size - 3
        layout = TokenLayout(kind, chunk, scope, cross_view=False, active=active,
                             history_dropout=torch.eye(5, device=args.device, dtype=torch.bool))
        mask, dense = layout.block_mask(), layout.dense(max_tokens=max(args.tokens))
        inputs = tuple(torch.randn(1, 7, size, 128, device=args.device, dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(3))
        upstream = torch.randn_like(inputs[0]) / size
        with sdpa_kernel(SDPBackend.MATH):
            reference_inputs = tuple(value.detach().float().requires_grad_(True) for value in inputs)
            reference_output = torch.nn.functional.scaled_dot_product_attention(*reference_inputs, attn_mask=dense)
            reference_gradients = torch.autograd.grad(reference_output, reference_inputs, upstream.float())
        for repeat in range(args.repeats):
            # Reuse dirty memory between repetitions; unwritten buffers must not
            # silently pass because a fresh allocation happened to contain zero.
            scratch = torch.empty(1, 7, ((size + 63) // 64) * 64, device=args.device)
            scratch.fill_(float("nan"))
            del scratch
            output = compiled_flex_attention(*inputs, block_mask=mask, kernel_options={"BACKEND": "FLASH"})
            gradients = torch.autograd.grad(output, inputs, upstream)
            relative = [float((value.float() - ref).norm() / ref.norm())
                        for value, ref in zip(gradients, reference_gradients)]
            row = dict(tokens=size, repeat=repeat, finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                       output_relative_l2=float((output.float() - reference_output).norm() / reference_output.norm()),
                       gradient_relative_l2=relative)
            rows.append(row)
            assert row["finite"] and max(relative) < 0.01 and row["output_relative_l2"] < 0.01, row
    torch.cuda.synchronize()
    report = dict(status="passed", scope="7-head BF16 sparse attention against identical-input FP32 SDPA",
                  repeats=rows, cpu_synchronization_inside_attention=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
