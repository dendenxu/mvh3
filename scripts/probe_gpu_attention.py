#!/usr/bin/env python3
"""Bounded CUDA kernel preflight before loading the full pretrained stack."""

import runtime_env  # noqa: F401
import argparse
import json
from pathlib import Path

import torch

from h3.modules.camera import precompute_camera
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout
from h3.modules.model import MiniMaxH3TransformerBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fa4", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    block = MiniMaxH3TransformerBlock(128, 4, 128, 256, 64, 1e-5, 1e-5).to(args.device, torch.bfloat16)
    block.attn.processor._attention_backend = "flex"
    block.attn.processor.fa4 = args.fa4
    size = 512
    hidden = torch.randn(1, size, 128, device=args.device, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.zeros(size, device=args.device, dtype=torch.long)
    kinds = torch.full_like(indices, NOISY)
    chunks = torch.arange(size, device=args.device) % 256 // 64
    kinds[:256] = CLEAN
    kinds[:16], chunks[:16] = CONDITION, -1
    active = torch.ones(size, device=args.device, dtype=torch.bool)
    active[-3:] = False
    layout = TokenLayout(kinds, chunks, indices, active=active,
                         history_dropout=torch.eye(4, device=args.device, dtype=torch.bool))
    pose = torch.zeros(1, 2, 10, device=args.device)
    pose[..., :2] = 1
    pose[:, 1, 7] = 0.1
    camera = precompute_camera(pose)
    camera_indices = torch.arange(size, device=args.device) % 2
    camera_indices[:16] = -1
    rope = (torch.ones(size, 96, device=args.device), torch.zeros(size, 96, device=args.device))
    mask = layout.block_mask()
    print("CUDA block mask constructed", flush=True)
    temb = torch.randn(1, 64, device=args.device, dtype=torch.bfloat16)
    output = block(hidden, temb, indices, rope, mask, camera, camera_indices)
    sparse_grad = torch.autograd.grad(output.float().square().mean(), (hidden, block.attn.to_q.weight))
    dense = block(hidden, temb, indices, rope, layout.dense(), camera, camera_indices)
    dense_grad = torch.autograd.grad(dense.float().square().mean(), (hidden, block.attn.to_q.weight))
    assert torch.isfinite(output).all() and all(torch.isfinite(value).all() for value in sparse_grad)
    torch.testing.assert_close(output, dense, atol=0.02, rtol=0.02)
    grad_relative = [((a.float() - b.float()).norm() / b.float().norm()).item()
                     for a, b in zip(sparse_grad, dense_grad)]
    assert max(grad_relative) < 0.03, grad_relative
    # A future CLEAN chunk must not affect earlier NOISY predictions through any layer.
    with torch.no_grad():
        reference, changed = hidden.detach().clone(), hidden.detach().clone()
        changed[:, (kinds == CLEAN) & (chunks == 3)] += 10
        for _ in range(3):
            reference = block(reference, temb, indices, rope, mask, camera, camera_indices)
            changed = block(changed, temb, indices, rope, mask, camera, camera_indices)
        earlier = (kinds == NOISY) & (chunks < 3)
        assert torch.equal(reference[:, earlier], changed[:, earlier])
        assert not torch.equal(reference, changed)
    report = {
        "status": "passed",
        "device": args.device,
        "dtype": "bfloat16",
        "tokens": size,
        "dense_sparse_max_abs": (output.float() - dense.float()).abs().max().item(),
        "gradient_relative_errors": grad_relative,
        "three_layer_future_invariance": "exact"
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
