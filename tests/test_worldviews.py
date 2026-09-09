import copy
from pathlib import Path

import pytest
import torch

from h3.modules.camera import apply_matrix, camera_projection, matrix_rotary
from utils.config import load_config
from h3.data import temporal_layout
from h3.distributed.fsdp import configure_model, compile_blocks, parameter_groups
from utils.h3_wrapper import source_documents
from h3.modules.masking import CLEAN, NOISY, TokenLayout
from model.diffusion import WorldViewsObjective, chunk_ids
from utils.checkpoint import save_checkpoint, load_checkpoint
from fixtures_h3 import tiny_model


def recipe():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/worldviews.yaml")
    cfg.h3.checkpoint, cfg.h3.vae = "/unused/h3", "/unused/vae"
    cfg.model.fa4 = False
    cfg.attn_block_compile = False
    return cfg


from fixtures_h3 import feature_document


def dense_inputs(inputs):
    inputs = dict(inputs)
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    return inputs


def test_replace_original_alternate_attention_without_new_parameters():
    model, cfg = tiny_model(), recipe()
    before = {n: tuple(p.shape) for n, p in model.named_parameters()}
    configure_model(model, cfg)
    assert before == {n: tuple(p.shape) for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        assert p.requires_grad == (n.startswith("transformer_blocks.") and ".attn." in n
                                   and int(n.split(".")[1]) % 2 == 0)
    assert parameter_groups(model, cfg)[0]["lr"] == 1e-5


def test_matrix_inverse_and_native_temporal_channels():
    torch.manual_seed(10)
    pose = torch.randn(1, 3, 10) * .1
    pose[..., :2] = .8
    matrix = camera_projection(pose)
    x = torch.randn(1, 4, 2, 128)
    ids = torch.tensor([-1, 0, 1, 2])
    transformed = apply_matrix(x, matrix.inverse, ids)
    recovered = apply_matrix(transformed, matrix.projection, ids)
    torch.testing.assert_close(recovered, x, atol=1e-6, rtol=1e-6)
    assert torch.equal(transformed[:, 0], x[:, 0])
    assert torch.equal(transformed[..., :16], x[..., :16])
    assert torch.equal(transformed[..., 48:64], x[..., 48:64])
    assert torch.equal(transformed[..., 96:], x[..., 96:])
    q = apply_matrix(x, matrix.projection.mT, ids)
    k = apply_matrix(x, matrix.inverse, ids)
    torch.testing.assert_close((q * k).sum(-1), x.square().sum(-1))


def test_subframe_camera_groups_preserve_global_head_offsets():
    pose = torch.zeros(1, 3, 4, 10)
    pose[..., :2] = 1
    pose[..., 7] = torch.arange(4) * .3
    matrix = camera_projection(pose.reshape(1, 12, 10)).inverse.reshape(1, 3, 4, 4, 4)
    x = torch.randn(1, 3, 56, 128)
    indices = torch.arange(3)
    whole = apply_matrix(x, matrix, indices, 0, 56)
    pieces = [apply_matrix(part, matrix, indices, i * 7, 56) for i, part in enumerate(x.split(7, dim=2))]
    torch.testing.assert_close(torch.cat(pieces, dim=2), whole, rtol=0, atol=0)
    assert not torch.equal(whole, x)


def test_sparse_attention_cannot_fall_back_to_eager():
    from h3.modules.attention import _flex
    with pytest.raises(RuntimeError, match="requires compilation"):
        _flex(None, None, None, None)


def test_short_mono_keeps_every_view_and_tail():
    f, v, h, w = 81, 2, 4, 4
    k = torch.eye(3).expand(f * v, 3, 3).clone()
    sample = dict(frames=torch.rand(f, 3, h, w * v),
                  mv=v,
                  Ks=k,
                  Rs=k,
                  Ts=torch.zeros(f * v, 3, 1),
                  projs=torch.eye(4).expand(f * v, 4, 4),
                  projs_inv=torch.eye(4).expand(f * v, 4, 4),
                  fps=16,
                  cpu=dict(pack=dict(height=h, width=w, rs=[1, 1], xs=[0, 1], ys=[0, 0]),
                           prompts="caption",
                           pose_stable_factor=1))
    docs = source_documents(sample, 1, 77)
    assert [len(d["views"][0]["pixels"]) for d in docs] == [77, 4, 77, 4]
    assert all(len(d["views"]) == 1 and d["isolated"] for d in docs)


def test_nonuniform_chunks_and_truncated_history():
    cfg, doc = recipe(), feature_document()
    inputs, target, weight, records, _ = WorldViewsObjective(cfg).pack(doc, "cpu")
    layout = inputs["attention_mask"]
    assert chunk_ids(doc["views"][0]["frames"], 5).tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert int((layout.kind == CLEAN).sum()) == 5
    assert int((layout.kind == NOISY).sum()) == 7
    assert weight[:5].eq(0).all() and weight[5:].ge(0).all()
    assert len(records) == 1
    mask = layout.dense()
    clean = torch.nonzero(layout.kind == CLEAN).flatten()
    noisy = torch.nonzero(layout.kind == NOISY).flatten()
    assert not mask[noisy[:5, None], clean].any()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_complete_objective_updates_existing_weights_and_restores(tmp_path, monkeypatch, checkpointing):
    monkeypatch.setenv("MVH3_DATA_ROOT", "/data")
    monkeypatch.setenv("MVH3_DATA_ROOT3", "/data3")
    cfg, doc, model = recipe(), feature_document(views=2), tiny_model()
    cfg.gradient_checkpointing = checkpointing
    configure_model(model, cfg)
    compile_blocks(model, cfg)
    objective = WorldViewsObjective(cfg)
    inputs, target, weights, _, _ = objective.pack(doc, "cpu")
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg),
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)
    before = {n: p.clone() for n, p in model.named_parameters()}
    prediction = model(**dense_inputs(inputs)).sample
    loss = ((prediction - target).square().mean(-1)[0] * weights).sum() / weights.sum()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all(".attn." in n for n in changed)
    path = save_checkpoint(model, optimizer, cfg, 123, 1, {"pending_rf": None}, tmp_path)
    saved = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    state = load_checkpoint(model, optimizer, cfg, path)
    assert state["step"] == 123 and state["stage"] == 1
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, saved[n])


def test_scale_condition_changes_prediction_without_parameters():
    cfg, model = recipe(), tiny_model()
    configure_model(model, cfg)
    inputs, _, _, _, _ = WorldViewsObjective(cfg).pack(feature_document(), "cpu")
    inputs = dense_inputs(inputs)
    embeddings, rotations = [], []
    time_hook = model.time_embedder.register_forward_hook(lambda module, args, output: embeddings.append(output.clone()))
    rope_hook = model.rope.register_forward_hook(lambda module, args, output: rotations.append(output))
    with torch.no_grad():
        scaled = model(**inputs).sample
        inputs["scale_log"] = torch.zeros_like(inputs["scale_log"])
        unit = model(**inputs).sample
    time_hook.remove()
    rope_hook.remove()
    assert not torch.equal(scaled, unit)
    assert len(embeddings) == 2 and torch.equal(embeddings[0], embeddings[1])
    for scaled_rotary, unit_rotary in zip(*rotations):
        assert torch.equal(scaled_rotary[:, :16], unit_rotary[:, :16])
        assert torch.equal(scaled_rotary[:, 48:64], unit_rotary[:, 48:64])
        assert torch.equal(scaled_rotary[inputs["text_indices"]], unit_rotary[inputs["text_indices"]])
        assert scaled_rotary.abs().max() <= 1


def test_history_dropout_removes_only_noisy_history_edges():
    kind = torch.tensor([CLEAN, CLEAN, NOISY, NOISY])
    chunk = torch.tensor([0, 1, 0, 1])
    layout = TokenLayout(kind,
                         chunk,
                         torch.zeros(4, dtype=torch.long),
                         True,
                         history_dropout=torch.ones(2, 2, dtype=torch.bool))
    mask = layout.dense()
    assert mask[1, 0] and not mask[3, 0] and mask[3, 3]


@pytest.mark.parametrize("raise_error", [False, True])
def test_visualization_output_failure_obeys_recipe(monkeypatch, raise_error):
    from trainer.diffusion import Trainer
    from pipeline import ar_inference
    from utils import visualization

    trainer = Trainer.__new__(Trainer)
    trainer.cfg, trainer.device = recipe(), torch.device("cpu")
    trainer.cfg.raise_vis_error = raise_error
    trainer.model = trainer.video = None
    trainer.text = lambda prompts: [torch.zeros(1)]
    trainer.step, trainer.stage = 0, 1
    monkeypatch.setattr(ar_inference, "generate", lambda *args, **kwargs: [])

    def fail_output(*args, **kwargs):
        raise OSError("Video output unavailable")

    monkeypatch.setattr(visualization, "write_visualization", fail_output)
    if raise_error:
        with pytest.raises(RuntimeError, match="Visualization output failed"):
            trainer.visualize({})
    else:
        trainer.visualize({})
