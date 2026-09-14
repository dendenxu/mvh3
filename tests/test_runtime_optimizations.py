from types import SimpleNamespace

import torch

from h3.encoders import TextEncoder
from h3.modules.masking import CLEAN, TokenLayout
from h3.modules.kv_cache import make_caches, HistoryCache


class Tokens(dict):

    def to(self, device):
        return Tokens({key: value.to(device) for key, value in self.items()})

    @property
    def input_ids(self):
        return self["input_ids"]


class Tokenizer:

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(letter) for letter in text]}

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 1, "<|image_pad|>": 2, "<|vision_end|>": 3}[token]

    def pad(self, data, **kwargs):
        rows = data["input_ids"]
        ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return Tokens(input_ids=ids, attention_mask=mask)


class ImageProcessor:
    merge_size = 2

    def __call__(self, images, **kwargs):
        return dict(
            image_grid_thw=torch.tensor([[1, 2, 2]] * len(images)), pixel_values=torch.ones(len(images), 4)
        )


def test_i2v_deduplicates_conditions_and_preserves_cached_collective_participant(tmp_path, monkeypatch):
    encoder = TextEncoder.__new__(TextEncoder)
    encoder.identity, encoder.cache, encoder.device = "fixture", tmp_path, torch.device("cpu")
    encoder.tokenizer = Tokenizer()
    encoder.processor = SimpleNamespace(
        image_processor=ImageProcessor(), create_mm_token_type_ids=lambda ids: [[0] * len(row) for row in ids]
    )
    encoder.ensure_encoder = lambda: None
    batches = []

    def forward(**inputs):
        batches.append(inputs["input_ids"].shape[0])
        value = inputs["input_ids"].float().unsqueeze(-1).expand(-1, -1, 5120)
        return SimpleNamespace(hidden_states=[None] * 50 + [value])

    encoder.encoder = forward
    picture = torch.zeros(3, 32, 32, dtype=torch.uint8)
    requests = [(caption, [picture]) for caption in ("alpha", "alpha", "beta", "alpha")]
    results = encoder.i2v(requests)
    assert batches == [2]
    assert encoder.last_i2v_stats["unique_requests"] == 2
    assert torch.equal(results[0]["features"], results[1]["features"])
    assert not torch.equal(results[0]["features"], results[2]["features"])
    for original, cached in zip(results, encoder.i2v(requests)):
        torch.testing.assert_close(original["features"], cached["features"], rtol=0, atol=0)
    assert batches == [2]
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.pt")}

    # A remote miss still requires one local FSDP forward, but cannot rewrite
    # the local cache or replace its already validated feature tensors.
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda value, **kwargs: value.fill_(1))
    monkeypatch.setattr(
        dist, "all_gather_object", lambda output, value: output.__setitem__(slice(None), [value, []])
    )
    for original, cached in zip(results, encoder.i2v(requests)):
        torch.testing.assert_close(original["features"], cached["features"], rtol=0, atol=0)
    assert batches == [2, 1]
    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.pt")}


def test_kv_budget_splits_at_token_boundary_and_preserves_values(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    key = torch.arange(16, dtype=torch.float32).reshape(1, 8, 1, 2)
    value = key + 10
    layout = TokenLayout(
        torch.full((8,), CLEAN),
        torch.arange(8),
        torch.zeros(8, dtype=torch.long),
        True,
        torch.ones(8, dtype=torch.bool),
    )
    cache = HistoryCache(offload=True, budget_bytes=3 * 2 * 2 * 4)
    cache.segments = [(key, value, layout)]
    cache.place_history()
    assert [entry[0].shape[1] for entry in cache.segments] == [5, 3]
    assert cache.segments[-1][0].untyped_storage().nbytes() == 3 * 2 * 4
    torch.testing.assert_close(torch.cat([entry[0] for entry in cache.segments], 1), key, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([entry[1] for entry in cache.segments], 1), value, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([entry[2].chunk for entry in cache.segments]), layout.chunk, rtol=0, atol=0
    )


def test_cfg_distilled_cache_uses_single_stream_budget():
    model = SimpleNamespace(transformer_blocks=[None] * 50)
    cfg = SimpleNamespace(
        kv_gpu_budget_gb=20,
        kv_offload=True,
        guidance_scale=1,
        kv_sink_size=0,
        kv_window_size=0,
        chunk_size=5,
        kv_sink_view0_only=False,
    )
    distilled = make_caches(model, cfg)
    cfg.guidance_scale = 5
    guided = make_caches(model, cfg)
    assert abs(sum(cache.budget_bytes for cache in distilled) - 20 * 1024**3) < 50
    assert all(abs(a.budget_bytes - 2 * b.budget_bytes) <= 1 for a, b in zip(distilled, guided))


def test_resampling_forcing_warmup_skips_history_copy_and_starts_on_absolute_boundary():
    from test_worldviews import recipe
    from fixtures_h3 import feature_document

    from model.diffusion import DiffusionObjective

    cfg, document = recipe(), feature_document()
    cfg.resampling_forcing = True
    cfg.resampling_forcing_warmup_steps = 12000
    objective = DiffusionObjective(cfg)
    objective.sample_sigmas = lambda count, device: (torch.full((count,), 0.5), torch.ones(count), False)

    def zero_velocity(**inputs):
        return SimpleNamespace(sample=torch.zeros_like(inputs["hidden_states"]))

    torch.manual_seed(124)
    before, cold = objective.compute_loss(zero_velocity, document, "cpu", 11999)
    torch.manual_seed(124)
    after, warm = objective.compute_loss(zero_velocity, document, "cpu", 12000)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert cold["resampling_forcing"] is False and cold["x0"] is None
    assert warm["resampling_forcing"] is True and len(warm["x0"]) == len(document["views"])
    assert cold["sigma"] == cold["sigma_min"] == cold["sigma_max"] == 0.5
    assert all(value.shape == view["latent"].shape for value, view in zip(warm["x0"], document["views"]))


def test_tracking_keeps_text_out_of_numeric_history_and_preserves_global_step():
    from utils.tracking import Tracker

    calls = []
    tracker = Tracker.__new__(Tracker)
    tracker.step_offset = 7
    tracker.run = SimpleNamespace(summary={}, log=lambda values, **kwargs: calls.append((values, kwargs)))
    tracker.log(
        {
            "loss": 0.25,
            "train/high": True,
            "data/source": "/source.parquet",
            "recipe": {"camera": "decomposed"},
        },
        128,
    )
    assert calls == [({"step": 128, "global_step": 128, "loss": 0.25, "train/high": 1.0}, {"step": 135})]
    assert tracker.run.summary == {"data/source": "/source.parquet", "recipe": {"camera": "decomposed"}}
