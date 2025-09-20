from typing import List

import torch as th

from openai_vpt.lib import xf
from openai_vpt.lib.memory_cull import MemoryCullStrategy, MemoryTier


def _gather(full: th.Tensor, indices: th.Tensor, detach: bool = False) -> th.Tensor:
    if indices.numel():
        gathered = full.index_select(1, indices)
    else:
        gathered = full[:, :0]
    return gathered.detach() if detach else gathered


def test_memory_cull_respects_real_time_strides():
    attn = xf.All2All(nhead=1, maxlen=8, mask=True)
    layer = xf.SelfAttentionLayer(x_size=4, attn=attn, scale=1.0, cache_keep_len=32)

    batch_size = 1
    state = layer.initial_state(batch_size, initial_T=0)

    tiers = [
        MemoryTier(stride=1, max_keep=3),
        MemoryTier(stride=3, max_keep=3),
        MemoryTier(stride=6, max_keep=2),
    ]
    strategy = MemoryCullStrategy(tiers)

    expected_step = -1
    resets = {0, 11}

    for step in range(24):
        is_reset = step in resets
        first_flag = th.tensor([[is_reset]], dtype=th.bool)
        if is_reset:
            expected_step = 0
        else:
            expected_step += 1

        key = th.full((batch_size, 1, layer.x_size), float(step))
        value = th.full((batch_size, 1, layer.x_size), float(step))

        state, full_k, full_v, full_steps = layer.update_state(state, key, value, first=first_flag)

        assert full_steps.shape == (batch_size, full_k.shape[1])
        assert full_steps[0, -1].item() == expected_step
        if is_reset and full_steps.shape[1] > 1:
            # Previous episode history should be invalidated.
            assert th.all(full_steps[0, :-1] < 0)

        selection = strategy.select_indices(full_k, full_steps)[0]
        device = selection.indices.device

        # Union of tier selections should match the overall index list.
        concatenated = (
            th.cat([idx for idx in selection.per_tier if idx.numel() > 0])
            if selection.per_tier
            else th.empty(0, dtype=th.long, device=device)
        )
        if concatenated.numel():
            assert th.equal(th.unique(concatenated, sorted=True), selection.indices)
        else:
            assert selection.indices.numel() == 0

        for tier, idx_tensor in zip(strategy.tiers, selection.per_tier):
            assert idx_tensor.numel() <= tier.max_keep
            if idx_tensor.numel() == 0:
                continue
            tier_steps = full_steps[0].index_select(0, idx_tensor)
            assert bool(th.all(tier_steps[:-1] <= tier_steps[1:]))
            if tier_steps.numel() > 1:
                diffs = tier_steps[1:] - tier_steps[:-1]
                assert bool(th.all(diffs >= tier.stride))

        state = (
            _gather(full_k, selection.indices, detach=True),
            _gather(full_v, selection.indices, detach=True),
            _gather(full_steps, selection.indices),
        )

        # Ensure state steps mirror the gathered indices exactly.
        if selection.indices.numel():
            assert th.equal(state[2], full_steps.index_select(1, selection.indices))
        else:
            assert state[2].numel() == 0


def test_memory_cull_populates_all_tiers_after_warmup():
    tiers = [
        MemoryTier(stride=1, max_keep=32),
        MemoryTier(stride=3, max_keep=32),
        MemoryTier(stride=9, max_keep=32),
        MemoryTier(stride=27, max_keep=32),
    ]
    strategy = MemoryCullStrategy(tiers)

    cached_steps: List[int] = []
    selection = None
    total_steps = 2600
    snapshot_step = 2000
    snapshot_mid = None
    first_full_step = [None] * len(tiers)

    final_tier_steps: List[List[int]] = []

    for step in range(total_steps):
        full_steps = cached_steps + [step]
        selection = strategy.select_indices(full=None, step_indices=[full_steps])[0]
        keep = selection.indices.tolist()

        tier_step_lists = [
            [full_steps[i] for i in tier_indices.tolist()]
            for tier_indices in selection.per_tier
        ]

        for idx, (tier, steps_list) in enumerate(zip(tiers, tier_step_lists)):
            if len(steps_list) == tier.max_keep and first_full_step[idx] is None:
                first_full_step[idx] = step

        if step == snapshot_step:
            snapshot_mid = [steps[-1] if steps else -1 for steps in tier_step_lists]

        cached_steps = [full_steps[i] for i in keep]
        final_tier_steps = tier_step_lists

    assert selection is not None
    assert snapshot_mid is not None

    total_kept = 0
    for tier, step_list in zip(tiers, final_tier_steps):
        assert len(step_list) == tier.max_keep
        total_kept += tier.max_keep

    assert len(cached_steps) == total_kept

    for earlier, later in zip(first_full_step, first_full_step[1:]):
        assert earlier is not None and later is not None
        assert earlier < later

    final_latest = [steps[-1] if steps else -1 for steps in final_tier_steps]
    for mid, end in zip(snapshot_mid, final_latest):
        assert end > mid
