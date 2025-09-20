import pytest

try:
    import torch as th
except ModuleNotFoundError as exc:  # pragma: no cover - torch is required for the policy itself
    pytest.skip("requires torch", allow_module_level=True)

from openai_vpt.lib.memory_cull import MemoryCullStrategy


def _diffs(indices):
    return [b - a for a, b in zip(indices[:-1], indices[1:])]


def test_select_indices_multi_scale_pattern():
    strategy = MemoryCullStrategy(keep_len=128, segments=[(32, 1), (32, 3), (32, 9), (32, 27)])
    indices = strategy.select_indices(total_len=1500, new_len=1)
    assert len(indices) == 128
    assert indices == sorted(indices)
    assert indices[-1] == 1499

    recent = indices[-33:-1]
    assert len(recent) == 32
    assert _diffs(recent) == [1] * 31

    medium = indices[-65:-33]
    assert len(medium) == 32
    assert _diffs(medium) == [3] * 31

    long = indices[-97:-65]
    assert len(long) == 32
    assert _diffs(long) == [9] * 31

    longest = indices[:32]
    assert len(longest) == 32
    assert _diffs(longest) == [27] * 31


def test_select_indices_handles_short_history():
    strategy = MemoryCullStrategy(keep_len=8, segments=[(4, 1), (4, 4)])
    indices = strategy.select_indices(total_len=3, new_len=1)
    assert len(indices) == 8
    assert indices[-1] == 2
    # The oldest slots are padded with zeros when not enough history exists.
    assert indices.count(0) >= 2


def test_select_indices_large_new_chunk():
    strategy = MemoryCullStrategy(keep_len=4, segments=[(2, 1), (2, 2)])
    indices = strategy.select_indices(total_len=10, new_len=6)
    assert indices == [6, 7, 8, 9]


def test_update_mask_matches_indices():
    strategy = MemoryCullStrategy(keep_len=4, segments=[(2, 1), (2, 2)])
    prev_mask = th.tensor([[[True, True, False, False]]])
    indices = th.tensor([0, 1, 2, 3])
    first = th.tensor([[False]])
    new_mask = strategy.update_mask(prev_mask, new_len=1, indices=indices, first=first)
    assert new_mask.shape == (1, 1, 4)
    # The last slot should be marked as coming from the new chunk.
    assert new_mask[0, 0, -1]
