"""Utilities for compressing the transformer memory cache.

The stock VPT transformer keeps a fixed-length FIFO buffer of the last
``attention_memory_size - timesteps`` key/value pairs.  ``MemoryCullStrategy``
implements an alternative policy where the cache keeps a hierarchical view of
past observations: the latest few steps are stored densely, while progressively
older steps are stored with an increasing stride.  This allows the 128 cache
slots used by VPT to cover a much longer temporal window without changing the
overall tensor shapes expected by the transformer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is required at runtime but not for pure index logic
    import torch as th
except ModuleNotFoundError:  # pragma: no cover
    th = None


@dataclass
class MemoryCullConfig:
    """Configuration describing how the cache should be thinned.

    ``segments`` is an iterable of ``(length, stride)`` pairs.  The strategy
    will allocate ``length`` cache slots to each segment and sample historical
    frames using the given ``stride``.  For example ``(32, 1)`` keeps the latest
    32 frames densely, whereas ``(32, 9)`` keeps 32 frames spaced nine steps
    apart.
    """

    segments: Sequence[Tuple[int, int]]


class MemoryCullStrategy:
    """Selects which cache entries to keep after appending new keys/values."""

    def __init__(self, keep_len: int, segments: Sequence[Tuple[int, int]]):
        if keep_len <= 0:
            raise ValueError("keep_len must be positive")
        if not segments:
            segments = [(keep_len, 1)]
        self.keep_len = keep_len
        self.offsets = self._build_offsets(segments)

    def _build_offsets(self, segments: Sequence[Tuple[int, int]]) -> List[int]:
        """Construct a list of offsets (oldest -> newest)."""
        offsets: List[int] = []
        cursor = 0
        for length, stride in segments:
            if length <= 0:
                continue
            stride = max(stride, 1)
            for _ in range(length):
                offsets.append(-cursor)
                cursor += stride
                if len(offsets) >= self.keep_len:
                    break
            if len(offsets) >= self.keep_len:
                break
        # If the schedule did not provide enough slots, continue stepping with
        # the last stride so that we always return exactly ``keep_len`` offsets.
        tail_stride = segments[-1][1] if segments else 1
        tail_stride = max(tail_stride, 1)
        while len(offsets) < self.keep_len:
            offsets.append(-cursor)
            cursor += tail_stride
        # ``offsets`` currently goes newest -> oldest; reverse to chronological.
        offsets.sort()
        return offsets

    def select_indices(self, total_len: int, new_len: int) -> List[int]:
        """Return indices (ascending) to keep from a concatenated cache.

        ``total_len`` is the length of ``[prev_cache, new_cache]``.  ``new_len``
        is the length of the freshly appended segment.  The result always
        contains ``self.keep_len`` indices (with repeated zeros if there is not
        enough history).
        """
        if total_len <= 0:
            return [0] * self.keep_len

        new_len = max(0, min(new_len, total_len))
        keep = min(self.keep_len, total_len)

        # If the full context fits inside the buffer, keep everything and pad
        # with the oldest element to maintain a fixed length.
        if keep == total_len:
            indices = list(range(total_len))
            if keep < self.keep_len:
                pad = [0] * (self.keep_len - keep)
                indices = pad + indices
            return indices

        # If the new block already fills or exceeds the cache, only the most
        # recent ``keep`` frames can be stored.
        if new_len >= keep:
            indices = list(range(total_len - keep, total_len))
            return indices

        cutoff = total_len - new_len  # index where the new block starts
        old_needed = keep - new_len
        base = total_len - 1
        old_indices: List[int] = []

        for offset in self.offsets:
            idx = base + offset
            if idx < 0:
                idx = 0
            if idx >= cutoff:
                continue
            if not old_indices or idx != old_indices[-1]:
                old_indices.append(idx)
            if len(old_indices) == old_needed:
                break

        if len(old_indices) < old_needed:
            missing = old_needed - len(old_indices)
            start = max(cutoff - missing, 0)
            filler = list(range(start, cutoff))
            for idx in filler:
                if not old_indices or idx != old_indices[-1]:
                    old_indices.append(idx)
                if len(old_indices) == old_needed:
                    break
            old_indices = old_indices[-old_needed:]

        old_indices = sorted(old_indices)
        tail = list(range(cutoff, total_len))
        result = old_indices + tail

        if len(result) < keep:
            pad = [0] * (keep - len(result))
            result = pad + result
        elif len(result) > keep:
            result = result[-keep:]

        if keep < self.keep_len:
            pad = [0] * (self.keep_len - keep)
            result = pad + result

        return result

    def update_mask(
        self,
        prev_mask: Optional[th.Tensor],
        new_len: int,
        indices: th.Tensor,
        first: th.Tensor,
    ) -> th.Tensor:
        """Rebuild the episodic mask after culling the cache."""
        if th is None:
            raise RuntimeError("torch is required to update transformer masks")
        device = indices.device
        batch = first.shape[0]
        if prev_mask is None:
            prev_mask = th.zeros((batch, 1, self.keep_len), dtype=th.bool, device=device)
        else:
            prev_mask = prev_mask.to(device)
            if prev_mask.shape[-1] != self.keep_len:
                prev_mask = prev_mask[..., -self.keep_len :]
        not_first = ~first.to(th.bool)
        prev_mask = prev_mask & not_first
        new_mask = prev_mask.new_ones((batch, 1, new_len))
        full_mask = th.cat([prev_mask, new_mask], dim=-1)
        max_index = full_mask.shape[-1] - 1
        idx = indices.clamp(min=0, max=max_index)
        state_mask = full_mask.index_select(-1, idx)
        return state_mask


def build_memory_cull_strategy(
    config: Optional[Iterable[Tuple[int, int]]],
    keep_len: int,
) -> Optional[MemoryCullStrategy]:
    """Instantiate a :class:`MemoryCullStrategy` from a user config."""
    if config is None:
        return None
    if isinstance(config, MemoryCullStrategy):
        return config
    if isinstance(config, MemoryCullConfig):
        segments = config.segments
    else:
        segments = list(config)
    segments = [(int(length), int(stride)) for length, stride in segments]
    return MemoryCullStrategy(keep_len=keep_len, segments=segments)
