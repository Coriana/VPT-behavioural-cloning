from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch as th


@dataclass(frozen=True)
class MemoryTier:
    """Configuration for a single culling tier."""

    stride: int
    max_keep: int

    def __post_init__(self):
        if self.stride <= 0:
            raise ValueError("stride must be strictly positive")
        if self.max_keep < 0:
            raise ValueError("max_keep must be non-negative")


@dataclass
class MemoryCullSelection:
    """Selection result for a single batch element."""

    indices: th.Tensor
    per_tier: List[th.Tensor]


class MemoryCullStrategy:
    """Select cache positions to keep according to multi-tier stride rules."""

    def __init__(self, tiers: Sequence[Union[MemoryTier, Tuple[int, int], dict]]):
        if not tiers:
            raise ValueError("At least one tier must be provided")
        self.tiers: List[MemoryTier] = [self._coerce_tier(tier) for tier in tiers]

    @staticmethod
    def _coerce_tier(tier: Union[MemoryTier, Tuple[int, int], dict]) -> MemoryTier:
        if isinstance(tier, MemoryTier):
            return tier
        if isinstance(tier, dict):
            if "stride" not in tier:
                raise KeyError("Tier dictionaries must include 'stride'")
            stride = int(tier["stride"])
            if "max_keep" in tier:
                max_keep = int(tier["max_keep"])
            elif "max_length" in tier:
                max_keep = int(tier["max_length"])
            else:
                raise KeyError("Tier dictionaries must include 'max_keep' or 'max_length'")
            return MemoryTier(stride=stride, max_keep=max_keep)
        if isinstance(tier, (list, tuple)):
            if len(tier) != 2:
                raise ValueError("Tier tuples must have exactly two elements: (stride, max_keep)")
            stride, max_keep = tier
            return MemoryTier(stride=int(stride), max_keep=int(max_keep))
        raise TypeError(f"Unsupported tier specification: {type(tier)!r}")

    def select_indices(
        self,
        full: th.Tensor,
        step_indices: Union[th.Tensor, Sequence[Sequence[int]], None],
    ) -> List[MemoryCullSelection]:
        """Return the indices that should be kept for each batch element.

        Args:
            full: Tensor whose leading dimensions correspond to ``[batch, time]``.
                Only used for device/shape inference; the contents are ignored.
            step_indices: Absolute step indices for each cached position. Negative
                values are treated as invalid and ignored.

        Returns:
            A list of :class:`MemoryCullSelection` objects, one per batch element.
        """

        if step_indices is None:
            if full is None:
                raise ValueError("Either 'step_indices' or 'full' must be provided")
            device = full.device
            batch, time = full.shape[0], full.shape[1]
            step_indices = (
                th.arange(time, device=device, dtype=th.long)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        else:
            if not th.is_tensor(step_indices):
                device = full.device if full is not None else None
                step_indices = th.as_tensor(step_indices, device=device, dtype=th.long)
            else:
                if full is not None:
                    step_indices = step_indices.to(device=full.device)
                step_indices = step_indices.to(dtype=th.long)

        if step_indices.dim() == 1:
            step_indices = step_indices.unsqueeze(0)
        if step_indices.dim() != 2:
            raise ValueError("'step_indices' must be a rank-2 tensor")

        batch = step_indices.shape[0]
        if full is not None and full.shape[0] != batch:
            raise ValueError("Batch dimension of 'full' and 'step_indices' must match")

        selections: List[MemoryCullSelection] = []
        for b in range(batch):
            selections.append(self._select_single(step_indices[b]))
        return selections

    def _select_single(self, steps_row: th.Tensor) -> MemoryCullSelection:
        device = steps_row.device
        valid_positions = (steps_row >= 0).nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            empty = th.empty(0, dtype=th.long, device=device)
            return MemoryCullSelection(indices=empty, per_tier=[empty.clone() for _ in self.tiers])

        valid_steps = steps_row.index_select(0, valid_positions)
        sorted_steps, order = th.sort(valid_steps, descending=True, stable=True)
        sorted_positions = valid_positions.index_select(0, order)

        tier_last_step: List[Optional[int]] = [None] * len(self.tiers)
        per_tier: List[List[int]] = [[] for _ in self.tiers]

        for pos, step in zip(sorted_positions.tolist(), sorted_steps.tolist()):
            for tier_idx, tier in enumerate(self.tiers):
                if tier.max_keep == 0:
                    continue
                if len(per_tier[tier_idx]) >= tier.max_keep:
                    continue
                last = tier_last_step[tier_idx]
                if last is None or last - step >= tier.stride:
                    per_tier[tier_idx].append(pos)
                    tier_last_step[tier_idx] = step
                    break

        per_tier_tensors: List[th.Tensor] = []
        combined_indices: List[int] = []
        for indices in per_tier:
            if indices:
                sorted_indices = sorted(indices)
                per_tier_tensors.append(
                    th.tensor(sorted_indices, dtype=th.long, device=device)
                )
                combined_indices.extend(sorted_indices)
            else:
                per_tier_tensors.append(th.empty(0, dtype=th.long, device=device))

        if combined_indices:
            combined_indices = sorted(set(combined_indices))
            indices_tensor = th.tensor(combined_indices, dtype=th.long, device=device)
        else:
            indices_tensor = th.empty(0, dtype=th.long, device=device)

        return MemoryCullSelection(indices=indices_tensor, per_tier=per_tier_tensors)
