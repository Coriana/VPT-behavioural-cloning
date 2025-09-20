from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, Union

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
        # Sort by step so we can reason about oldest/newest ordering directly.
        steps_list: List[Tuple[int, int]] = sorted(
            zip(valid_steps.tolist(), valid_positions.tolist())
        )

        used: List[int] = []
        used_set: Set[int] = set()
        per_tier_indices: List[List[int]] = [[] for _ in self.tiers]

        # Tier 0 always keeps the newest frames exactly as requested.
        first_tier = self.tiers[0]
        if first_tier.max_keep > 0 and steps_list:
            newest = steps_list[-first_tier.max_keep :]
            tier_indices = sorted(pos for _, pos in newest)
            per_tier_indices[0] = tier_indices
            used.extend(tier_indices)
            used_set.update(tier_indices)

        remaining = [entry for entry in steps_list if entry[1] not in used_set]

        for tier_idx, tier in enumerate(self.tiers[1:], start=1):
            if tier.max_keep == 0 or not remaining:
                continue

            tier_selected: List[int] = []
            last_step: Optional[int] = None

            for step_value, pos in remaining:
                if last_step is not None and step_value - last_step < tier.stride:
                    continue

                tier_selected.append(pos)
                last_step = step_value
                if len(tier_selected) >= tier.max_keep:
                    break

            if tier_selected:
                tier_selected.sort()
                per_tier_indices[tier_idx] = tier_selected
                used.extend(tier_selected)
                used_set.update(tier_selected)
                remaining = [entry for entry in steps_list if entry[1] not in used_set]
            else:
                per_tier_indices[tier_idx] = []

        used = sorted(set(used))

        per_tier_tensors: List[th.Tensor] = []
        for tier_indices in per_tier_indices:
            if tier_indices:
                per_tier_tensors.append(
                    th.tensor(tier_indices, dtype=th.long, device=device)
                )
            else:
                per_tier_tensors.append(th.empty(0, dtype=th.long, device=device))

        if used:
            indices_tensor = th.tensor(used, dtype=th.long, device=device)
        else:
            indices_tensor = th.empty(0, dtype=th.long, device=device)

        return MemoryCullSelection(indices=indices_tensor, per_tier=per_tier_tensors)
