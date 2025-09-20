"""Utility to visualize the multi-tier memory culling schedule.

The script replicates the policy's default compression schedule and shows
which frames from an input video would be retained at each tier when the
``MemoryCullStrategy`` is applied. Four OpenCV windows are created—one per
memory tier—so you can watch frames move through the cache in real time.
"""

import argparse
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from openai_vpt.lib.memory_cull import MemoryCullStrategy, MemoryTier

# Match the policy's default compression schedule (newest -> oldest tiers).
DEFAULT_SEGMENTS: Sequence[Tuple[int, int]] = (
    (32, 1),
    (32, 3),
    (32, 9),
    (32, 27),
)


def build_mosaic(frames: Sequence[np.ndarray], tile: int, cols: int) -> np.ndarray:
    """Arrange frames into a tiled mosaic for visualization."""

    if not frames:
        return np.zeros((tile, tile, 3), dtype=np.uint8)

    resized = [cv2.resize(frame, (tile, tile)) for frame in frames]
    rows = (len(resized) + cols - 1) // cols
    mosaic = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)

    for idx, frame in enumerate(resized):
        row, col = divmod(idx, cols)
        start_row = row * tile
        start_col = col * tile
        mosaic[start_row : start_row + tile, start_col : start_col + tile] = frame

    return mosaic


def _select_frames(
    cache: Sequence[np.ndarray],
    per_tier: Sequence[Iterable[int]],
    tiers: Sequence[MemoryTier],
    blank: np.ndarray,
) -> List[List[np.ndarray]]:
    """Map tier selections to concrete frame lists with left padding."""

    if not cache:
        return [[blank] * tier.max_keep for tier in tiers]

    tier_frames: List[List[np.ndarray]] = []
    for tier, indices in zip(tiers, per_tier):
        materialized = [cache[idx] for idx in indices]
        pad = tier.max_keep - len(materialized)
        if pad > 0:
            materialized = [blank] * pad + materialized
        tier_frames.append(materialized)

    return tier_frames


def main(
    video_path: str,
    segments: Sequence[Tuple[int, int]] = DEFAULT_SEGMENTS,
    tile: int = 128,
    cols: int = 8,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    tiers = [MemoryTier(stride=stride, max_keep=length) for length, stride in segments]
    strategy = MemoryCullStrategy(tiers=tiers)

    cache: List[np.ndarray] = []
    steps: List[int] = []
    blank: Optional[np.ndarray] = None
    next_step = 0

    window_titles = [f"Tier {i + 1} (stride {tier.stride})" for i, tier in enumerate(tiers)]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if blank is None:
                blank = np.zeros_like(frame)

            full_cache = cache + [frame]
            full_steps = steps + [next_step]
            next_step += 1

            selection = strategy.select_indices(full=None, step_indices=[full_steps])[0]
            keep_indices = selection.indices.tolist()

            cache = [full_cache[i] for i in keep_indices]
            steps = [full_steps[i] for i in keep_indices]

            index_map = {original: new for new, original in enumerate(keep_indices)}
            per_tier_indices = []
            for tier_indices in selection.per_tier:
                mapped = [index_map[idx.item()] for idx in tier_indices]
                per_tier_indices.append(mapped)

            assert blank is not None  # for type checkers
            tier_frames = _select_frames(cache, per_tier_indices, tiers, blank)

            for title, frames in zip(window_titles, tier_frames):
                mosaic = build_mosaic(frames, tile=tile, cols=cols)
                cv2.imshow(title, mosaic)

            # ESC to exit early
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the video file to visualize")
    parser.add_argument("--tile", type=int, default=128, help="Tile size for each thumbnail (pixels)")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns in each mosaic")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.video, tile=args.tile, cols=args.cols)
