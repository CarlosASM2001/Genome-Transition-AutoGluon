from __future__ import annotations

from typing import Iterable

from .base import TranscriptExons, TransitionExtractor, TransitionSpec


class EzTransitionExtractor(TransitionExtractor):
    def __init__(self) -> None:
        super().__init__(
            TransitionSpec(
                key="ez",
                local_position_column="Last_Exon_End",
                window_start_offset=-49,
                window_end_offset=500,
                window_len=550,
                min_distance_from_positive=120,
                shift_candidates=(-2000, -1800, -1600, -1400, -1200, -1000, -800, 800, 1000, 1200, 1400, 1600, 1800, 2000),
            )
        )

    def iter_anchor_positions(self, ordered_exons: TranscriptExons) -> Iterable[int]:
        if ordered_exons:
            yield ordered_exons[-1][1]
