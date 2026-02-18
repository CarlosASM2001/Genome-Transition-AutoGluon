from typing import Iterable

from .base import TranscriptExons, TransitionExtractor, TransitionSpec


class EiTransitionExtractor(TransitionExtractor):
    def __init__(self) -> None:
        super().__init__(
            TransitionSpec(
                key="ei",
                local_position_column="Intron_Start",
                window_start_offset=-5,
                window_end_offset=6,
                window_len=12,
                min_distance_from_positive=8,
                shift_candidates=(-120, -100, -80, -60, -40, -30, -20, 20, 30, 40, 60, 80, 100, 120),
            )
        )

    def iter_anchor_positions(self, ordered_exons: TranscriptExons) -> Iterable[int]:
        for exon_left, _ in zip(ordered_exons, ordered_exons[1:]):
            yield exon_left[1] + 1