from typing import Iterable

from base import TranscriptExons, TransitionExtractor, TransitionSpec


class IeTransitionExtractor(TransitionExtractor):
    def __init__(self) -> None:
        super().__init__(
            TransitionSpec(
                key="ie",
                local_position_column="Exon_Start",
                window_start_offset=-100,
                window_end_offset=4,
                window_len=105,
                min_distance_from_positive=15,
                shift_candidates=(-400, -350, -300, -250, -200, -150, -120, 120, 150, 200, 250, 300, 350, 400),
            )
        )

    def iter_anchor_positions(self, ordered_exons: TranscriptExons) -> Iterable[int]:
        for _, exon_right in zip(ordered_exons, ordered_exons[1:]):
            yield exon_right[0]