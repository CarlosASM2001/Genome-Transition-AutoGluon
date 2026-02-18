from dataclasses import dataclass
from typing import Iterable, Sequence

ExonCoordinates = tuple[int, int]
TranscriptExons = Sequence[ExonCoordinates]


@dataclass(frozen=True)
class TransitionSpec:
    key: str
    local_position_column: str
    window_start_offset: int
    window_end_offset: int
    window_len: int
    min_distance_from_positive: int
    shift_candidates: tuple[int, ...]


class TransitionExtractor:
    def __init__(self, spec: TransitionSpec) -> None:
        self.spec = spec

    def iter_anchor_positions(self, ordered_exons: TranscriptExons) -> Iterable[int]:
        raise NotImplementedError