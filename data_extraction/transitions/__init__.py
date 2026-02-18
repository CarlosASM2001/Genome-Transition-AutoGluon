from .base import TransitionExtractor
from .ei import EiTransitionExtractor
from .ez import EzTransitionExtractor
from .ie import IeTransitionExtractor
from .ze import ZeTransitionExtractor


def build_transition_extractors() -> dict[str, TransitionExtractor]:
    extractors = [
        EiTransitionExtractor(),
        IeTransitionExtractor(),
        ZeTransitionExtractor(),
        EzTransitionExtractor(),
    ]
    return {extractor.spec.key: extractor for extractor in extractors}


__all__ = [
    "TransitionExtractor",
    "build_transition_extractors",
]
