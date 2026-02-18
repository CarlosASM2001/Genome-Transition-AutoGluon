from base import TransitionExtractor
from ei_extracion import EiTransitionExtractor
from ez_extracion import EzTransitionExtractor
from ie_extracion import IeTransitionExtractor
from ze_extracion import ZeTransitionExtractor


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