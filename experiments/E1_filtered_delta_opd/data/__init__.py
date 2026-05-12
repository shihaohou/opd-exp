"""E1 data loaders + builders."""

from .virl39k_loader import ViRL39KSample, iter_virl39k, parse_boxed
from .pope_style_builder import (
    POPEStyleSample,
    COCOInstanceIndex,
    iter_pope_style,
    load_pope_adv_image_ids,
)
from .tallyqa_loader import TallyQASample, iter_tallyqa
from .synthesize_counterfactuals import (
    SyntheticSample,
    build_synthetic_counterfactuals,
    iter_synthetic_counterfactuals,
)
from .mixture import (
    build_mixture,
    iter_mixture_manifest,
    DEFAULT_TARGETS,
)

__all__ = [
    # ViRL39K (bucket 1)
    "ViRL39KSample",
    "iter_virl39k",
    "parse_boxed",
    # POPE-style on COCO train (bucket 2)
    "POPEStyleSample",
    "COCOInstanceIndex",
    "iter_pope_style",
    "load_pope_adv_image_ids",
    # TallyQA complex (bucket 3a)
    "TallyQASample",
    "iter_tallyqa",
    # Synthetic counterfactuals (bucket 3b)
    "SyntheticSample",
    "build_synthetic_counterfactuals",
    "iter_synthetic_counterfactuals",
    # 8K E1-mini mixture
    "build_mixture",
    "iter_mixture_manifest",
    "DEFAULT_TARGETS",
]
