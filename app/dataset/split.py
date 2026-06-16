"""Deterministic match-level train/val/test split.

Split by match, never by ball: deliveries within a match are correlated, so a
random row split would leak. Defined in Phase 2 (so the teacher narrates only
training matches and held-out matches stay clean for evaluation) and reused
unchanged in Phase 3. The split is a pure function of the match id, so it is
stable across runs and machines without storing a split file.
"""

from __future__ import annotations

import hashlib
from enum import Enum

_SALT = "cricket-commentary-split-v1"
_SCALE = 0xFFFFFFFF


class Split(str, Enum):
    """Which partition a match belongs to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def split_for_match(
    match_id: str, *, val_fraction: float = 0.1, test_fraction: float = 0.1
) -> Split:
    """Return the partition for a match, deterministically from its id.

    The first ``test_fraction`` of the hash space is test, the next
    ``val_fraction`` is val, the remainder is train. Same id always maps to the
    same split.
    """
    digest = hashlib.sha256(f"{_SALT}:{match_id}".encode()).hexdigest()
    position = int(digest[:8], 16) / _SCALE
    if position < test_fraction:
        return Split.TEST
    if position < test_fraction + val_fraction:
        return Split.VAL
    return Split.TRAIN
