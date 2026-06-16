"""Tests for the deterministic match-level split."""

from __future__ import annotations

from collections import Counter

from app.dataset.split import Split, split_for_match


def test_split_is_deterministic() -> None:
    assert split_for_match("1082592") == split_for_match("1082592")


def test_split_distribution_is_roughly_right() -> None:
    counts: Counter[Split] = Counter(split_for_match(str(i)) for i in range(5000))
    train_frac = counts[Split.TRAIN] / 5000
    # default 80/10/10; allow slack on a finite sample
    assert 0.74 < train_frac < 0.86
    assert counts[Split.VAL] > 0
    assert counts[Split.TEST] > 0


def test_zero_fractions_make_everything_train() -> None:
    assert all(
        split_for_match(str(i), val_fraction=0.0, test_fraction=0.0) is Split.TRAIN
        for i in range(50)
    )
