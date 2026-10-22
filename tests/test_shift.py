"""The automatic time shift (shift.py)."""

import math

import pytest

from forge_fibo2.native import shift


def test_reference_points():
    # the reference pipeline's line: 256 tokens -> 0.5, 4096 -> 1.15
    assert shift.calculate_shift(256) == pytest.approx(0.5)
    assert shift.calculate_shift(4096) == pytest.approx(1.15)


def test_sizes():
    assert shift.dynamic_shift(1024, 1024) == pytest.approx(1.15)
    assert shift.dynamic_shift(256, 256) == pytest.approx(0.5)
    assert shift.dynamic_shift(1424, 800) == pytest.approx(0.5 + 0.65 / 3840 * (89 * 50 - 256))
    assert shift.dynamic_shift(832, 1216) < 1.15 < shift.dynamic_shift(1424, 800)
    # not clamped above 1 MP, as upstream
    assert shift.dynamic_shift(2048, 2048) == pytest.approx(0.5 + 0.65 / 3840 * (128 * 128 - 256))
    # a side that is not a multiple of 16 counts its whole patches only
    assert shift.dynamic_shift(1030, 1024) == shift.dynamic_shift(1024, 1024)
    # mu is the exponent: at 1024x1024 the sigma halfway is e^1.15 / (e^1.15 + 1)
    assert math.exp(shift.dynamic_shift(1024, 1024)) == pytest.approx(math.exp(1.15))
