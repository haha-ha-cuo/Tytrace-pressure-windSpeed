import numpy as np
import pytest

from tytrace_fields.interpolation import bilinear_resample, build_axis


def test_bilinear_interpolates_four_corners() -> None:
    values = np.asarray([[0.0, 10.0], [20.0, 30.0]])
    result = bilinear_resample(values, [0, 1], [0, 1], [0.5], [0.5])
    assert result.tolist() == [[15.0]]


def test_bilinear_accepts_descending_latitude() -> None:
    values = np.asarray([[20.0, 30.0], [0.0, 10.0]])
    result = bilinear_resample(values, [1, 0], [0, 1], [0, 0.5, 1], [0, 0.5, 1])
    assert result[0, 0] == 0
    assert result[1, 1] == 15
    assert result[2, 2] == 30


def test_bilinear_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="outside"):
        bilinear_resample([[0, 1], [2, 3]], [0, 1], [0, 1], [-0.1], [0.5])


def test_build_axis_includes_non_aligned_end_without_exceeding_it() -> None:
    assert build_axis(0, 1, 0.4).tolist() == pytest.approx([0, 0.4, 0.8, 1])

