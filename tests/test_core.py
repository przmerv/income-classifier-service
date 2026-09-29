import pytest

from app.core import normalize


def test_range() -> None:
    """Test that the normalized values are in the range [0,1]."""
    assert normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_constants() -> None:
    """Test that the normalized values are all 0.0 when the input list has min == max."""
    assert normalize([9.0, 9.0]) == [0.0, 0.0]


def test_empty_raises() -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError, match="Input list cannot be empty"):
        normalize([])
