from __future__ import annotations

from datetime import date

from scripts.runner import is_stale


def test_is_stale_helper_logic() -> None:
    today = date(2026, 3, 3)

    assert is_stale(None, today=today, stale_days=3) is True
    assert is_stale(date(2026, 3, 3), today=today, stale_days=3) is False
    assert is_stale(date(2026, 3, 1), today=today, stale_days=3) is False
    assert is_stale(date(2026, 2, 28), today=today, stale_days=3) is True
