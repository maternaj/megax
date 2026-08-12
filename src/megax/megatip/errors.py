"""Megatip API errors."""


class MegatipJokerError(RuntimeError):
    """Joker assignment did not stick (API returned jokerUsed=false)."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Joker assignment failed")


class LineupSubmitError(ValueError):
    """Could not map optimizer tips to round matches."""

    def __init__(self, missing_match_ids: list[int]):
        self.missing_match_ids = missing_match_ids
        super().__init__(f"No round match for match_id(s): {missing_match_ids}")
