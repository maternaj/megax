"""Bookmaker client errors."""


class BookmakerAuthError(RuntimeError):
    """Raised when an authenticated request cannot obtain a logged-in session."""
