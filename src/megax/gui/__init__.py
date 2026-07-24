"""MegaX web GUI."""

__all__ = ["create_app", "main"]


def __getattr__(name: str):
    if name in __all__:
        from megax.gui.app import create_app, main

        return create_app if name == "create_app" else main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
