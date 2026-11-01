from .cli import main


def _check_for_update() -> None:
    """A newer release of the tools: say so, and offer it (see heat3d_update in the repository root)."""
    import sys as _sys
    from pathlib import Path as _Path

    root = str(_Path(__file__).resolve().parents[2])
    if root not in _sys.path:
        _sys.path.append(root)
    try:
        import heat3d_update
    except ImportError:
        return
    heat3d_update.startup("heat3d_gamefiles")


_check_for_update()
raise SystemExit(main())
