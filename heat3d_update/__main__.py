"""`python -m heat3d_update [--check]`: update this copy of the tools to the newest release."""

from __future__ import annotations

import argparse
import sys

from . import apply, installed_version, is_newer, latest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="heat3d_update", description=__doc__)
    p.add_argument("--check", action="store_true", help="only say whether a newer release is out")
    args = p.parse_args(argv)
    release = latest(use_cache=False)
    current = installed_version()
    if release is None:
        print(f"No release found (this is {current}); offline, or nothing published yet.")
        return 0
    if not is_newer(release.version, current):
        print(f"Up to date: {current}.")
        return 0
    print(f"{release.version} is out (this is {current}): {release.url}")
    if args.check:
        return 0
    try:
        print(apply(release))
    except RuntimeError as e:
        print(f"Not updated: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
