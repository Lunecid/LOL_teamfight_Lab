"""CLI: ``python -m studies.training_runtime_20261002``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .profiles import profile_json
from .preflight import _main as _preflight_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["profiles"] or values == ["--profiles"]:
        print(profile_json(), flush=True)
        return 0
    if values and values[0] == "profiles":
        parser = argparse.ArgumentParser(description="runtime profile JSON")
        parser.error("profiles does not accept additional arguments")
    return _preflight_main(values)


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(main())


__all__ = ["main"]
