"""Permite ejecutar el paquete con ``python -m nba_sh``."""

from __future__ import annotations

import sys

from nba_sh.cli import main

if __name__ == "__main__":
    sys.exit(main())
