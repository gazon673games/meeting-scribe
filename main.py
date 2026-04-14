from __future__ import annotations

import sys

from app_bootstrap import main


if __name__ == "__main__":
    if "--smoke-import" in sys.argv:
        raise SystemExit(0)
    main()
