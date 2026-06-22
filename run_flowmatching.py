from __future__ import annotations

import sys
from singlecell_generative_unified.run import main

if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "flowmatching"])
    main()
