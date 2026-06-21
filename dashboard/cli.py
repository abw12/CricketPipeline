from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    os.environ.setdefault("CRICKET_DASHBOARD_ALLOW_SHUTDOWN", "1")
    app_path = Path(__file__).resolve().parent / "app" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
