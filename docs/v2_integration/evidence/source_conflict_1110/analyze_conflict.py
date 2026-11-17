"""Optional local conflict analyzer. Defaults to in-repo SYNTHETIC_DEMO evidence CSVs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "LOCAL_2026-11-10.csv"
GITHUB = HERE / "GITHUB_2026-11-10.csv"


def main() -> None:
    local_sha = hashlib.sha256(LOCAL.read_bytes()).hexdigest()
    github_sha = hashlib.sha256(GITHUB.read_bytes()).hexdigest()
    payload = {
        "demo_flag": "SYNTHETIC_DEMO",
        "local_sha256": local_sha,
        "github_sha256": github_sha,
        "conflict": local_sha != github_sha,
        "note": "Compares in-repo synthetic evidence only; set TW_V211_LOCAL / TW_GITHUB_REVIEW_ROOT for live roots.",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
