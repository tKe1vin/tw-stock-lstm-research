"""Check the package layout, hashes, and that launchers do not embed a user path."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHTS = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"
EXPECTED_SIGNAL = "37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659"
REQUIRED = [
    "AGENTS.md",
    "README.md",
    "QUICK_START.md",
    "AI_HANDOFF.md",
    "ARCHITECTURE.md",
    "DATA_DICTIONARY.md",
    "MODEL_CARDS.md",
    "RESEARCH_HISTORY.md",
    "HANDOVER_STATUS.md",
    "requirements.txt",
    "install.bat",
    "check_environment.bat",
    "run_all.bat",
    "run_444.bat",
    "run_555.bat",
    "run_555_v2.bat",
    "run_v211.bat",
    "models/model_444/444.ipynb",
    "models/model_555/555.ipynb",
    "models/model_555_v2/555_V2.ipynb",
    "models/model_555_v2/weights/weights.npz",
    "research/v211/signal_registry/developer_original/2026-11-05.csv",
    "experiments/results/v2/v211/daily/2026-11-05.csv",
    "experiments/results/v2/v211/weights/weights.npz",
    "docs/handover_checklist.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    if missing:
        print("missing: " + ", ".join(missing))
        return 1
    weight = sha256(ROOT / "experiments/results/v2/v211/weights/weights.npz")
    signal = sha256(ROOT / "experiments/results/v2/v211/daily/2026-11-05.csv")
    archived = sha256(ROOT / "research/v211/signal_registry/developer_original/2026-11-05.csv")
    copy = sha256(ROOT / "models/model_555_v2/weights/weights.npz")
    if weight != EXPECTED_WEIGHTS or copy != EXPECTED_WEIGHTS:
        print(f"weight hash {weight}")
        return 1
    if signal != EXPECTED_SIGNAL or archived != EXPECTED_SIGNAL:
        print(f"signal hash {signal}")
        return 1
    for name in ("install.bat", "run_all.bat", "run_444.bat", "run_555.bat", "run_555_v2.bat", "run_v211.bat"):
        text = (ROOT / name).read_text(encoding="utf-8")
        if "陳麒方" in text or "C:\\Users" in text or "C:/Users" in text:
            print(f"{name} contains a machine-specific path")
            return 1
    if sys.version_info[:2] != (3, 12):
        print(f"python {sys.version} is not 3.12")
        return 1
    print("package ok")
    print(f"weights {weight}")
    print(f"signal {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
