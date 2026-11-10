"""Hash the frozen weights from the file bytes and load them with TensorFlow."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED = "53a1467fdba66e7f22a45e5f2fbc1cb78fc99825b15d516b3e7153d7bfc28144"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-only", action="store_true", help="verify the complete bundle without importing TensorFlow")
    args = parser.parse_args(argv)
    from scripts.model_bundle import verify_bundle

    bundle = verify_bundle()
    print(f"bundle_sha256 {bundle['bundle_sha256']}")
    path = ROOT / "experiments" / "results" / "v2" / "v211" / "weights" / "weights.npz"
    card_path = path.with_name("model_card.json")
    digest = sha256(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    print(f"computed_sha256 {digest}")
    print(f"card_sha256 {sha256(card_path)}")
    print(f"card_declared_weight_sha256 {card.get('weight_sha256')}")
    if digest != EXPECTED:
        print("computed hash does not match the frozen weight hash")
        return 1
    if card.get("weight_sha256") != digest:
        print("model card hash does not match the file")
        return 1
    if args.hash_only:
        print("model bundle verified without TensorFlow")
        return 0
    os_level = __import__("os")
    os_level.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np

    store = np.load(path)
    from experiments.risk_experiment import FEATURES
    from experiments.v2.v29 import CorrectedFitter

    fitter = CorrectedFitter()
    for ticker, count in card["weight_counts"].items():
        _key, model = fitter._fitter._model(2, 50, len(FEATURES))
        model.set_weights([store[f"{ticker}_{i}"] for i in range(int(count))])
        print(f"loaded {ticker} arrays {count}")
    print("model load ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
