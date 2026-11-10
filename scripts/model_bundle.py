"""Verify the complete frozen inference bundle without loading TensorFlow.

The manifest is a reviewed repository artifact. Updating it is an explicit
maintenance action, never part of a daily prediction or an automatic repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_RELATIVE = Path("experiments/results/v2/v211/weights")
MANIFEST_RELATIVE = Path("models/model_555_v2/bundle_manifest.json")
INFERENCE_SOURCES = (
    "experiments/risk_experiment.py",
    "experiments/calendar_rules.py",
    "experiments/horizon_experiment.py",
    "experiments/v2/v28.py",
    "experiments/v2/v29.py",
    "experiments/v2/v28_rules.py",
    "experiments/v2/books.py",
    "experiments/v2/v211.py",
    "experiments/v2/v211_market.py",
    "scripts/news_update.py",
    "scripts/ledger_integrity.py",
    "scripts/common.py",
    "scripts/model_bundle.py",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_manifest(root=None, weight_dir=None):
    """Build a proposed manifest. Does not alter weights, cards, or signals."""
    root = Path(root) if root is not None else ROOT
    folder = Path(weight_dir) if weight_dir is not None else root / WEIGHT_RELATIVE
    card = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
    contract = {name: card[name] for name in (
        "model_version", "features", "time_step", "layers", "units", "dropout",
    )}
    contract["scalers_sha256"] = _json_hash(card["scalers"])
    contract["tau_sha256"] = _json_hash(card["tau"])
    contract["weight_counts"] = card["weight_counts"]
    if card["weight_sha256"] != _hash(folder / "weights.npz"):
        raise ValueError("model card weight hash mismatch")
    document = {
        "schema_version": 1,
        "model_version": card["model_version"],
        "artifacts": {
            "weights.npz": _hash(folder / "weights.npz"),
            "model_card.json": _hash(folder / "model_card.json"),
        },
        "input_contract": contract,
        "inference_sources": {name: _hash(root / name) for name in INFERENCE_SOURCES},
    }
    document["bundle_sha256"] = _json_hash(document)
    return document


def verify_bundle(weight_dir=None, root=None, card=None):
    """Fail closed on a changed weight, scaler, feature contract, or source.

Only the canonical card and weights are required. Redundant delivery copies
under models/model_555_v2 are not used to resolve an inference bundle.
"""
    root = Path(root) if root is not None else ROOT
    manifest = json.loads((root / MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    actual = create_manifest(root=root, weight_dir=weight_dir)
    if manifest != actual:
        changed = [key for key in actual if manifest.get(key) != actual[key]]
        raise ValueError("frozen model bundle mismatch: " + ", ".join(changed))
    if card is not None:
        folder = Path(weight_dir) if weight_dir is not None else root / WEIGHT_RELATIVE
        canonical = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
        if card != canonical:
            raise ValueError("in-memory model card differs from the verified bundle")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="explicitly update the reviewed bundle manifest")
    args = parser.parse_args()
    if args.write:
        manifest = create_manifest()
        path = ROOT / MANIFEST_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = verify_bundle()
    print(json.dumps({"bundle_sha256": manifest["bundle_sha256"], "verified": not args.write}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
