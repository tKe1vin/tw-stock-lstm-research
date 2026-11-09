"""Bundle verification catches changes beyond the NPZ without importing TF."""
import builtins
import copy
import json
import shutil

import pytest

from scripts import model_bundle as bundle


@pytest.fixture
def frozen_bundle(tmp_path):
    root = bundle.ROOT
    for name in (*bundle.INFERENCE_SOURCES, str(bundle.WEIGHT_RELATIVE / "model_card.json"),
                 str(bundle.WEIGHT_RELATIVE / "weights.npz")):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    manifest = bundle.create_manifest(root=tmp_path)
    path = tmp_path / bundle.MANIFEST_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_bundle_does_not_need_delivery_copies_or_tensorflow(frozen_bundle, monkeypatch):
    before = {p: p.read_bytes() for p in frozen_bundle.rglob("*") if p.is_file()}
    original = builtins.__import__

    def forbid_tf(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            pytest.fail("bundle checks must not import TensorFlow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_tf)
    assert not (frozen_bundle / "models/model_555_v2/weights/weights.npz").exists()
    assert bundle.verify_bundle(root=frozen_bundle)["bundle_sha256"]
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("mutation", ["scaler", "tau", "feature_order", "time_step", "layers", "weight", "source"])
def test_bundle_rejects_changed_inference_inputs(frozen_bundle, mutation):
    folder = frozen_bundle / bundle.WEIGHT_RELATIVE
    path = folder / "model_card.json"
    card = json.loads(path.read_text())
    if mutation == "weight":
        (folder / "weights.npz").write_bytes(b"different frozen model")
    elif mutation == "source":
        source = frozen_bundle / "experiments/risk_experiment.py"
        source.write_bytes(source.read_bytes() + b"\n# changed feature computation\n")
    else:
        if mutation == "scaler":
            card["scalers"]["2330.TW"]["scale"][0] *= 2
        elif mutation == "tau":
            card["tau"]["2330.TW"] += 1
        elif mutation == "feature_order":
            card["features"][0], card["features"][1] = card["features"][1], card["features"][0]
        else:
            card[mutation] += 1
        path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="hash mismatch|bundle mismatch"):
        bundle.verify_bundle(root=frozen_bundle)


def test_bundle_rejects_unverified_in_memory_scaler(frozen_bundle):
    card = json.loads((frozen_bundle / bundle.WEIGHT_RELATIVE / "model_card.json").read_text())
    changed = copy.deepcopy(card)
    changed["scalers"]["2330.TW"]["min"][0] += 0.1
    with pytest.raises(ValueError, match="in-memory model card"):
        bundle.verify_bundle(root=frozen_bundle, card=changed)


def test_prediction_rejects_runtime_feature_order_before_model_loading(monkeypatch):
    from experiments.v2 import v211

    card = json.loads((bundle.ROOT / bundle.WEIGHT_RELATIVE / "model_card.json").read_text())
    monkeypatch.setattr(v211, "verify_bundle", lambda **kwargs: {})
    monkeypatch.setattr(v211, "FEATURES", list(reversed(card["features"])))
    with pytest.raises(ValueError, match="runtime feature order"):
        v211.predict_rows(card, None, "2026-11-06")
