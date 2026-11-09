import json

import pytest

from scripts.reproducibility import preserve_inputs, restore_inputs


def test_sources_survive_original_removal_and_replay_is_byte_identical(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"Date,Close\n2026-11-06,100.0000000001\n")
    original = source.read_bytes()
    manifest = tmp_path / "report/inputs.json"
    first = preserve_inputs({"prices": source}, manifest, root=tmp_path)
    preserve_inputs({"prices": source}, manifest, root=tmp_path)
    assert len(list((tmp_path / "data/research_snapshots").iterdir())) == 1
    source.unlink()
    restored = restore_inputs(manifest, tmp_path / "replay", root=tmp_path)
    assert restored["prices"].read_bytes() == original
    assert first["training_bitwise_reproduction_claimed"] is False


def test_replay_rejects_changed_content_and_path_escape_before_writing(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n")
    manifest = tmp_path / "manifest.json"
    preserve_inputs({"prices": source}, manifest, root=tmp_path)
    document = json.loads(manifest.read_text())
    document["inputs"]["prices"]["archive"] = "../outside.gz"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leaves"):
        restore_inputs(manifest, tmp_path / "replay", root=tmp_path)
    assert not (tmp_path / "replay").exists()
