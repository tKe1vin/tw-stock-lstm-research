"""A fresh checkout can omit archived predictions without weakening guards."""
import pytest

from experiments.v2 import guard


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    original_root = guard.ROOT
    required = tuple(tmp_path / path.relative_to(original_root) for path in guard.REQUIRED_PROTECTED)
    optional = tuple(tmp_path / path.relative_to(original_root) for path in guard.OPTIONAL_PROTECTED)
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected input\n", encoding="utf-8")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "REQUIRED_PROTECTED", required)
    monkeypatch.setattr(guard, "OPTIONAL_PROTECTED", optional)
    return tmp_path, required, optional


def test_fresh_checkout_without_archived_predictions_has_explicit_markers(checkout):
    root, required, optional = checkout
    saved = guard.snapshot()
    assert len(saved) == len(required) + len(optional)
    assert all(isinstance(value, str) for value in saved.values())
    assert {str(path.relative_to(root)) for path in optional} == {
        "experiments/results/walk_forward_predictions.csv",
        "experiments/results/signal_models/predictions.csv",
    }
    assert all(saved[str(path.relative_to(root))] == guard.MISSING_OPTIONAL for path in optional)
    guard.assert_unchanged(saved)


def test_missing_required_input_is_still_an_error(checkout):
    _root, required, _optional = checkout
    required[0].unlink()
    with pytest.raises(FileNotFoundError):
        guard.snapshot()


def test_optional_archive_appearance_is_detected(checkout):
    _root, _required, optional = checkout
    saved = guard.snapshot()
    optional[0].parent.mkdir(parents=True, exist_ok=True)
    optional[0].write_text("archived predictions\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="walk_forward_predictions.csv"):
        guard.assert_unchanged(saved)


def test_optional_archive_deletion_is_detected(checkout):
    _root, _required, optional = checkout
    optional[0].parent.mkdir(parents=True, exist_ok=True)
    optional[0].write_text("archived predictions\n", encoding="utf-8")
    saved = guard.snapshot()
    optional[0].unlink()
    with pytest.raises(AssertionError, match="walk_forward_predictions.csv"):
        guard.assert_unchanged(saved)


def test_required_input_changes_are_detected(checkout):
    _root, required, _optional = checkout
    saved = guard.snapshot()
    required[0].write_text("modified input\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="444_baseline.ipynb"):
        guard.assert_unchanged(saved)
