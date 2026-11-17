"""New daily journals persist while published attempts remain immutable."""
from pathlib import Path
import subprocess

import pytest

from scripts import persist_daily_state as persist


NAME = "reports/attempts/2026-11-17/20261117T070153255335+0800_0123456789abcdef.json"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "commit", "--allow-empty", "-qm", "initial")
    return tmp_path


def test_attempt_allowlist_is_narrow_and_does_not_accept_hidden_or_raw_files():
    assert persist.allowed_path(NAME)
    assert persist.allowed_path("reports/attempts/2026-11-17/summary.md")
    for name in ("reports/attempts/2026-11-17/.env", "reports/attempts/2026-11-17/raw.json",
                 "reports/attempts/not-a-date/summary.md", "reports/attempts/2026-11-17/../secrets.json"):
        assert not persist.allowed_path(name)


def test_attempt_is_selected_for_durable_state_then_cannot_be_rewritten(tmp_path):
    root = repository(tmp_path)
    attempt = root / NAME
    attempt.parent.mkdir(parents=True)
    attempt.write_text('{"actual_run_date":"2026-11-17","as_of":"2026-11-16"}\n')
    assert NAME in persist.changed_state(root)
    persist.validate_state(root, immutable={})
    git(root, "add", NAME)
    git(root, "commit", "-qm", "published attempt")
    persist.validate_state(root, immutable={})
    attempt.write_text('{"as_of":"2026-11-17"}\n')
    with pytest.raises(ValueError, match="Published daily attempt changed"):
        persist.validate_state(root, immutable={})


def test_new_attempt_and_mutable_day_index_can_append_without_changing_prior(tmp_path):
    root = repository(tmp_path)
    attempt = root / NAME
    attempt.parent.mkdir(parents=True)
    attempt.write_text('{"state":"waiting"}\n')
    git(root, "add", NAME)
    git(root, "commit", "-qm", "first")
    prior = attempt.read_bytes()
    second = attempt.with_name("20261117T210000000000+0800_fedcba9876543210.json")
    second.write_text('{"state":"complete"}\n')
    (attempt.parent / "summary.md").write_text("Both attempts are retained.\n")
    persist.validate_state(root, immutable={})
    assert attempt.read_bytes() == prior
    assert set(persist.changed_state(root)) == {str(second.relative_to(root)), str((attempt.parent / "summary.md").relative_to(root))}
