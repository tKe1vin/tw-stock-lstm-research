"""Offline recovery checks for durable partial state and rejected Git pushes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pytest

from scripts import daily_reports, persist_daily_state as persist


DAY = "2026-11-16"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Recovery test")
    git("config", "user.email", "recovery@example.invalid")
    git("commit", "--allow-empty", "-qm", "initial")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(persist, "ROOT", tmp_path)
    monkeypatch.setattr(persist, "IMMUTABLE_SHA256", {})
    monkeypatch.setattr(daily_reports, "ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    return tmp_path, git


def state_files(root: Path) -> dict[str, bytes]:
    files = {
        "data/news_update_state.json": b'{"status":"failed","reason":"network_error"}\n',
        f"outputs/555/{DAY}/status.json": b'{"status":"waiting","reason":"news_unknown"}\n',
        "reports/attempts/2026-11-17/20261117T070000000000+0800_0123456789abcdef.json":
            b'{"as_of_date":"2026-11-16","run_date":"2026-11-17","status":"partial"}\n',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def failed_run(run_date: str) -> dict:
    return {
        "run_date": run_date, "output_day": DAY, "completed_session": DAY,
        "started": f"{run_date}T07:01:00+08:00", "status": "failed", "delayed_run": True,
        "market": {"status": "failed", "exit_code": 1, "error": "provider network timeout"},
        "steps": {
            "444": {"status": "failed", "exit_code": 1, "reason": "network_error"},
            "555": {"status": "waiting", "exit_code": 2, "reason": "news_unknown"},
            "555_v2": {"status": "success", "exit_code": 0},
            "v211": {"status": "waiting", "exit_code": 2},
        },
    }


@pytest.mark.parametrize("operation", [["--commit"], ["--commit", "--push"]])
@pytest.mark.parametrize("marker", ["rebase-merge", "rebase-apply"])
def test_existing_user_rebase_is_rejected_before_state_commit_or_staging(repository, operation, marker, monkeypatch):
    root, git = repository
    files = state_files(root)
    before_head = git("rev-parse", "HEAD").stdout
    before_index = git("diff", "--cached", "--name-only").stdout
    rebase = root / ".git" / marker
    rebase.mkdir()
    sentinel = rebase / "health-test-user-state"
    sentinel.write_text("Preserve the user's paused rebase")
    monkeypatch.setattr(persist, "validate_state", lambda *_a, **_k: pytest.fail("Rebase must be rejected before state processing"))
    with pytest.raises(RuntimeError, match="rebase"):
        persist.main(operation)
    assert git("rev-parse", "HEAD").stdout == before_head
    assert git("diff", "--cached", "--name-only").stdout == before_index
    assert sentinel.read_text() == "Preserve the user's paused rebase"
    assert all((root / name).read_bytes() == raw for name, raw in files.items())


def test_cross_midnight_retries_keep_valid_partial_exports_and_both_attempts(repository):
    root, git = repository
    historical = root / "reports/daily/2026-11-13/summary.md"
    historical.parent.mkdir(parents=True)
    historical.write_text("Original complete research report\n", encoding="utf-8")
    write_json(root / "reports/latest/four_systems.json", {
        "report_schema": daily_reports.REPORT_SCHEMA, "as_of_date": "2026-11-13",
        "status": "complete", "models": {model: {"current": True} for model in daily_reports.MODELS},
    })
    git("add", ".")
    git("commit", "-qm", "published complete history")
    original = historical.read_bytes()

    prediction = root / f"outputs/555_v2/{DAY}/prediction.csv"
    prediction.parent.mkdir(parents=True)
    with prediction.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Ticker", "Signal_Date"])
        writer.writeheader()
        writer.writerows({"Ticker": ticker, "Signal_Date": DAY} for ticker in sorted(persist.TICKERS))
    write_json(prediction.with_name("status.json"), {"status": "success", "completed_session": DAY})
    saved_prediction = prediction.read_bytes()

    attempt_names = []
    for run_date in ("2026-11-17", "2026-11-18"):
        report = daily_reports.build_reports(failed_run(run_date), DAY)
        assert report["status"] == "partial"
        assert report["models"]["555_v2"]["current"] is True
        assert report["models"]["555"]["current"] is False
        assert report["last_successful_as_of_date"] == "2026-11-13"
        assert report["news"]["signal_session_scan_complete"] is False
        assert report["news"]["articles_on_signal_calendar_day"] is None
        assert report["market"]["status"] == "failed"
        assert persist.main(["--commit"]) == 0
        name = report["attempt_journal"]
        attempt_names.append(name)
        recorded = json.loads(git("show", f"HEAD:{name}").stdout)
        assert recorded["run_date"] == run_date and recorded["as_of_date"] == DAY
        assert recorded["steps"]["555"]["reason"] == "news_unknown"
        assert recorded["report"]["news"]["articles_on_signal_calendar_day"] is None
        assert git("show", f"HEAD:{prediction.relative_to(root).as_posix()}").stdout == saved_prediction
        assert historical.read_bytes() == original

    assert len(set(attempt_names)) == 2
    assert all(git("show", f"HEAD:{name}").stdout for name in attempt_names)
    assert git("show", "HEAD:reports/daily/2026-11-13/summary.md").stdout == original


@pytest.mark.parametrize("failure", ["fetch", "rejected", "interrupted"])
def test_push_retry_resumes_complete_local_checkpoint_without_new_files(repository, monkeypatch, failure):
    root, git = repository
    expected = state_files(root)
    real_git = persist.git
    calls = []
    pushes = 0
    retrying = False

    def simulated_git(repo, *args, check=True):
        nonlocal pushes
        calls.append(args)
        if args[0] == "push":
            pushes += 1
            if retrying:
                return subprocess.CompletedProcess(args, 0, b"", b"")
            if failure == "interrupted":
                raise KeyboardInterrupt("simulated interruption after local commit")
            if pushes == 1:
                return subprocess.CompletedProcess(args, 1, b"", b"rejected by remote")
            raise RuntimeError("second push rejected by remote")
        if args[0] == "fetch":
            if failure == "fetch":
                raise RuntimeError("network unavailable")
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] == "rebase":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", simulated_git)
    error = KeyboardInterrupt if failure == "interrupted" else RuntimeError
    with pytest.raises(error):
        persist.main(["--commit", "--push"])
    checkpoint = git("rev-parse", "HEAD").stdout
    assert not git("diff", "--cached", "--name-only").stdout
    assert persist.changed_state(root) == []
    for name, content in expected.items():
        assert git("show", f"HEAD:{name}").stdout == content
    before_retry = pushes
    retrying = True
    assert persist.main(["--commit", "--push"]) == 0
    assert pushes == before_retry + 1
    assert git("rev-parse", "HEAD").stdout == checkpoint
    assert not (root / ".git/rebase-merge").exists()
    assert not any("--force" in arg or "--force-with-lease" in arg for args in calls for arg in args)


@pytest.mark.parametrize("interrupted", [False, True])
def test_real_rebase_conflict_aborts_back_to_complete_checkpoint(repository, monkeypatch, interrupted):
    root, git = repository
    state = root / "data/news_update_state.json"
    write_json(state, {"status": "base"})
    git("add", ".")
    git("commit", "-qm", "shared state baseline")
    git("checkout", "-qb", "incoming")
    write_json(state, {"status": "remote"})
    git("commit", "-qam", "research: daily state 2026-11-17 [skip ci]")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("checkout", "-q", "main")
    write_json(state, {"status": "local", "reason": "partial progress"})
    local_bytes = state.read_bytes()
    real_git = persist.git
    checkpoint = None
    remote_calls = []
    retrying = False

    def simulate_remote(repo, *args, check=True):
        nonlocal checkpoint
        if args[0] == "push":
            remote_calls.append(args)
            checkpoint = git("rev-parse", "HEAD").stdout
            return subprocess.CompletedProcess(args, 0 if retrying else 1, b"", b"concurrent update")
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        result = real_git(repo, *args, check=check)
        if interrupted and args[0] == "rebase" and args[-1] != "--abort":
            assert (root / ".git/rebase-merge").exists()
            raise KeyboardInterrupt("simulated interruption while resolving rebase")
        return result

    monkeypatch.setattr(persist, "git", simulate_remote)
    error = KeyboardInterrupt if interrupted else RuntimeError
    with pytest.raises(error, match="simulated interruption|rebase aborted without overwriting either ledger"):
        persist.main(["--commit", "--push"])
    assert git("rev-parse", "HEAD").stdout == checkpoint
    assert state.read_bytes() == local_bytes
    assert git("show", "refs/remotes/origin/main:data/news_update_state.json").stdout == b'{"status": "remote"}'
    assert not git("diff", "--cached", "--name-only").stdout
    assert not (root / ".git/rebase-merge").exists()
    assert not (root / ".git/rebase-apply").exists()
    retrying = True
    assert persist.main(["--commit", "--push"]) == 0
    assert len(remote_calls) == 2


@pytest.mark.parametrize("name", ["rebase-merge", "rebase-apply"])
def test_preexisting_user_rebase_is_neither_pushed_nor_aborted(repository, monkeypatch, name):
    root, _git = repository
    metadata = root / ".git" / name
    metadata.mkdir()
    marker = metadata / "user_operation"
    marker.write_text("Preserve the user's existing rebase", encoding="utf-8")
    real_git = persist.git

    def forbid_remote_or_abort(repo, *args, check=True):
        assert args[0] not in {"push", "fetch", "rebase"}
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", forbid_remote_or_abort)
    with pytest.raises(RuntimeError, match="rebase is already in progress"):
        persist.push_state(root, "main", {})
    assert marker.read_text(encoding="utf-8") == "Preserve the user's existing rebase"


def test_failed_abort_is_explicit_and_preserves_the_interruption_cause(repository, monkeypatch):
    root, _git = repository
    real_git = persist.git

    def simulate_interrupted_rebase(repo, *args, check=True):
        if args[0] == "push":
            return subprocess.CompletedProcess(args, 1, b"", b"rejected")
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] == "rebase":
            if args[-1] == "--abort":
                return subprocess.CompletedProcess(args, 1, b"", b"index.lock prevents recovery")
            (root / ".git/rebase-merge").mkdir()
            raise KeyboardInterrupt("interrupted this invocation's rebase")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", simulate_interrupted_rebase)
    with pytest.raises(RuntimeError, match="abort failed; manual recovery required: index.lock") as failure:
        persist.push_state(root, "main", {})
    assert isinstance(failure.value.__context__, KeyboardInterrupt)
    assert (root / ".git/rebase-merge").exists()


def test_dirty_unrelated_file_stops_automatic_rebase_but_keeps_checkpoint(repository, monkeypatch):
    root, git = repository
    source = root / "scripts/local_change.py"
    source.parent.mkdir()
    source.write_text("original = True\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "tracked source")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    source.write_text("private_local_change = True\n", encoding="utf-8")
    expected = state_files(root)
    real_git = persist.git
    remote_calls = []

    def reject_push(repo, *args, check=True):
        if args[0] in {"push", "fetch", "rebase"}:
            remote_calls.append(args)
            assert args[0] == "push", "dirty source must stop before fetch or rebase"
            return subprocess.CompletedProcess(args, 1, b"", b"rejected")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", reject_push)
    with pytest.raises(RuntimeError, match="unrelated tracked files are dirty"):
        persist.main(["--commit", "--push"])
    assert len(remote_calls) == 1
    assert git("show", "HEAD:scripts/local_change.py").stdout == b"original = True\n"
    assert source.read_text(encoding="utf-8") == "private_local_change = True\n"
    assert all(git("show", f"HEAD:{name}").stdout == value for name, value in expected.items())
    assert not git("diff", "--cached", "--name-only").stdout


def test_partial_checkpoint_excludes_raw_provider_data_and_local_model_files(repository):
    root, git = repository
    expected = state_files(root)
    private = {
        "data/finmind/private/raw/TaiwanStockPrice.json": b"provider raw response",
        "data/finmind/private/bars.csv": b"full normalized provider prices",
        "research/v3/local_cache/market.csv": b"private historical prices",
        "models/model_555_v2/weights/model_card.json": b"private model card",
        "models/model_555_v2/weights/scaler.json": b"private scaler",
        "models/model_555_v2/weights/weights.npz": b"private weights",
        f"outputs/444/{DAY}/native/historical_prices.csv": b"full training prices",
        f"outputs/444/{DAY}/runs/local/model.keras": b"local trained model",
    }
    for name, content in private.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert set(persist.changed_state(root)) == set(expected)
    assert persist.main(["--commit"]) == 0
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD").stdout.decode().splitlines())
    assert tracked == set(expected)
    assert all((root / name).read_bytes() == value for name, value in private.items())


@pytest.mark.parametrize("failure", ["partial_add", "commit", "interrupted"])
def test_failed_commit_unstages_only_its_checkpoint_and_can_retry(repository, monkeypatch, failure):
    root, git = repository
    expected = state_files(root)
    initial = git("rev-parse", "HEAD").stdout
    real_git = persist.git

    def fail_commit(repo, *args, check=True):
        if args[0] == "add" and failure == "partial_add":
            real_git(repo, "add", "--", args[2])
            raise RuntimeError("simulated interruption after partially staging")
        if args[0] == "commit":
            if failure == "interrupted":
                raise KeyboardInterrupt("simulated commit interruption")
            raise RuntimeError("simulated transient commit failure")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", fail_commit)
    with pytest.raises(KeyboardInterrupt if failure == "interrupted" else RuntimeError):
        persist.main(["--commit"])
    assert git("rev-parse", "HEAD").stdout == initial
    assert not git("diff", "--cached", "--name-only").stdout
    assert set(persist.changed_state(root)) == set(expected)
    assert all((root / name).read_bytes() == value for name, value in expected.items())
    monkeypatch.setattr(persist, "git", real_git)
    assert persist.main(["--commit"]) == 0
    assert all(git("show", f"HEAD:{name}").stdout == value for name, value in expected.items())


def test_preexisting_staged_content_is_preserved_and_never_joined_to_checkpoint(repository):
    root, git = repository
    secret = root / ".env"
    secret.write_text("PRIVATE_TOKEN=fixture\n", encoding="utf-8")
    git("add", ".env")
    before = git("diff", "--cached", "--binary").stdout
    state_files(root)
    with pytest.raises(ValueError, match="index already contains staged changes"):
        persist.main(["--commit"])
    assert git("diff", "--cached", "--binary").stdout == before
    assert git("diff", "--cached", "--name-only").stdout == b".env\n"


@pytest.mark.parametrize("disguised", [False, True])
def test_no_change_retry_refuses_unrelated_local_commits(repository, monkeypatch, disguised):
    root, git = repository
    name = "scripts/unrelated.py" if disguised else "data/news_update_state.json"
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    git("add", name)
    subject = "research: daily state 2026-11-17 [skip ci]" if disguised else "unrelated local work"
    git("commit", "-qm", subject)
    before = git("rev-parse", "HEAD").stdout
    real_git = persist.git

    def forbid_remote(repo, *args, check=True):
        assert args[0] not in {"push", "fetch", "rebase"}, "unrelated history must never reach a remote operation"
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", forbid_remote)
    with pytest.raises(RuntimeError, match="Unpublished commits include unrelated changes"):
        persist.main(["--commit", "--push"])
    assert git("rev-parse", "HEAD").stdout == before
    assert not git("diff", "--cached", "--name-only").stdout


def test_no_pending_checkpoint_is_a_local_noop(repository, monkeypatch):
    _root, _git = repository
    real_git = persist.git

    def forbid_remote(repo, *args, check=True):
        assert args[0] not in {"push", "fetch", "rebase"}
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", forbid_remote)
    assert persist.main(["--commit", "--push"]) == 0


def test_missing_remote_tracking_reference_cannot_push_unknown_history(repository, monkeypatch):
    _root, git = repository
    git("update-ref", "-d", "refs/remotes/origin/main")
    real_git = persist.git

    def forbid_remote(repo, *args, check=True):
        assert args[0] not in {"push", "fetch", "rebase"}
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", forbid_remote)
    with pytest.raises(RuntimeError, match="known remote-tracking branch"):
        persist.main(["--commit", "--push"])


def test_rebase_cannot_accept_remote_rewrite_of_an_existing_published_signal(repository, monkeypatch):
    root, git = repository
    book = root / persist.BOOK_ROOTS[0]
    signal = book / f"daily/{DAY}.csv"
    signal.parent.mkdir(parents=True)
    with signal.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Ticker", "Signal_Date", "Model_Sha256"])
        writer.writeheader()
        writer.writerows({"Ticker": ticker, "Signal_Date": DAY, "Model_Sha256": persist.WEIGHT_SHA256}
                         for ticker in sorted(persist.TICKERS))
    registry = book / "signal_registry.json"
    entry = {"model_sha256": persist.WEIGHT_SHA256, "file_sha256": persist.digest(signal)}
    write_json(registry, {signal.name: entry})
    git("add", ".")
    git("commit", "-qm", "published original signal")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    original_hash = persist.digest(signal)
    state_files(root)
    real_git = persist.git
    pushes = []

    def simulate_remote_rewrite(repo, *args, check=True):
        if args[0] == "push":
            pushes.append(args)
            return subprocess.CompletedProcess(args, 1, b"", b"concurrent remote update")
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] == "rebase":
            # Model a remote history whose HEAD already contains the rewritten
            # signal and matching registry. HEAD-relative checks alone accept it.
            signal.write_bytes(signal.read_bytes() + b"\n")
            entry["file_sha256"] = persist.digest(signal)
            write_json(registry, {signal.name: entry})
            git("add", str(signal.relative_to(root)), str(registry.relative_to(root)))
            git("commit", "-qm", "simulated remote signal rewrite")
            return subprocess.CompletedProcess(args, 0, b"", b"")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(persist, "git", simulate_remote_rewrite)
    with pytest.raises(RuntimeError, match="published signal changed during rebase"):
        persist.main(["--commit", "--push"])
    assert len(pushes) == 1
    assert persist.digest(signal) != original_hash
    assert not git("diff", "--cached", "--name-only").stdout
