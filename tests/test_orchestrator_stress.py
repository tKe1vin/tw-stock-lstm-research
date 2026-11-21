"""Real process failures must not admit overlapping writers or leave late writers."""
from __future__ import annotations

import multiprocessing as mp
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import run_all_daily as daily


def _race_owner(lock, gate, active, violations, completed):
    daily.LOCK = Path(lock)
    gate.wait(timeout=15)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        fd = daily._acquire()
        if fd is None:
            time.sleep(.002)
            continue
        try:
            with active.get_lock():
                active.value += 1
                if active.value > 1:
                    violations.value += 1
            time.sleep(.008)
            with active.get_lock():
                active.value -= 1
            with completed.get_lock():
                completed.value += 1
        finally:
            daily._release(fd)
        return
    raise AssertionError("A contending process could not recover the released lock")


def _paused_owner(lock, before_write, allow_write, owner_ready, stop):
    daily.LOCK = Path(lock)
    original = os.write
    def paused(fd, value):
        if value == str(os.getpid()).encode("ascii"):
            before_write.set()
            assert allow_write.wait(timeout=10)
        return original(fd, value)
    os.write = paused
    fd = daily._acquire()
    assert fd is not None
    owner_ready.set()
    try:
        assert stop.wait(timeout=10)
    finally:
        daily._release(fd)


def _holding_owner(lock, ready):
    daily.LOCK = Path(lock)
    fd = daily._acquire()
    assert fd is not None
    ready.set()
    time.sleep(30)


def _orchestrator_with_model(root):
    root = Path(root)
    daily.ROOT = root
    daily.LOCK = root / "run.lock"
    fd = daily._acquire()
    assert fd is not None
    try:
        result = daily._run_process([sys.executable, str(root / "model.py")], "444", timeout_seconds=10)
        (root / "result.json").write_text(json.dumps(result))
    finally:
        daily._release(fd)


def _stop_workers(workers):
    for worker in workers:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=3)


@pytest.mark.skipif(os.name == "nt", reason="Production lock/process-group stress uses POSIX fork")
def test_twenty_four_simultaneous_writers_are_serialized_without_lost_work(tmp_path):
    context = mp.get_context("fork")
    gate = context.Barrier(25)
    active, violations, completed = (context.Value("i", 0) for _ in range(3))
    workers = [context.Process(target=_race_owner, args=(str(tmp_path / "run.lock"), gate, active, violations, completed))
               for _ in range(24)]
    try:
        for worker in workers:
            worker.start()
        gate.wait(timeout=15)
        deadline = time.monotonic() + 15
        for worker in workers:
            worker.join(timeout=max(.01, deadline - time.monotonic()))
        assert all(worker.exitcode == 0 for worker in workers)
        assert completed.value == 24 and active.value == 0 and violations.value == 0
    finally:
        _stop_workers(workers)


@pytest.mark.skipif(os.name == "nt", reason="Production lock stress uses POSIX fork")
def test_an_empty_file_cannot_be_stolen_before_the_owner_writes_its_pid(tmp_path, monkeypatch):
    context = mp.get_context("fork")
    before_write, allow_write, owner_ready, stop = (context.Event() for _ in range(4))
    monkeypatch.setattr(daily, "LOCK", tmp_path / "run.lock")
    owner = context.Process(target=_paused_owner, args=(str(daily.LOCK), before_write, allow_write, owner_ready, stop))
    owner.start()
    acquired = None
    try:
        assert before_write.wait(timeout=5)
        assert daily.LOCK.read_bytes() == b""
        acquired = daily._acquire()
        assert acquired is None
        allow_write.set()
        assert owner_ready.wait(timeout=5)
        assert daily._acquire() is None
        stop.set()
        owner.join(timeout=5)
        assert owner.exitcode == 0
    finally:
        if acquired is not None:
            daily._release(acquired)
        allow_write.set()
        stop.set()
        _stop_workers([owner])


@pytest.mark.parametrize("contents", [b"", b"broken pid", b"-1", b"0", b"9" * 4000, str(os.getpid()).encode()])
def test_diagnostic_pid_content_never_creates_a_permanent_busy_lock(tmp_path, monkeypatch, contents):
    monkeypatch.setattr(daily, "LOCK", tmp_path / "run.lock")
    daily.LOCK.write_bytes(contents)
    inode = daily.LOCK.stat().st_ino
    fd = daily._acquire()
    assert fd is not None
    try:
        assert daily._acquire() is None
        assert daily.LOCK.stat().st_ino == inode
    finally:
        daily._release(fd)
    assert daily.LOCK.is_file() and daily.LOCK.stat().st_ino == inode
    recovered = daily._acquire()
    assert recovered is not None
    daily._release(recovered)


def test_invalid_pid_is_never_passed_to_an_os_process_group_signal(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("Invalid PID reached the OS"))
    for invalid in (-1, 0, 2**128, "123", True, None):
        assert not daily._alive(invalid)


def test_releasing_an_old_descriptor_never_unlinks_a_replaced_path(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "LOCK", tmp_path / "run.lock")
    fd = daily._acquire()
    assert fd is not None
    replacement = tmp_path / "replacement.lock"
    replacement.write_bytes(b"replacement owned elsewhere")
    replacement.replace(daily.LOCK)
    daily._release(fd)
    assert daily.LOCK.read_bytes() == b"replacement owned elsewhere"


@pytest.mark.skipif(os.name == "nt", reason="Production lock stress uses POSIX fork")
def test_killing_the_owner_releases_the_os_lock_without_deleting_evidence(tmp_path, monkeypatch):
    context = mp.get_context("fork")
    ready = context.Event()
    monkeypatch.setattr(daily, "LOCK", tmp_path / "run.lock")
    owner = context.Process(target=_holding_owner, args=(str(daily.LOCK), ready))
    owner.start()
    try:
        assert ready.wait(timeout=5)
        assert daily._acquire() is None
        inode = daily.LOCK.stat().st_ino
        owner.kill()
        owner.join(timeout=5)
        fd = daily._acquire()
        assert fd is not None and daily.LOCK.stat().st_ino == inode
        daily._release(fd)
    finally:
        _stop_workers([owner])


def _writer_command(path, *, parent_wait=True, detached_stdio=False, ready_path=None):
    child = f"import time; from pathlib import Path; time.sleep(.8); Path({str(path)!r}).write_text('late write')"
    redirect = ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" if detached_stdio else ""
    command = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]{redirect}); print('child started',flush=True); "
    if ready_path is not None:
        command += f"from pathlib import Path; Path({str(ready_path)!r}).write_text('ready'); "
    if parent_wait:
        command += "time.sleep(30)"
    return [sys.executable, "-c", command]


@pytest.mark.skipif(os.name == "nt", reason="Production subprocess deadlines use POSIX process groups")
def test_timeout_stops_a_real_grandchild_before_it_can_write_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    late = tmp_path / "late.txt"
    started = time.monotonic()
    result = daily._run_process(_writer_command(late), "444", timeout_seconds=.2)
    assert result["exit_code"] == 1 and result["status"] == "failed"
    assert "timed out" in result["output"] and "child started" in result["output"]
    assert time.monotonic() - started < 3
    time.sleep(.9)
    assert not late.exists()
    assert (tmp_path / result["log"]).read_text() == result["output"]


@pytest.mark.skipif(os.name == "nt", reason="Production subprocess cleanup uses POSIX process groups")
def test_a_successful_parent_cannot_leave_a_detached_stdio_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    late = tmp_path / "late.txt"
    result = daily._run_process(_writer_command(late, parent_wait=False, detached_stdio=True), "444", timeout_seconds=2)
    assert result["exit_code"] == 0 and result["status"] == "success"
    time.sleep(.9)
    assert not late.exists()


@pytest.mark.skipif(os.name == "nt", reason="Production subprocess interruption uses POSIX process groups")
def test_interruption_cleans_up_the_real_process_tree_before_propagating(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    late, ready = tmp_path / "late.txt", tmp_path / "ready.txt"
    original = subprocess.Popen
    def interrupting(*args, **kwargs):
        proc = original(*args, **kwargs)
        communicate = proc.communicate
        called = False
        def once(*values, **options):
            nonlocal called
            if not called:
                called = True
                deadline = time.monotonic() + 3
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.005)
                assert ready.exists()
                raise KeyboardInterrupt()
            return communicate(*values, **options)
        proc.communicate = once
        return proc
    monkeypatch.setattr(subprocess, "Popen", interrupting)
    with pytest.raises(KeyboardInterrupt):
        daily._run_process(_writer_command(late, ready_path=ready), "444", timeout_seconds=2)
    time.sleep(.9)
    assert not late.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_process_deadline_fails_before_starting_a_command(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="positive finite"):
        daily._run_process([sys.executable, "-c", "pass"], "444", timeout_seconds=timeout)
    assert not (tmp_path / "logs").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Kernel parent-death supervision is Linux specific")
@pytest.mark.parametrize("target", ["orchestrator_sigkill", "supervisor_sigterm"])
def test_parent_death_and_supervisor_termination_stop_child_and_grandchild_writers(tmp_path, monkeypatch, target):
    (tmp_path / "grandchild.py").write_text(
        "import os,signal,time\nfrom pathlib import Path\nroot=Path(__file__).parent\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "(root/'grandchild.pid').write_text(str(os.getpid()))\ntime.sleep(1)\n"
        "(root/'grandchild-late.txt').write_text('late')\ntime.sleep(30)\n")
    (tmp_path / "model.py").write_text(
        "import os,signal,subprocess,sys,time\nfrom pathlib import Path\nroot=Path(__file__).parent\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "(root/'supervisor.pid').write_text(str(os.getppid()))\n"
        "(root/'model.pid').write_text(str(os.getpid()))\n"
        "subprocess.Popen([sys.executable,str(root/'grandchild.py')])\n"
        "while not (root/'grandchild.pid').exists(): time.sleep(.005)\n"
        "(root/'ready').write_text('ready')\ntime.sleep(1)\n"
        "(root/'model-late.txt').write_text('late')\ntime.sleep(30)\n")
    context = mp.get_context("fork")
    orchestrator = context.Process(target=_orchestrator_with_model, args=(str(tmp_path),))
    orchestrator.start()
    pids = []
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.005)
        assert (tmp_path / "ready").exists()
        pids = [int((tmp_path / name).read_text()) for name in ("supervisor.pid", "model.pid", "grandchild.pid")]
        if target == "orchestrator_sigkill":
            # Force the parent-death handler to remain pending. Its inherited
            # lock must prevent a replacement run during this exact window.
            os.kill(pids[0], signal.SIGSTOP)
            os.kill(orchestrator.pid, signal.SIGKILL)
        else:
            os.kill(pids[0], signal.SIGTERM)
        orchestrator.join(timeout=5)
        assert orchestrator.exitcode == (-signal.SIGKILL if target == "orchestrator_sigkill" else 0)
        if target == "supervisor_sigterm":
            result = json.loads((tmp_path / "result.json").read_text())
            assert result["status"] == "failed" and result["exit_code"] not in (0, 2)
        monkeypatch.setattr(daily, "LOCK", tmp_path / "run.lock")
        if target == "orchestrator_sigkill":
            assert daily._acquire() is None
            os.kill(pids[0], signal.SIGCONT)
        deadline = time.monotonic() + 3
        recovered = daily._acquire()
        while recovered is None and time.monotonic() < deadline:
            time.sleep(.005)
            recovered = daily._acquire()
        assert recovered is not None
        daily._release(recovered)
        time.sleep(1.1)
        assert not (tmp_path / "model-late.txt").exists()
        assert not (tmp_path / "grandchild-late.txt").exists()
        for pid in pids:
            stat = Path(f"/proc/{pid}/stat")
            # Orphan zombies can await the host's reaper; they cannot write.
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        _stop_workers([orchestrator])
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Kernel parent-death supervision is Linux specific")
def test_a_supervisor_with_an_already_dead_parent_cannot_start_the_model(tmp_path):
    helper = Path(daily.__file__).with_name("process_supervisor.py")
    marker = tmp_path / "must-not-start.txt"
    result = subprocess.run([sys.executable, str(helper), "--parent-pid", str(os.getpid() + 1000000), "--",
                             sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('started')"],
                            start_new_session=True, capture_output=True, timeout=3)
    assert result.returncode == -signal.SIGKILL and not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Kernel parent-death supervision is Linux specific")
def test_supervisor_refuses_to_signal_a_shared_job_process_group(tmp_path):
    helper = Path(daily.__file__).with_name("process_supervisor.py")
    result = subprocess.run([sys.executable, str(helper), "--parent-pid", str(os.getpid()), "--", sys.executable, "-c", "pass"],
                            capture_output=True, timeout=3)
    assert result.returncode != 0
    assert b"isolated process session" in result.stderr


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_supervision_preserves_model_success_error_and_waiting_codes(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    result = daily._run_process([sys.executable, "-c", f"raise SystemExit({exit_code})"], "444", timeout_seconds=2)
    assert result["exit_code"] == exit_code
    assert result["status"] == {0: "success", 1: "failed", 2: "waiting"}[exit_code]
