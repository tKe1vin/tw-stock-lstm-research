"""Tie a Linux model process group to the orchestrator's lifetime.

Run as a fresh session through run_all_daily. A kernel parent-death signal
terminates the complete owned process group, including grandchildren, even
when the orchestrator is killed before its Python cleanup can execute.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

LOCK_FDS_ENV = "TW_LSTM_LOCK_FDS"


def validated_lock_fds(descriptors):
    """Forward only explicit, open regular-file descriptors above stdio."""
    result = []
    for fd in descriptors:
        if not isinstance(fd, int) or isinstance(fd, bool) or fd <= 2:
            raise ValueError("Invalid inherited orchestrator lock descriptor")
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Inherited orchestrator lock must be a regular file")
        if fd not in result:
            result.append(fd)
    return tuple(result)


def inherited_lock_fds():
    value = os.environ.get(LOCK_FDS_ENV, "")
    if not value:
        return ()
    if not all(part.isascii() and part.isdecimal() for part in value.split(",")):
        raise ValueError("Invalid inherited orchestrator lock descriptor list")
    return validated_lock_fds(int(part) for part in value.split(","))


def supervised_command(command, expected_parent, lock_fds=()):
    descriptors = validated_lock_fds(lock_fds)
    options = [value for fd in descriptors for value in ("--lock-fd", str(fd))]
    return ([sys.executable, str(Path(__file__).resolve()), "--parent-pid", str(expected_parent),
             *options, "--", *command], descriptors)


def supervise(command, expected_parent, lock_fds=()):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Parent-death supervision requires Linux")
    owner = os.getpid()
    if os.getpgrp() != owner or os.getsid(0) != owner:
        raise RuntimeError("Supervisor requires its own isolated process session")
    if not isinstance(expected_parent, int) or expected_parent <= 1 or not command:
        raise ValueError("Supervisor requires a live parent PID and a command")
    descriptors = validated_lock_fds(lock_fds)

    def stop_owned_group(signum, _frame):
        # This handler is installed only after proving that the process owns a
        # separate session. Never signal the orchestrator or another job group.
        os.killpg(owner, signal.SIGKILL)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop_owned_group)
    signal.signal(signal.SIGINT, stop_owned_group)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    # Parent death between Popen and prctl otherwise binds the signal to a new
    # reaper rather than the intended orchestrator. Do not start the command.
    if os.getppid() != expected_parent:
        stop_owned_group(signal.SIGTERM, None)
    # Nested source workers create their own sessions. Keep both parent-death
    # protection and the same lock ownership available through that boundary.
    environment = {**os.environ, LOCK_FDS_ENV: ",".join(str(fd) for fd in descriptors)}
    child = subprocess.Popen(command, pass_fds=descriptors, env=environment)
    result = child.wait()
    # Preserve ordinary model status 0/1/2. A signal is a conventional fatal
    # shell status, never a successful or expected-waiting outcome.
    return result if result >= 0 else 128 - result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--lock-fd", action="append", type=int, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(command, args.parent_pid, args.lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
