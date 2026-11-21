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
import subprocess
import sys


def supervise(command, expected_parent):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Parent-death supervision requires Linux")
    owner = os.getpid()
    if os.getpgrp() != owner or os.getsid(0) != owner:
        raise RuntimeError("Supervisor requires its own isolated process session")
    if not isinstance(expected_parent, int) or expected_parent <= 1 or not command:
        raise ValueError("Supervisor requires a live parent PID and a command")

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
    child = subprocess.Popen(command)
    result = child.wait()
    # Preserve ordinary model status 0/1/2. A signal is a conventional fatal
    # shell status, never a successful or expected-waiting outcome.
    return result if result >= 0 else 128 - result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(command, args.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
