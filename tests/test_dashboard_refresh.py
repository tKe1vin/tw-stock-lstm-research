"""A connected page rereads snapshots without recursive reruns or side effects."""
from __future__ import annotations

import builtins
from datetime import datetime
import importlib.util
import io
from pathlib import Path
import socket

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class FullAppRerun(BaseException):
    """Emulate Streamlit stopping the current run to restart the whole page."""


def refresh_body(monkeypatch):
    """Keep the real body and emulate timer dispatch without waiting five minutes."""
    intervals, captions, reruns = [], [], []

    def fragment(*, run_every):
        intervals.append(run_every)
        return lambda function: function

    def rerun(*, scope):
        reruns.append(scope)
        raise FullAppRerun

    monkeypatch.setattr(st, "fragment", fragment)
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(st, "caption", captions.append)
    monkeypatch.setattr(st, "rerun", rerun)
    spec = importlib.util.spec_from_file_location("isolated_dashboard_refresh", ROOT / "dashboard/refresh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, intervals, captions, reruns


def test_initial_and_user_initiated_full_renders_never_start_a_rerun_loop(monkeypatch):
    module, intervals, captions, reruns = refresh_body(monkeypatch)
    for render in ("initial", "manual-refresh", "navigate-news", "navigate-journals"):
        module.automatic_refresh(render)
    assert intervals == [300]
    assert not reruns
    assert sum("每5分鐘" in item for item in captions) == 4


def test_timer_stops_for_one_full_app_rerun_then_new_render_resets(monkeypatch):
    module, _intervals, _captions, reruns = refresh_body(monkeypatch)
    module.automatic_refresh("page-a")
    with pytest.raises(FullAppRerun):
        module.automatic_refresh("page-a")
    assert reruns == ["app"]
    # The fresh ID produced by a complete page run must not cause another rerun.
    module.automatic_refresh("page-b")
    assert reruns == ["app"]
    with pytest.raises(FullAppRerun):
        module.automatic_refresh("page-b")
    assert reruns == ["app", "app"]
    module.automatic_refresh("page-c")
    assert reruns == ["app", "app"]


def test_refresh_clock_uses_taipei_time_and_never_writes_or_calls_a_provider(monkeypatch):
    import requests

    module, _intervals, captions, _reruns = refresh_body(monkeypatch)
    times = iter((datetime(2026, 11, 17, 15), datetime(2026, 11, 17, 15, 5)))

    class Clock:
        @staticmethod
        def now(zone):
            assert str(zone) == "Asia/Taipei"
            return next(times).replace(tzinfo=zone)

    monkeypatch.setattr(module, "datetime", Clock)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Refreshing a view cannot fetch providers or modify files")

    for name in ("write_text", "write_bytes", "touch", "mkdir", "unlink", "rename", "replace"):
        monkeypatch.setattr(Path, name, forbidden)
    for target in (builtins, io):
        original = target.open

        def read_only(file, mode="r", *args, _open=original, **kwargs):
            if any(flag in mode for flag in "wax+"):
                forbidden()
            return _open(file, mode, *args, **kwargs)

        monkeypatch.setattr(target, "open", read_only)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    module.automatic_refresh("first-render")
    with pytest.raises(FullAppRerun):
        module.automatic_refresh("first-render")
    module.automatic_refresh("next-render")
    clocks = [item for item in captions if item.startswith("本頁讀取：")]
    assert clocks == ["本頁讀取：2026-11-17 15:00:00（台北）", "本頁讀取：2026-11-17 15:05:00（台北）"]


def test_actual_page_remains_usable_after_manual_refresh_and_navigation():
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("每5分鐘" in caption.value for caption in at.caption)
    assert any(caption.value.startswith("本頁讀取：") for caption in at.caption)
    button = next(button for button in at.button if button.label == "重新讀取保存報告")
    button.click().run()
    assert not at.exception
    at.radio(key="project_page").set_value("新聞與情緒").run()
    assert not at.exception and not at.error
    assert at.radio(key="project_page").value == "新聞與情緒"
