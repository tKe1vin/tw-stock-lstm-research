import json
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import run_all_daily as daily

TZ = ZoneInfo('Asia/Taipei')


def setup_run(monkeypatch, tmp_path):
    monkeypatch.setattr(daily, 'ROOT', tmp_path)
    monkeypatch.setattr(daily, 'LOCK', tmp_path / 'logs/run.lock')
    monkeypatch.setattr(daily, 'current_completed_session', lambda now: __import__('pandas').Timestamp('2026-11-06'))
    import scripts.daily_reports as reports
    monkeypatch.setattr(reports, 'build_reports', lambda summary, day: {'status': 'success'})


def test_market_once_before_v211_then_independent_models(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(daily, '_update_market', lambda: calls.append('market') or {'exit_code': 0})
    monkeypatch.setattr(daily, '_run_one', lambda model, mode: calls.append(model) or {'status': 'success', 'exit_code': 0})
    assert daily.main([], now=datetime(2026, 11, 6, 17, 37, tzinfo=TZ)) == 0
    assert calls == ['market', 'v211', '555_v2', '444', '555']
    assert not daily.LOCK.exists()


def test_partial_failure_keeps_running_and_writes_truthful_summary(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(daily, '_update_market', lambda: {'exit_code': 2})
    def run(model, mode):
        calls.append(model)
        if model == '444':
            raise RuntimeError('training failed')
        return {'status': 'waiting' if model == '555' else 'success', 'exit_code': 2 if model == '555' else 0}
    monkeypatch.setattr(daily, '_run_one', run)
    assert daily.main([], now=datetime(2026, 11, 6, 17, 37, tzinfo=TZ)) == 1
    summary = json.loads((tmp_path/'outputs/daily_summary/2026-11-06/summary.json').read_text())
    assert len(calls) == 4 and summary['steps']['v211']['status'] == 'success'
    assert summary['steps']['555']['status'] == 'waiting'
    assert summary['status'] == 'failed'


def test_before_close_or_holiday_does_not_publish_old_signals(monkeypatch, tmp_path):
    monkeypatch.setattr(daily, 'ROOT', tmp_path)
    monkeypatch.setattr(daily, 'LOCK', tmp_path/'logs/run.lock')
    monkeypatch.setattr(daily, 'current_completed_session', lambda now: __import__('pandas').Timestamp('2026-11-05'))
    monkeypatch.setattr(daily, '_update_market', lambda: (_ for _ in ()).throw(AssertionError('must not fetch')))
    assert daily.main([], now=datetime(2026, 11, 6, 13, 0, tzinfo=TZ)) == 0
    assert json.loads((tmp_path/'outputs/daily_summary/2026-11-06/summary.json').read_text())['status'] == 'skipped'


def test_successful_exports_skipped_on_retry_but_v211_maintenance_runs(monkeypatch, tmp_path):
    setup_run(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(daily, '_update_market', lambda: {'exit_code': 0})
    monkeypatch.setattr(daily, '_completed_model', lambda model, session, day: None if model == 'v211' else {'status': 'success', 'exit_code': 0})
    monkeypatch.setattr(daily, '_run_one', lambda model, mode: calls.append(model) or {'status': 'success', 'exit_code': 0})
    assert daily.main([], now=datetime(2026, 11, 6, 21, 37, tzinfo=TZ)) == 0
    assert calls == ['v211']
