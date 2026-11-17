"""Delayed Actions recover research, preserving real times and signal boundaries."""
from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_all_daily as daily
from scripts import run_model as runner
from scripts.common import TZ


def context(monkeypatch, started, session):
    monkeypatch.setenv('TW_LSTM_TARGET_SESSION', session)
    monkeypatch.setenv('TW_LSTM_OUTPUT_DAY', session)
    monkeypatch.setenv('TW_LSTM_RUN_STARTED_AT', started.isoformat())


def test_context_freezes_asof_when_slow_job_crosses_next_close(monkeypatch):
    started = datetime(2026, 11, 17, 12, 0, tzinfo=TZ)
    context(monkeypatch, started, '2026-11-16')
    actual = started + timedelta(hours=4)
    now, today, selected = runner.run_day(actual)
    assert now == actual and today == pd.Timestamp('2026-11-17')
    assert selected == pd.Timestamp('2026-11-16')
    assert runner._output_day(today, selected) == selected


@pytest.mark.parametrize('kind', ['older', 'future', 'expired', 'future_clock'])
def test_context_cannot_select_arbitrary_backfill_or_future_session(monkeypatch, kind):
    started = datetime(2026, 11, 17, 12, 0, tzinfo=TZ)
    context(monkeypatch, started, '2026-11-16')
    actual = started + timedelta(minutes=1)
    if kind == 'older': monkeypatch.setenv('TW_LSTM_TARGET_SESSION', '2026-11-13')
    if kind == 'future': monkeypatch.setenv('TW_LSTM_TARGET_SESSION', '2026-11-17')
    if kind == 'expired': actual += timedelta(hours=7)
    if kind == 'future_clock': actual = started - timedelta(minutes=1)
    with pytest.raises(ValueError): runner.run_day(actual)


def test_two_sessions_generated_on_one_calendar_day_keep_distinct_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    today = pd.Timestamp('2026-11-17')
    for session in ('2026-11-16', '2026-11-17'):
        monkeypatch.setenv('TW_LSTM_OUTPUT_DAY', session)
        runner._write_status('444', today, {'completed_session': session, 'run_date': '2026-11-17', 'status': 'success'})
    early = json.loads((tmp_path/'outputs/444/2026-11-16/status.json').read_text())
    evening = json.loads((tmp_path/'outputs/444/2026-11-17/status.json').read_text())
    assert early['completed_session'] == '2026-11-16'
    assert evening['completed_session'] == '2026-11-17'
    assert early['run_date'] == evening['run_date'] == '2026-11-17'


def test_generated_after_entry_close_is_expired_even_before_15h_data_cutoff():
    assert not runner._entry_expired('2026-11-17', datetime(2026,11,17,13,29,59,tzinfo=TZ))
    assert runner._entry_expired('2026-11-17', datetime(2026,11,17,13,30,tzinfo=TZ))
    assert runner._entry_expired('2026-11-17', datetime(2026,11,17,14,0,tzinfo=TZ))


def test_delayed_v211_launches_only_recorded_maintenance(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    selected = pd.Timestamp('2026-11-16')
    now = datetime(2026,11,17,12,0,tzinfo=TZ)
    monkeypatch.setattr(runner, 'run_day', lambda: (now, pd.Timestamp('2026-11-17'), selected))
    monkeypatch.setenv('TW_LSTM_OUTPUT_DAY','2026-11-16')
    monkeypatch.setattr(runner, '_frozen_card', lambda: ('sha', {}))
    original = tmp_path/'experiments/results/v2/v211/daily/2026-11-05.csv'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original-published-signal')
    monkeypatch.setattr(runner, 'EXPECTED_SIGNAL', runner.sha256(original))
    seen = []
    def process(command, **kwargs):
        seen.append(command)
        assert '-c' in command and 'update_recorded_results' in command[-1]
        assert command != [runner.sys.executable, '-m', 'experiments.v2.v211_daily']
        return SimpleNamespace(returncode=0, stdout='{"scoring":{},"book":{}}', stderr='')
    monkeypatch.setattr(runner.subprocess, 'run', process)
    assert runner.run_v211('full') == 2
    payload = json.loads((tmp_path/'outputs/v211/2026-11-16/status.json').read_text())
    assert payload['status'] == 'maintenance_only'
    assert payload['maintenance_successful'] and payload['signal_published'] is False
    assert original.read_bytes() == b'original-published-signal'
    assert not (original.parent/'2026-11-16.csv').exists()
    assert len(seen) == 1


def test_delayed_retry_reuses_same_session_even_on_later_calendar_day(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, 'ROOT', tmp_path)
    folder=tmp_path/'outputs/555/2026-11-16'
    folder.mkdir(parents=True)
    (folder/'unified_signal.csv').write_text('fixture')
    (folder/'status.json').write_text(json.dumps({'status':'success','completed_session':'2026-11-16',
        'unified_csv':'outputs/555/2026-11-16/unified_signal.csv','research_protocol':'paired-444-core-allocation-v1'}))
    assert daily._completed_model('555','2026-11-16','2026-11-17')['reason']=='already_completed_for_this_session'
    assert daily._completed_model('555','2026-11-17','2026-11-17') is None


def test_maintenance_only_step_preserves_distinct_status_from_child_json(monkeypatch):
    payload={'model':'v211','status':'maintenance_only','reason':'delayed_session_maintenance_only',
             'maintenance_successful':True,'signal_published':False}
    monkeypatch.setattr(daily,'_run_process',lambda *args,**kwargs:{'exit_code':2,'status':'waiting','output':json.dumps(payload)})
    step=daily._run_one('v211','full')
    assert step['status']=='maintenance_only' and step['exit_code']==2
    assert step['maintenance_successful'] and step['signal_published'] is False


def test_weekend_retry_recovers_friday_research_and_real_market_errors_still_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(daily,'ROOT',tmp_path)
    monkeypatch.setattr(daily,'LOCK',tmp_path/'logs/run.lock')
    monkeypatch.setattr(daily,'_completed_model',lambda *args:None)
    monkeypatch.setattr(daily,'_update_market',lambda:{'exit_code':0})
    def run(model,mode):
        return ({'status':'maintenance_only','exit_code':2,'reason':'delayed_session_maintenance_only',
                 'maintenance_successful':True,'signal_published':False} if model=='v211'
                else {'status':'success','exit_code':0})
    monkeypatch.setattr(daily,'_run_one',run)
    import scripts.daily_reports as reports
    monkeypatch.setattr(reports,'build_reports',lambda *args:{'status':'partial'})
    actual=datetime(2026,11,14,4,0,tzinfo=TZ)
    assert daily.main([],now=actual)==0
    summary=json.loads((tmp_path/'outputs/daily_summary/2026-11-14/summary.json').read_text())
    assert summary['completed_session']==summary['output_day']=='2026-11-13'
    assert summary['run_date']=='2026-11-14'
    assert summary['status']=='research_complete_tracker_deferred'
    assert (tmp_path/'logs/orchestrator/latest_summary.json').read_bytes()==(tmp_path/'outputs/daily_summary/2026-11-14/summary.json').read_bytes()
    monkeypatch.setattr(daily,'_update_market',lambda:(_ for _ in ()).throw(RuntimeError('source failed')))
    assert daily.main([],now=actual)==1
    assert not daily._RUN_CONTEXT


def test_expired_delayed_notebook_export_retains_source_and_flags_both_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    out=tmp_path/'outputs/444/2026-11-16'
    original=out/'runs/retained'
    original.mkdir(parents=True)
    frame=pd.DataFrame([{'Ticker':ticker,'Signal_Date':'2026-11-16','Target_Date':'2026-11-17',
                         'Created_At':'2026-11-17T14:00:00+08:00','Predicted_Return_Pct':1.0}
                        for ticker in runner.NAMES])
    source=original/'next_day_recommendation.csv'
    frame.to_csv(source,index=False)
    before=source.read_bytes()
    for name in ('strategy_metrics.csv','evaluation_protocol.json'):
        (original/name).write_text('{}')
    prices=tmp_path/'prices.csv';prices.write_text('source-hash-input')
    payload={'run_date':'2026-11-17','delayed_run':True,'execution_assumption':'research_only'}
    runner._export_notebook_csv('444',out,pd.Timestamp('2026-11-16'),prices,payload)
    unified=pd.read_csv(out/'unified_signal.csv',keep_default_na=False)
    native=pd.read_csv(out/'native/next_day_recommendation.csv')
    assert unified.execution_status.eq('delayed_research_target_expired').all()
    assert not unified.trade_eligible.any()
    assert unified.intended_execution_date.eq(runner.NOT_APPLICABLE).all()
    assert native.Entry_Window_Expired.all() and not native.Trade_Eligible.any()
    assert source.read_bytes()==before
    assert payload['forecast_timing']=='expired_entry'


def test_expired_history_is_diagnostic_and_overwritten_prior_bytes_are_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    history=tmp_path/'outputs/444/recommendation_history'
    history.mkdir(parents=True)
    retained=history/'same-name.csv'
    retained.write_bytes(b'original-published-bytes')
    before={retained:retained.read_bytes()}
    expired=b'Target_Date,Created_At\n2026-11-17,2026-11-17T14:00:00+08:00\n'
    retained.write_bytes(expired)
    new=history/'new-expired.csv';new.write_bytes(expired)
    archived=runner._finish_delayed_history(history,before,'444')
    assert retained.read_bytes()==b'original-published-bytes'
    assert not new.exists()
    assert len(archived)==2
    assert all((tmp_path/name).read_bytes()==expired for name in archived)
    assert all('/expired_recommendation_history/' in name for name in archived)


def test_legacy_calendar_keyed_444_is_only_reused_for_its_matching_asof(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(daily,'ROOT',tmp_path)
    monkeypatch.setenv('TW_LSTM_OUTPUT_DAY','2026-11-16')
    folder=tmp_path/'outputs/444/2026-11-17'
    native=folder/'native';native.mkdir(parents=True)
    for name in ('test_predictions.csv','next_day_recommendation.csv','model_metrics.csv','strategy_metrics.csv','backtest_nav.csv','unified_signal.csv'):
        (native/name).write_text('fixture')
    (native/'evaluation_protocol.json').write_text(json.dumps({'as_of_date':'2026-11-16'}))
    status={'status':'success','completed_session':'2026-11-16',
            'daily_native_dir':'outputs/444/2026-11-17/native',
            'unified_csv':'outputs/444/2026-11-17/native/unified_signal.csv'}
    (folder/'status.json').write_text(json.dumps(status))
    path,payload=runner._current_444_core(pd.Timestamp('2026-11-17'),pd.Timestamp('2026-11-16'))
    assert path==native and payload['completed_session']=='2026-11-16'
    assert daily._completed_model('444','2026-11-16','2026-11-17')
    (native/'test_predictions.csv').unlink()
    assert daily._completed_model('444','2026-11-16','2026-11-17') is None
    assert runner._current_444_core(pd.Timestamp('2026-11-17'),pd.Timestamp('2026-11-16'))[1]=={}
