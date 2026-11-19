"""Retained project results remain scoped to their saved protocols and dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

pytest.importorskip('streamlit')
pytest.importorskip('altair')
from streamlit.testing.v1 import AppTest

from dashboard.data import DataIntegrityError, ROOT
from dashboard.project import load_model_output, load_paired, load_tracker, model_dates, paired_dates


def copy_pair(tmp_path):
    relative = 'reports/daily/2026-11-16/paired'
    shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path / relative


def test_paired_history_keeps_each_period_and_identical_core_without_native_roi_mix():
    current = load_paired('2026-11-16')
    older = load_paired(root=ROOT, archived=True)
    assert current['start'] == '2026-04-09' and current['end'] == '2026-11-16'
    assert older['start'] == '2026-03-30' and older['end'] == '2026-11-06'
    assert current['protocol']['model_training_executed'] is False
    assert current['protocol']['news_evidence']['news_point_in_time_verified'] is False
    saved = pd.read_csv(ROOT / current['base'] / 'metrics.csv', encoding='utf-8-sig')
    pd.testing.assert_frame_equal(current['metrics'].drop(columns='組別'), saved)
    counts = current['nav'].groupby('Arm').Date.nunique()
    assert counts.nunique() == 1
    native = load_model_output('444', '2026-11-16')['tables']['native_strategy_metrics']['data']
    paired_roi = current['metrics'].set_index('Arm').loc['static_equal444', 'ROI_Pct']
    assert paired_roi != native.ROI_Pct.iloc[0]
    latest = load_paired(paired_dates()[0])
    latest_saved = pd.read_csv(ROOT / latest['base'] / 'metrics.csv', encoding='utf-8-sig')
    assert latest_saved.First_Execution_Date.nunique() == 1
    assert latest_saved.Last_Valuation_Date.nunique() == 1
    assert latest['start'] == latest_saved.First_Execution_Date.iloc[0]
    assert latest['end'] == latest_saved.Last_Valuation_Date.iloc[0]
    assert latest['end'] >= current['end']
    pd.testing.assert_frame_equal(latest['metrics'].drop(columns='組別'), latest_saved)


def test_changed_pair_fails_closed_and_never_replaces_archive(tmp_path):
    base = copy_pair(tmp_path)
    (base / 'metrics.csv').write_bytes((base / 'metrics.csv').read_bytes() + b'\n')
    with pytest.raises(DataIntegrityError, match='雜湊'):
        load_paired('2026-11-16', tmp_path)


def test_unaligned_pair_is_rejected_even_with_matching_local_hash(tmp_path):
    base = copy_pair(tmp_path)
    nav = pd.read_csv(base / 'nav.csv')
    nav = nav.iloc[1:]
    nav.to_csv(base / 'nav.csv', index=False)
    manifest = json.loads((base / 'reproduction_manifest.json').read_text())
    manifest['outputs_sha256']['nav.csv'] = hashlib.sha256((base / 'nav.csv').read_bytes()).hexdigest()
    (base / 'reproduction_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(DataIntegrityError, match='日期|觀察數'):
        load_paired('2026-11-16', tmp_path)


def test_forecast_only_and_missing_output_are_never_fabricated_as_realized_results():
    v2 = load_model_output('555_v2', model_dates('555_v2')[0])
    assert v2['status']['signal_published'] is False
    assert v2['status']['recommendation_execution_status'] == 'forecast_only'
    # Older exports can contain the erroneous eligible_unfilled flag on an
    # eligible rebalance day. Preserve the evidence and explicitly surface the
    # discrepancy; do not rewrite it to make the current test pass.
    signal = v2['tables']['unified_signal']
    saved = pd.read_csv(ROOT / signal['path'], encoding='utf-8-sig')
    pd.testing.assert_frame_equal(signal['data'], saved)
    audit = v2['execution_audit']
    assert audit['forecast_only']
    assert audit['saved_execution_statuses'] == sorted(set(saved.execution_status))
    assert audit['legacy_status_discrepancy'] == (not bool(saved.execution_status.isin(['forecast_only', 'forecast_expired']).all()))
    # The delayed 11/16 output remains maintenance-only even after a later
    # session has legitimately published a new signal.
    tracker = load_model_output('v211', '2026-11-16')
    assert tracker['status']['maintenance_successful'] is True
    assert tracker['status']['signal_published'] is False and tracker['tables'] == {}
    with pytest.raises(DataIntegrityError):
        load_model_output('444', '../../models')
    with pytest.raises(DataIntegrityError):
        load_model_output('444', '2099-01-01')


@pytest.mark.parametrize('saved_status', ['eligible_unfilled', 'forecast_only', 'forecast_expired'])
def test_candidate_status_discrepancy_is_visible_without_rewriting_saved_evidence(tmp_path, saved_status):
    base = tmp_path / 'outputs/555_v2/2026-11-19'
    base.mkdir(parents=True)
    (base / 'status.json').write_text(json.dumps({'signal_published': False,
        'recommendation_execution_status': 'forecast_only'}))
    path = base / 'unified_signal.csv'
    pd.DataFrame([{'model_id': '555_v2', 'execution_status': saved_status}]).to_csv(path, index=False)
    before = path.read_bytes()
    result = load_model_output('555_v2', '2026-11-19', tmp_path)
    assert result['execution_audit'] == {'forecast_only': True,
        'saved_execution_statuses': [saved_status], 'legacy_status_discrepancy': saved_status == 'eligible_unfilled'}
    assert result['tables']['unified_signal']['data'].execution_status.tolist() == [saved_status]
    assert path.read_bytes() == before


def test_tracker_retains_separate_signal_and_valuation_dates_and_checks_ledger(tmp_path):
    tracker = load_tracker()
    relative = 'experiments/results/v2/v211'
    signal_files = list((ROOT / relative / 'daily').glob('????-??-??.csv'))
    assert tracker['last_signal_date'] == max(path.stem for path in signal_files)
    assert tracker['signal_count'] == len(signal_files)
    assert tracker['historical_availability_certified'] is False
    tables = tracker['tables']
    saved_nav = pd.read_csv(ROOT / relative / 'live_nav.csv')
    pd.testing.assert_frame_equal(tables['live_nav']['data'], saved_nav)
    assert tables['live_nav']['data'].Date.max() == saved_nav.Date.max()
    matured = tables['matured_scores']['data']
    pd.testing.assert_frame_equal(matured, pd.read_csv(ROOT / relative / 'matured_scores.csv'))
    assert {'2026-11-05', '2026-11-06'}.issubset(set(matured.Signal_Date))
    assert matured.groupby('Signal_Date').size().eq(5).all()
    assert not matured.duplicated(['Signal_Date', 'Ticker']).any()
    for signal_day, part in matured.groupby('Signal_Date'):
        saved_signal = pd.read_csv(ROOT / relative / 'daily' / f'{signal_day}.csv')
        assert set(part.Ticker) == set(saved_signal.Ticker)
    base = tmp_path / relative
    base.mkdir(parents=True)
    for name in ['ledger_integrity.json', 'matured_scores.csv']:
        shutil.copyfile(ROOT / relative / name, base / name)
    (base / 'matured_scores.csv').write_text('Signal_Date,Ticker\n2026-11-05,fake\n')
    with pytest.raises(DataIntegrityError, match='完整性'):
        load_tracker(tmp_path)


def test_project_page_model_switch_and_saved_day_switch_are_read_only_without_models(monkeypatch):
    import requests

    def reject(*_args, **_kwargs):
        raise AssertionError('Presentation cannot request a provider or train')

    paths = [ROOT / 'reports/daily/2026-11-16/paired/metrics.csv', ROOT / 'reports/daily/2026-11-16/paired/nav.csv',
             ROOT / 'experiments/results/v2/v211/live_nav.csv', ROOT / 'experiments/results/v2/v211/weights/weights.npz']
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    monkeypatch.setattr(requests.sessions.Session, 'request', reject)
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    at = AppTest.from_string('from dashboard.project import render_project\nrender_project()', default_timeout=30).run()
    assert not at.exception and not at.error
    for model in ['555', '555_v2', 'v211', '444']:
        at.selectbox(key='project_model').set_value(model).run()
        assert not at.exception and not at.error
        if model == '555_v2':
            saved_day = model_dates('555_v2')[0]
            if load_model_output('555_v2', saved_day)['execution_audit']['legacy_status_discrepancy']:
                assert any('不視為可交易訊號或成交紀錄' in item.value for item in at.warning)
    at.selectbox(key='project_pair_day').set_value('2026-11-09').run()
    assert not at.exception and not at.error
    assert any('2026-03-31' in v.value and '2026-11-09' in v.value for v in at.caption)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
