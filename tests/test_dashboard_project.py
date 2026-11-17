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
from streamlit.testing.v1 import AppTest

from dashboard.data import DataIntegrityError, ROOT
from dashboard.project import load_model_output, load_paired, load_tracker, model_dates, paired_dates


def copy_pair(tmp_path):
    relative = 'reports/daily/2026-11-16/paired'
    shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path / relative


def test_paired_history_keeps_each_period_and_identical_core_without_native_roi_mix():
    current = load_paired(paired_dates()[0])
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
    assert v2['tables']['unified_signal']['data'].execution_status.eq('forecast_only').all()
    tracker = load_model_output('v211', model_dates('v211')[0])
    assert tracker['status']['maintenance_successful'] is True
    assert tracker['status']['signal_published'] is False and tracker['tables'] == {}
    with pytest.raises(DataIntegrityError):
        load_model_output('444', '../../models')
    with pytest.raises(DataIntegrityError):
        load_model_output('444', '2099-01-01')


def test_tracker_retains_separate_signal_and_valuation_dates_and_checks_ledger(tmp_path):
    tracker = load_tracker()
    assert tracker['last_signal_date'] == '2026-11-10'
    assert tracker['historical_availability_certified'] is False
    tables = tracker['tables']
    assert tables['live_nav']['data'].Date.max() == '2026-11-16'
    assert len(tables['matured_scores']['data']) == 10
    assert tables['matured_scores']['data'].Signal_Date.nunique() == 2
    relative = 'experiments/results/v2/v211'
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
    at.selectbox(key='project_pair_day').set_value('2026-11-09').run()
    assert not at.exception and not at.error
    assert any('2026-03-31' in v.value and '2026-11-09' in v.value for v in at.caption)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
