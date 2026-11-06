"""Deterministic read-only checks and notebook output validation."""
from __future__ import annotations

import builtins
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts import run_model as runner

DAY = pd.Timestamp('2026-11-06')
NOW = datetime(2026, 11, 6, 17, 0, tzinfo=ZoneInfo('Asia/Taipei'))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'run_day', lambda now=None: (now or NOW, DAY, DAY))
    return tmp_path


def write_sources(folder):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Date': str(DAY.date()), 'ticker': ticker, 'Close': 100.0}
                  for ticker in [*runner.NAMES, '^TWII']]).to_csv(folder / 'prices.csv', index=False)
    pd.DataFrame([{'Date': str(DAY.date()), 'ticker': ticker, 'Daily_Sentiment': 0.0,
                   'News_Count': 0, 'Latest_News_Timestamp': None,
                   'Source_Status': 'verified_no_matching_article', 'Scoring_Model': 'neutral_no_news'}
                  for ticker in runner.NAMES]).to_csv(folder / 'sentiment.csv', index=False)


def frozen_files(root, monkeypatch):
    ledger = root / 'experiments/results/v2/v211'
    folder = ledger / 'weights'
    folder.mkdir(parents=True)
    (folder / 'weights.npz').write_bytes(b'test-frozen-weights')
    digest = hashlib.sha256(b'test-frozen-weights').hexdigest()
    (folder / 'model_card.json').write_text(json.dumps({'weight_sha256': digest}))
    (ledger / 'daily').mkdir()
    (ledger / 'daily/2026-11-05.csv').write_bytes(b'imported-signal')
    monkeypatch.setattr(runner, 'EXPECTED_WEIGHTS', digest)
    monkeypatch.setattr(runner, 'EXPECTED_SIGNAL', hashlib.sha256(b'imported-signal').hexdigest())
    return ledger


def file_snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_all_check_entry_points_are_read_only_and_do_not_import_tensorflow(isolated, monkeypatch):
    root = isolated
    write_sources(root / 'data/snapshots/2026-11-06')
    ledger = frozen_files(root, monkeypatch)
    write_sources(ledger / 'market')
    before = file_snapshot(root)
    original_import = builtins.__import__

    def no_tensorflow(name, *args, **kwargs):
        if name == 'tensorflow' or name.startswith('tensorflow.'):
            raise AssertionError('check must not load TensorFlow')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_tensorflow)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('check launched subprocess'))
    monkeypatch.setattr(runner, '_write_status', lambda *a, **kw: pytest.fail('check wrote status'))
    for model in ('444', '555', '555_v2', 'v211'):
        assert runner.inspect_model(model, NOW)['ready'] is True
        assert runner.run(model, 'check') == 0
    assert runner.run_notebook('444', 'check') == 0
    assert runner.run_notebook('555', 'check') == 0
    assert runner.run_555_v2('check') == 0
    assert runner.run_v211('check') == 0
    assert file_snapshot(root) == before


def test_missing_inputs_check_returns_waiting_without_creating_dirs(isolated):
    before = list(isolated.iterdir())
    assert runner.inspect_model('444')['exit_code'] == 2
    assert runner.run_notebook('555', 'check') == 2
    assert list(isolated.iterdir()) == before


def test_check_rejects_incomplete_or_duplicate_current_prices(isolated):
    folder = isolated / 'data/snapshots/2026-11-06'
    write_sources(folder)
    prices = pd.read_csv(folder / 'prices.csv')
    pd.concat([prices, prices.iloc[:1]]).to_csv(folder / 'prices.csv', index=False)
    status = runner.inspect_model('444')
    assert status['ready'] is False
    assert status['price_reason'] == 'prices_incomplete'


def test_check_rejects_tampered_frozen_weights_without_training(isolated, monkeypatch):
    ledger = frozen_files(isolated, monkeypatch)
    write_sources(ledger / 'market')
    (ledger / 'weights/weights.npz').write_bytes(b'changed')
    assert runner.inspect_model('555_v2')['exit_code'] == 1
    with pytest.raises(ValueError, match='training is prohibited'):
        runner._frozen_card()


def test_parent_skip_market_update_flag_is_preserved(isolated, monkeypatch):
    frozen_files(isolated, monkeypatch)
    monkeypatch.setenv('TW_LSTM_SKIP_MARKET_UPDATE', '1')
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs['env'])
        return SimpleNamespace(stdout='{}', stderr='', returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    assert runner.run_v211('full') == 0
    assert seen['TW_LSTM_SKIP_MARKET_UPDATE'] == '1'


def recommendation(folder, day='2026-11-06', tickers=None):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Ticker': ticker, 'Signal_Date': day, 'Target_Date': '2026-11-09',
                   'Predicted_Return_Pct': 0.1, 'Target_Weight': 0.2, 'Cash_Weight': 0.0}
                  for ticker in (list(runner.NAMES) if tickers is None else tickers)]).to_csv(folder / 'next_day_recommendation.csv', index=False)
    (folder / 'evaluation_protocol.json').write_text('{}')
    (folder / 'strategy_metrics.csv').write_text('ROI_Pct\n1.0\n')


def test_notebook_export_records_native_and_compact_directories(isolated):
    out = isolated / 'outputs/555/2026-11-06'
    native = out / 'runs/current'
    recommendation(native)
    price = isolated / 'price.csv'
    price.write_text('test-price-snapshot')
    payload = {'run_date': '2026-11-06', 'execution_assumption': 'test'}
    runner._export_notebook_csv('555', out, DAY, price, payload)
    assert payload['native_output_dir'] == 'outputs/555/2026-11-06/runs/current'
    assert payload['daily_native_dir'] == 'outputs/555/2026-11-06/native'
    assert (isolated / payload['daily_native_dir'] / 'strategy_metrics.csv').exists()
    assert len(pd.read_csv(isolated / payload['unified_csv'])) == 5


@pytest.mark.parametrize('day,tickers', [('2026-11-05', None), ('2026-11-06', ['2330.TW'] * 5)])
def test_notebook_export_rejects_stale_or_duplicate_stocks(isolated, day, tickers):
    out = isolated / 'outputs/444/2026-11-06'
    recommendation(out, day, tickers)
    with pytest.raises(ValueError):
        runner._export_notebook_csv('444', out, DAY, isolated / 'unused.csv', {'run_date': '2026-11-06'})
    assert not (out / 'unified_signal.csv').exists()


def test_no_recommendation_is_a_failed_run(isolated, monkeypatch):
    write_sources(isolated / 'data/snapshots/2026-11-06')
    import scripts.execute_nb
    monkeypatch.setattr(scripts.execute_nb, 'execute', lambda kind: None)
    assert runner.run_notebook('444', 'full') == 1
    payload = json.loads((isolated / 'outputs/444/2026-11-06/status.json').read_text())
    assert payload['status'] == 'failed'
    assert payload['forecast_produced'] is False


def test_existing_same_day_recommendation_is_not_a_new_success(isolated, monkeypatch):
    write_sources(isolated / 'data/snapshots/2026-11-06')
    out = isolated / 'outputs/444/2026-11-06'
    recommendation(out)
    import scripts.execute_nb
    monkeypatch.setattr(scripts.execute_nb, 'execute', lambda kind: None)
    assert runner.run_notebook('444', 'full') == 1
    payload = json.loads((out / 'status.json').read_text())
    assert payload['status'] == 'failed'
    assert 'no current' in payload['reason']


def mirror_sources(root, monkeypatch):
    ledger = frozen_files(root, monkeypatch)
    market = ledger / 'market'
    write_sources(market)
    for name, source in [('prices_seed.csv', 'prices.csv'), ('sentiment_seed.csv', 'sentiment.csv')]:
        (market / name).write_bytes((market / source).read_bytes())
    (market / 'manifest.json').write_text('{"seed":"fixture"}')
    (market / 'state.json').write_text('{"sessions":{"2026-11-06":{}}}')
    (market / 'news_cache.csv').write_text('news_id,ticker\nfixture,2330.TW\n')
    (market / 'news_scan_coverage.json').write_text('{"2026-11-06":true}')
    archived = root / 'research/v211/signal_registry/developer_original/2026-11-05.csv'
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b'imported-signal')
    return market


def append_market_day(market, date):
    for name in ['prices.csv', 'sentiment.csv']:
        frame = pd.read_csv(market / name)
        extra = frame.copy()
        extra['Date'] = date
        pd.concat([frame, extra], ignore_index=True).to_csv(market / name, index=False)
    (market / 'state.json').write_text(json.dumps({'sessions': {'2026-11-06': {}, date: {}}}))
    (market / 'news_scan_coverage.json').write_text(json.dumps({'2026-11-06': True, date: True}))


def test_independent_market_mirror_refreshes_without_importing_or_replacing_decisions(isolated, monkeypatch):
    source = mirror_sources(isolated, monkeypatch)
    dest = runner.prepare_independent_ledger()
    assert list((dest / 'daily').iterdir()) == []
    assert (dest / 'market/prices.csv').read_bytes() == (source / 'prices.csv').read_bytes()
    assert (dest / 'market/news_scan_coverage.json').read_bytes() == (source / 'news_scan_coverage.json').read_bytes()
    independent_signal = dest / 'daily/2026-11-06.csv'
    independent_signal.write_bytes(b'independent-decision')
    independent_book = dest / 'live_book.csv'
    independent_book.write_bytes(b'independent-book')
    origin_before = (dest / 'LEDGER_ORIGIN.json').read_bytes()
    append_market_day(source, '2026-11-09')
    assert runner.prepare_independent_ledger() == dest
    assert pd.read_csv(dest / 'market/prices.csv').Date.max() == '2026-11-09'
    assert (dest / 'market/state.json').read_bytes() == (source / 'state.json').read_bytes()
    assert independent_signal.read_bytes() == b'independent-decision'
    assert independent_book.read_bytes() == b'independent-book'
    assert (dest / 'LEDGER_ORIGIN.json').read_bytes() == origin_before


@pytest.mark.parametrize('kind', ['divergent', 'newer'])
def test_independent_market_mirror_rejects_divergent_or_newer_data_before_any_copy(isolated, monkeypatch, kind):
    mirror_sources(isolated, monkeypatch)
    dest = runner.prepare_independent_ledger()
    market = dest / 'market'
    if kind == 'newer':
        append_market_day(market, '2026-11-09')
    else:
        frame = pd.read_csv(market / 'prices.csv')
        frame.loc[0, 'Close'] = 123.45
        frame.to_csv(market / 'prices.csv', index=False)
    before = file_snapshot(dest)
    with pytest.raises(ValueError, match='newer than or divergent'):
        runner.prepare_independent_ledger()
    assert file_snapshot(dest) == before


def test_live_notebook_history_uses_stable_model_directory(isolated, monkeypatch):
    write_sources(isolated / 'data/snapshots/2026-11-06')
    import scripts.execute_nb
    seen = {}

    def fake_execute(kind):
        import os
        seen['history'] = os.environ['LSTM_RECOMMENDATION_HISTORY_DIR']

    monkeypatch.setattr(scripts.execute_nb, 'execute', fake_execute)
    assert runner.run_notebook('555', 'full') == 1  # No recommendation in this mocked run.
    assert seen['history'] == str(isolated / 'outputs/555/recommendation_history')
