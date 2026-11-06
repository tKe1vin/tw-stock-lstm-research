"""Exercise the original notebook's review function against durable prior-day data."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import ROOT, TICKERS, sessions


def test_555_reviews_yesterdays_saved_forecast_from_stable_history(tmp_path, monkeypatch):
    history = tmp_path / 'outputs/555/recommendation_history'
    history.mkdir(parents=True)
    pd.DataFrame([{
        'Created_At': '2026-11-05T17:37:00+08:00', 'Run_ID': 'yesterday',
        'Run_Mode': 'daily', 'Signal_Date': '2026-11-05', 'Target_Date': '2026-11-06',
        'Ticker': ticker, 'Predicted_Return_Pct': 2., 'Target_Weight': .2,
        'Cash_Weight': 0.,
    } for ticker in TICKERS]).to_csv(history/'yesterday.csv', index=False)
    monkeypatch.setenv('LSTM_RECOMMENDATION_HISTORY_DIR', str(history))
    notebook = json.loads((ROOT/'notebooks/555_multivariate.ipynb').read_text())
    code = next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and 'def review_past_recommendations' in ''.join(c['source']))
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
                 and (isinstance(node, ast.FunctionDef) or any(isinstance(t, ast.Name) and t.id in {'REVIEW_COLUMNS', 'PORTFOLIO_REVIEW_COLUMNS'} for t in node.targets))]
    panel = pd.DataFrame({ticker: [100., 110.] for ticker in TICKERS},
                         index=pd.to_datetime(['2026-11-05', '2026-11-06']))
    namespace = {'Path': Path, 'os': os, 'pd': pd, 'np': np, 'TICKERS': TICKERS,
                 'OUTPUT_ROOT': tmp_path/'outputs/555/2026-11-06',
                 'PRICE_PANEL': panel, 'RESOLVED_AS_OF': pd.Timestamp('2026-11-06'),
                 'CREATED_AT': pd.Timestamp('2026-11-06T17:37:00+08:00'),
                 'trading_sessions': sessions, 'direction': lambda values: np.sign(values)}
    exec(compile(tree, 'notebook_history_review', 'exec'), namespace)
    stocks, portfolio = namespace['review_past_recommendations']()
    assert len(stocks) == 5 and set(stocks.Review_Status) == {'Reviewed'}
    assert portfolio.Run_ID.tolist() == ['yesterday']
    assert abs(portfolio.Actual_Gross_Return_Pct.iloc[0] - 10.) < 1e-10
