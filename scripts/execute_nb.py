"""Run original notebook code cells as separate processes; never merge model source."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import nbformat
from scripts.common import ROOT


def execute(kind):
    import matplotlib
    matplotlib.use('Agg')
    name = '444_baseline.ipynb' if kind=='444' else '555_multivariate.ipynb'
    notebook=nbformat.read(ROOT/'notebooks'/name,as_version=4)
    ns={'__name__':'__main__','__file__':str(ROOT/'notebooks'/name)}
    # 444 calculations / export are kept; the six-model, 30-day scenario and
    # heavy display-only sections are not necessary for daily one-day forecast.
    skip_444={18,20,22,24,26,28,30,32,36}
    import os
    os.chdir(ROOT)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code' or cell.source.lstrip().startswith(('%pip','!pip','!')):
            continue
        if kind=='444' and index in skip_444:
            continue
        print(f'[{kind}] notebook cell {index}',flush=True)
        exec(compile(cell.source,str(ROOT/'notebooks'/name)+f':cell{index}','exec'),ns)
    if kind=='444':
        from scripts.baseline_adapter import finalize
        finalize(ns)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['444','555'])
    args=p.parse_args()
    execute(args.kind)
