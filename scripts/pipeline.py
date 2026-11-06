"""Compatibility entry point for the unified four-system daily pipeline."""
from __future__ import annotations
import argparse
from scripts.common import check_date_in_calendar, current_completed_session


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-fetch', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.as_of and check_date_in_calendar(args.as_of) != current_completed_session():
        raise ValueError('The daily pipeline only accepts the latest completed session; use explicit replay fixtures for history.')
    from scripts.run_all_daily import main as run_all
    forwarded = ['--mode', 'check' if args.check else 'full']
    if args.force:
        forwarded.append('--force')
    if args.no_fetch:
        forwarded.append('--skip-download')
    return run_all(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
