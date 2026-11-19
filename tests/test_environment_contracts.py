"""Keep production checks independent of the optional presentation runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _requirements(relative):
    """Read requirement specifications, including relative -r files."""
    path = ROOT / relative
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            result.update(_requirements((path.parent / line[3:]).relative_to(ROOT)))
            continue
        requirement = Requirement(line)
        result[canonicalize_name(requirement.name)] = requirement
    return result


def test_ci_pins_satisfy_declared_production_requirements():
    locked = _requirements("requirements-ci.txt")
    for name, required in _requirements("requirements.txt").items():
        assert name in locked, f"Production requirement missing from CI lock: {name}"
        pins = list(locked[name].specifier)
        assert len(pins) == 1 and pins[0].operator == "==", f"CI dependency is not pinned: {name}"
        assert required.specifier.contains(pins[0].version), f"CI pin violates requirements.txt: {name}"


def test_production_and_presentation_keep_separate_dependency_contracts():
    production = _requirements("requirements-ci.txt")
    assert not {"streamlit", "altair"}.intersection(production)
    for relative in ("dashboard/requirements.txt", "dashboard_v2/requirements.txt"):
        presentation = _requirements(relative)
        assert "streamlit" in presentation
        assert not {"tensorflow", "tensorflow-cpu", "torch", "transformers", "yfinance"}.intersection(presentation)
        # NumPy 2 is shared by the model and chart environments. A new UI
        # constraint must not silently exclude the pinned production version.
        if "numpy" in presentation:
            numpy_pin = next(iter(production["numpy"].specifier)).version
            assert presentation["numpy"].specifier.contains(numpy_pin)


_ISOLATED_PYTEST = textwrap.dedent("""\
    import importlib.abc
    import json
    from pathlib import Path
    import sys

    blocked = set(json.loads(sys.argv[1]))
    output = Path(sys.argv[2])

    class MissingPresentation(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition('.')[0] in blocked:
                raise ModuleNotFoundError(
                    'Presentation dependency intentionally absent: ' + fullname,
                    name=fullname,
                )

    sys.meta_path.insert(0, MissingPresentation())
    import pytest

    class Evidence:
        def __init__(self):
            self.items = []
            self.skipped = []
            self.failed = []

        def pytest_collection_modifyitems(self, items):
            self.items = [item.nodeid for item in items]

        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped.append(report.nodeid)
            if report.failed:
                self.failed.append(report.nodeid)

    evidence = Evidence()
    result = pytest.main(['-q', '-p', 'no:cacheprovider', *sys.argv[3:]], plugins=[evidence])
    output.write_text(json.dumps({
        'items': evidence.items,
        'skipped': evidence.skipped,
        'failed': evidence.failed,
        'loaded_blocked': sorted(blocked.intersection(sys.modules)),
    }), encoding='utf-8')
    raise SystemExit(result)
    """)


def _isolated_pytest(tmp_path, blocked, arguments):
    evidence_path = tmp_path / "pytest-evidence.json"
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATED_PYTEST, json.dumps(blocked), str(evidence_path), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert not evidence["loaded_blocked"]
    return evidence


@pytest.mark.parametrize("blocked", [("altair",), ("streamlit", "altair")])
def test_optional_ui_dependencies_cannot_abort_production_collection(tmp_path, blocked):
    # Discover the presentation modules, so newly added UI tests participate
    # without having to update a hand-maintained list after another incident.
    targets = sorted({
        *ROOT.glob("tests/test_dashboard*.py"),
        *ROOT.glob("tests/test_research_charts.py"),
        *ROOT.glob("tests/test_v2_history.py"),
        *ROOT.glob("dashboard_v2/tests/test_*.py"),
    })
    # These real integrity checks also run in the lightweight presentation
    # environment. ML training/metrics checks still run in the complete daily
    # suite; they require scikit-learn independently of presentation packages.
    core = ("tests/test_guard.py", "tests/test_persistence.py", "tests/test_report_persistence.py")
    evidence = _isolated_pytest(tmp_path, blocked, [
        "--collect-only", *(str(path.relative_to(ROOT)) for path in targets), *core,
    ])
    for path in core:
        assert any(node.startswith(path + "::") for node in evidence["items"]), path
    # These modules contain data integrity checks that still belong to the
    # production suite; a blanket skip of all dashboard tests is unacceptable.
    for path in ("tests/test_dashboard_data.py", "tests/test_dashboard_news.py", "tests/test_dashboard_journals.py"):
        assert any(node.startswith(path + "::") for node in evidence["items"]), path


def test_core_checks_execute_with_presentation_dependencies_absent(tmp_path):
    evidence = _isolated_pytest(tmp_path, ("streamlit", "altair"), [
        "tests/test_guard.py", "tests/test_persistence.py", "tests/test_report_persistence.py",
    ])
    assert evidence["items"]
    assert not evidence["skipped"], "Missing UI packages must never skip model or integrity checks"
    assert not evidence["failed"]
