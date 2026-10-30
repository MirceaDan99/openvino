import shutil
import os
from pathlib import Path

import pytest

# Redirect tmp_path / tmp_path_factory into the repo tree so CI cleanup
# (deleting the cloned repo) also removes all temporary test artifacts.
# Without this, the default system temp directory accumulates large files.
_TESTS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _TESTS_DIR / "_test_output"


class NonCriticalError(AssertionError):
    pass


def pytest_configure(config):
    """Override basetemp so tmp_path* fixtures write inside the repo.

    The built-in tmpdir plugin creates TmpPathFactory (with tryfirst=True)
    before subdirectory conftest hooks run, so setting config.option.basetemp
    alone is too late — the factory already captured None.  We also patch the
    factory's _given_basetemp so the override actually takes effect.
    """
    if config.option.basetemp is None:
        config.option.basetemp = str(_OUTPUT_DIR)
        # Patch the already-created TmpPathFactory (its _basetemp is a
        # cached_property not yet accessed, so changing the input is safe).
        if hasattr(config, "_tmp_path_factory"):
            config._tmp_path_factory._given_basetemp = _OUTPUT_DIR


_KOKORO_BROKEN_PLATFORMS = {"NPU3720"}


def pytest_collection_modifyitems(config, items):
    if len(_KOKORO_BROKEN_PLATFORMS):
        platform = os.environ.get("NPUW_TEST_PLATFORM", "")
        if not any(p in platform for p in _KOKORO_BROKEN_PLATFORMS):
            return
        skip = pytest.mark.skip(reason=f"Kokoro tests temporarily broken on {platform}")
        for item in items:
            if item.fspath.basename == "test_audio_quality_sweep.py":
                item.add_marker(skip)


def pytest_sessionfinish(session, exitstatus):
    # If only non-critical assertions were detected
    # we will set exit status to 5
    if exitstatus == pytest.ExitCode.TESTS_FAILED:
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        failed_reports = terminal.stats.get("failed", [])
        non_critical_reports = terminal.stats.get("noncritical", [])

        # Only non critical tests will produce exit status 5
        if not failed_reports and non_critical_reports:
            session.exitstatus = 5

    # Clean up _test_output so the repo stays clean after local runs.
    # On CI the entire repo is deleted anyway, but this helps locally.
    # To preserve artifacts for debugging, run with:
    #     pytest --basetemp=./my_debug_output ...
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR, ignore_errors=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # After each test mark whether it was non-critical or not
    outcome = yield
    report = outcome.get_result()

    if (
        report.when == "call"
        and report.failed
        and call.excinfo
        and isinstance(call.excinfo.value, NonCriticalError)
    ):
        report.was_non_critical = True


def pytest_report_teststatus(report, config):
    # Required to mark status of non critical tests
    if getattr(report, "was_non_critical", False):
        return "noncritical", "N", "NON-CRITICAL"
