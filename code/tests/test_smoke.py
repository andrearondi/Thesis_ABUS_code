"""Smoke test for the abus package scaffold.

Verifies:
  - `abus` is importable.
  - `abus.__version__` is a non-empty PEP 440 string.
  - Subpackage namespaces `abus.io`, `abus.geometry`, `abus.data` are importable.
"""

import re

import abus
import abus.data
import abus.geometry
import abus.io


def test_version_is_nonempty_string() -> None:
    assert isinstance(abus.__version__, str)
    assert len(abus.__version__) > 0


def test_version_is_pep440() -> None:
    # Minimal PEP 440 pattern: N.N.N optionally followed by pre/post/dev suffixes.
    # The end-anchor prevents garbage like "0.1.0extra_garbage" from passing.
    pep440_re = re.compile(r"^\d+\.\d+\.\d+(\.?(a|b|rc|post|dev)\d+)?$")
    assert pep440_re.match(
        abus.__version__
    ), f"__version__ '{abus.__version__}' does not match PEP 440 N.N.N pattern"


def test_subpackage_imports() -> None:
    # Subpackages must be importable; they are placeholders for later stories.
    assert abus.io is not None
    assert abus.geometry is not None
    assert abus.data is not None
