"""Shared pytest fixtures and path setup for the globalbudget test suite."""

import sys
from pathlib import Path

import pytest

# Make the local package importable without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalbudget import data_loader  # noqa: E402


@pytest.fixture(scope="session")
def master():
    """The wide master table, loaded once for the whole test session."""
    return data_loader.load_master()
