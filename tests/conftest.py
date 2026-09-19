import os
import pytest

CACHE_FILES = [".state_cache.json", "state_cache.json"]


@pytest.fixture(autouse=True)
def cleanup_state_cache():
    """Autouse fixture to clean up state cache files before and immediately after each test."""
    for f in CACHE_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    yield
    for f in CACHE_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


@pytest.fixture(scope="session", autouse=True)
def session_cleanup_state_cache():
    """Autouse session-level fixture to clean up state cache files before and after the entire test session."""
    for f in CACHE_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    yield
    for f in CACHE_FILES:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
