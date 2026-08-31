"""Guards the test harness itself.

`agentic_rag/config.py` calls `load_dotenv()` at import time, which copies the
developer's `.env` into `os.environ`. `Settings(_env_file=None)` still reads
`os.environ`, so `_env_file=None` is NOT sufficient to make settings hermetic.

A local `AUTH_ENABLED=false` collapsed every request onto the single anonymous
tenant, which made a tenant-isolation integration test report a data leak
against entirely correct code. These tests pin the guard that prevents it.
"""

import os

import pytest

from agentic_rag.settings import Settings
from tests.conftest import _BEHAVIOURAL_ENV_VARS

pytestmark = pytest.mark.unit


def test_behavioural_env_vars_are_cleared_for_tests():
    for name in _BEHAVIOURAL_ENV_VARS:
        assert name not in os.environ, (
            f"{name} leaked into the test environment; the hermetic_settings_env "
            "fixture should have cleared it"
        )


def test_settings_resolve_to_code_defaults_not_the_developers_env():
    settings = Settings(_env_file=None)

    # The specific value that broke the suite: .env had AUTH_ENABLED=false.
    assert settings.auth_enabled is True
    assert settings.api_keys == ""
    assert settings.default_tenant == "public"


def test_auth_fails_closed_by_default():
    """No configured keys must mean no access, never open access."""
    settings = Settings(_env_file=None)

    assert settings.auth_enabled is True
    assert settings.api_key_map == {}


def test_guard_covers_every_security_relevant_setting():
    """A new auth/tenancy setting must be added to the guard list."""
    for name in ("API_KEYS", "AUTH_ENABLED", "DEFAULT_TENANT"):
        assert name in _BEHAVIOURAL_ENV_VARS


def test_env_var_still_wins_when_a_test_sets_it_deliberately(monkeypatch):
    """The guard clears ambient config; it must not block intentional overrides."""
    monkeypatch.setenv("AUTH_ENABLED", "false")

    assert Settings(_env_file=None).auth_enabled is False
