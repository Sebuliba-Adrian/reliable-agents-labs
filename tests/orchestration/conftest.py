"""Chapter 30's API-key guardrail on `/ask` is real, but tier 2's own
tests exist to check each endpoint's own logic, not to also carry a
real key on every unrelated call. Overriding the dependency away by
default keeps every existing test exactly as it was;
test_api_auth.py removes the override to test the guardrail itself.
"""

import pytest

from reliable_agents_labs.api import app, require_api_key


@pytest.fixture(autouse=True)
def _bypass_api_key_by_default():
    app.dependency_overrides[require_api_key] = lambda: None
    yield
    app.dependency_overrides.pop(require_api_key, None)
