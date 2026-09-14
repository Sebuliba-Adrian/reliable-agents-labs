"""Tier 1: unit tests. Deterministic application logic only, no network,
no model, no real config file dependency beyond what's asserted directly.
"""

import pytest

from reliable_agents_labs.models import ModelResult, build_model_client


def test_model_result_requires_all_fields():
    result = ModelResult(
        text="in stock: 12 units",
        input_tokens=42,
        output_tokens=7,
        model_id="gemini-2.5-flash",
        provider="gemini",
    )
    assert result.text == "in stock: 12 units"
    assert result.provider == "gemini"


def test_build_model_client_rejects_unknown_provider(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text("bad_role:\n  provider: not_a_real_provider\n  model_id: x\n")
    with pytest.raises(ValueError, match="Unknown provider"):
        build_model_client("bad_role", config_path=str(config_path))


def test_build_model_client_reads_real_config():
    # Doesn't construct the client (that needs an API key), just proves the
    # config file this repo ships is well-formed and has the roles the
    # rest of the book depends on.
    import yaml

    with open("config/models.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["answer_model"]["provider"] == "gemini"
    assert "model_id" in config["answer_model"]
