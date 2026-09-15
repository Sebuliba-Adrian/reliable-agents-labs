"""Tier 1: unit tests for the fixed inventory dataset. Deterministic,
no network, no model.
"""

import json

from reliable_agents_labs.inventory import check_inventory, run_check_inventory_tool


def test_check_inventory_returns_a_known_sku():
    record = check_inventory("SKU-1029")
    assert record is not None
    assert record.quantity == 4
    assert record.reorder_point == 20


def test_check_inventory_returns_none_for_unknown_sku():
    assert check_inventory("SKU-9999") is None


def test_run_check_inventory_tool_returns_the_record_as_json():
    output = json.loads(run_check_inventory_tool({"sku": "SKU-1029"}))
    assert output["quantity"] == 4


def test_run_check_inventory_tool_returns_an_error_payload_for_an_unknown_sku():
    output = json.loads(run_check_inventory_tool({"sku": "SKU-9999"}))
    assert "error" in output
