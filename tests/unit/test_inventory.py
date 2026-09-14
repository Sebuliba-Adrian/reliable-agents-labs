"""Tier 1: unit tests for the fixed inventory dataset. Deterministic,
no network, no model.
"""

from reliable_agents_labs.inventory import check_inventory


def test_check_inventory_returns_a_known_sku():
    record = check_inventory("SKU-1029")
    assert record is not None
    assert record.quantity == 4
    assert record.reorder_point == 20


def test_check_inventory_returns_none_for_unknown_sku():
    assert check_inventory("SKU-9999") is None
