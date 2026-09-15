"""Tier 1: the stock-check graph's own behavior, no network, no model,
`check_inventory` is a plain in-memory lookup. See chapter 23.
"""

from reliable_agents_labs.stock_check_graph import build_stock_check_graph

_EMPTY = {"quantity": 0, "reorder_point": 0, "decision": ""}


def test_routes_to_reorder_when_below_the_reorder_point():
    graph = build_stock_check_graph()
    result = graph.invoke({"sku": "SKU-1029", **_EMPTY})
    assert "reorder SKU-1029" in result["decision"]


def test_routes_to_no_reorder_when_above_the_reorder_point():
    graph = build_stock_check_graph()
    result = graph.invoke({"sku": "SKU-2040", **_EMPTY})
    assert result["decision"] == "no reorder needed for SKU-2040"


def test_out_of_stock_sku_routes_to_reorder():
    graph = build_stock_check_graph()
    result = graph.invoke({"sku": "SKU-3311", **_EMPTY})
    assert "reorder SKU-3311" in result["decision"]


def test_unknown_sku_currently_reads_as_no_reorder_needed():
    # A real, honest limitation, not a hidden bug: an unknown SKU gets
    # quantity=0 and reorder_point=0 from check_stock, and 0 < 0 is
    # False, so it routes the same way a well-stocked SKU would.
    # Chapter 23's own exercise asks you to fix this.
    graph = build_stock_check_graph()
    result = graph.invoke({"sku": "SKU-9999", **_EMPTY})
    assert result["decision"] == "no reorder needed for SKU-9999"
