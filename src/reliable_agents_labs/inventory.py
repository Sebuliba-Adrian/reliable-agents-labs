"""A tiny, deterministic inventory data source.

Real data access starts here, deliberately as plain as possible: a fixed
in-memory dataset, no database yet (Project 2 introduces one). The point
of chapter 6 is the tool-calling mechanism itself, not where the data
lives, a database-backed version would teach the exact same lesson with
more setup in the way.
"""

from pydantic import BaseModel


class InventoryRecord(BaseModel):
    sku: str
    quantity: int
    reorder_point: int


_INVENTORY: dict[str, InventoryRecord] = {
    record.sku: record
    for record in [
        InventoryRecord(sku="SKU-1029", quantity=4, reorder_point=20),
        InventoryRecord(sku="SKU-2040", quantity=150, reorder_point=30),
        InventoryRecord(sku="SKU-3311", quantity=0, reorder_point=10),
    ]
}


def check_inventory(sku: str) -> InventoryRecord | None:
    """The reorder agent's first real tool: look up one SKU's actual
    quantity on hand. Returns None for a SKU that doesn't exist, callers
    must handle that, not assume every SKU the model asks about is real.
    """
    return _INVENTORY.get(sku)
