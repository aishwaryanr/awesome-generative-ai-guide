"""A tiny in-memory analytics warehouse (SQLite) with a read-only executor.

In production this stands in for your real data warehouse (Snowflake, BigQuery, Postgres),
reached over a read-only connection with least-privilege credentials. Here it is a small
SQLite database built in memory, so the whole system runs offline with no external services.

The schema is a small star layout: fact-like `orders` and `order_items` around dimension-like
`customers` and `products`. The executor is where safety lives: every query runs under
`PRAGMA query_only = ON`, so even a write that slipped past the static guardrail cannot change
the database.
"""
import sqlite3

# The catalog the agent links questions against. Keeping the schema as data (not buried in DDL)
# is what lets the schema-linking step select a small, relevant subset instead of the whole thing.
SCHEMA = {
    "customers": ["customer_id", "name", "region", "signup_date"],
    "products": ["product_id", "name", "category", "unit_price"],
    "orders": ["order_id", "customer_id", "order_date", "status"],
    "order_items": ["order_id", "product_id", "quantity"],
}

# Natural-language hints per table: business words that should map to this table during schema
# linking. This is a small stand-in for column descriptions and value samples in a real catalog.
TABLE_HINTS = {
    "customers": "customer customers user users account accounts region regions signup",
    "products": "product products item items catalog category categories price unit",
    "orders": "order orders purchase purchases status shipped delivered cancelled date when",
    "order_items": "line item quantity units sold revenue sales amount spend total",
}

_DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT, region TEXT, signup_date TEXT
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT, category TEXT, unit_price REAL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER, order_date TEXT, status TEXT
);
CREATE TABLE order_items (
    order_id INTEGER, product_id INTEGER, quantity INTEGER
);
"""

_CUSTOMERS = [
    (1, "Alice", "West", "2025-01-05"), (2, "Bob", "East", "2025-02-10"),
    (3, "Carol", "West", "2025-03-01"), (4, "Dan", "North", "2025-03-15"),
    (5, "Eve", "East", "2025-04-02"), (6, "Frank", "North", "2025-05-20"),
]
_PRODUCTS = [
    (101, "Widget", "Hardware", 25.0), (102, "Gadget", "Hardware", 40.0),
    (103, "Manual", "Media", 10.0), (104, "Cable", "Hardware", 5.0),
    (105, "License", "Software", 100.0),
]
_ORDERS = [
    (1001, 1, "2025-05-01", "delivered"), (1002, 2, "2025-05-03", "delivered"),
    (1003, 1, "2025-05-10", "cancelled"), (1004, 3, "2025-06-01", "delivered"),
    (1005, 4, "2025-06-05", "shipped"), (1006, 5, "2025-06-10", "delivered"),
    (1007, 6, "2025-06-15", "shipped"), (1008, 2, "2025-07-01", "delivered"),
    (1009, 3, "2025-07-02", "cancelled"), (1010, 4, "2025-07-05", "delivered"),
]
_ORDER_ITEMS = [
    (1001, 101, 2), (1001, 103, 1), (1002, 102, 1), (1003, 104, 5),
    (1004, 105, 1), (1005, 101, 3), (1006, 102, 2), (1006, 104, 4),
    (1007, 103, 2), (1008, 105, 1), (1008, 101, 1), (1009, 102, 1),
    (1010, 104, 10),
]


def _build():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(_DDL)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", _CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", _PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", _ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?)", _ORDER_ITEMS)
    conn.commit()
    # Read-only from here on. query_only blocks every write at the engine level, so the
    # executor is a genuine read-only sandbox rather than a promise in a prompt.
    conn.execute("PRAGMA query_only = ON")
    return conn


_CONN = _build()


def schema_text(tables=None):
    """Render the schema (optionally a linked subset) as compact DDL-style text for the model."""
    tables = tables or list(SCHEMA)
    lines = []
    for t in tables:
        cols = ", ".join(SCHEMA[t])
        lines.append(f"{t}({cols})")
    return "\n".join(lines)


def execute(sql):
    """Run a read-only query and return (columns, rows). Raises sqlite3.Error on a bad query.

    The connection is pinned to query_only, so any write raises instead of mutating data.
    """
    cur = _CONN.execute(sql)
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    return columns, rows
