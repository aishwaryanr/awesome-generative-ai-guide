"""Static guardrail: is this SQL a single, read-only query?

This runs before execution and is the first line of defense. It is deliberately strict:
allow one SELECT (optionally a leading WITH), and refuse everything else. The runtime
read-only sandbox in db.py (PRAGMA query_only) is the second line, so a destructive query
is blocked twice. Allowlisting the safe shape is stronger than trying to blocklist every
dangerous keyword, because you cannot enumerate every way to write a mutation.
"""
import re

# Whole-word keywords that must never appear in a read-only analytics query.
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "replace",
    "grant", "revoke", "attach", "detach", "pragma", "vacuum", "reindex", "merge",
    "into",  # blocks SELECT ... INTO, a write disguised as a read
)


def _strip(sql):
    # Remove -- line comments and /* */ block comments, then normalize whitespace.
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql.strip()


def is_read_only(sql):
    """Return (ok, reason). ok is True only for a single read-only SELECT query."""
    s = _strip(sql)
    if not s:
        return False, "empty query"

    # One statement only. A trailing semicolon is fine; a second statement is not.
    body = s[:-1] if s.endswith(";") else s
    if ";" in body:
        return False, "multiple statements are not allowed"

    low = body.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "only SELECT queries are allowed"

    for kw in _FORBIDDEN:
        if re.search(rf"\b{kw}\b", low):
            return False, f"write or destructive keyword blocked: {kw.upper()}"

    return True, "read-only SELECT"
