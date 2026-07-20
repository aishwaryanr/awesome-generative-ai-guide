"""The model layer, provider-agnostic, for the analytics copilot.

Three model-facing jobs live here, each with a deterministic offline version so the whole
graph runs and tests pass with no API key, plus a real path that works with ANY provider
through LangChain's init_chat_model:

- link_schema:      map the question to the tables it needs (schema linking).
- generate_sql:     write a read-only SELECT for the question and linked schema, and rewrite
                    it when execution returns an error (execution-guided self-correction).
- synthesize_answer: turn the returned rows into a sentence that answers the business question.

Selecting a real model (any one of these):
    export SQL_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export SQL_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export SQL_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If SQL_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or langchain-google-genai.
"""
import os
import re

from db import SCHEMA, TABLE_HINTS, schema_text

_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline logic."""
    model = os.environ.get("SQL_AGENT_MODEL")
    provider = os.environ.get("SQL_AGENT_PROVIDER")
    if not model:
        for key, m, p in _AUTODETECT:
            if os.environ.get(key):
                model, provider = m, p
                break
    if not model:
        return None
    try:
        from langchain.chat_models import init_chat_model
        return init_chat_model(model, model_provider=provider) if provider else init_chat_model(model)
    except Exception:
        return None


_STOPWORDS = {
    "what", "is", "the", "of", "a", "an", "to", "how", "do", "i", "my", "where", "who",
    "when", "why", "you", "your", "are", "for", "and", "in", "on", "it", "this", "that",
    "have", "has", "we", "our", "many", "much", "get", "show", "give", "me", "with",
    "by", "per", "each", "which", "does", "did", "was", "were", "there", "was",
}


def _words(text):
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _STOPWORDS and len(w) > 1}


# --- schema linking -------------------------------------------------------------------

def link_schema(question):
    """Return the tables the question plausibly needs. Empty means out of scope.

    Scores each table by word overlap between the question and the table name, its columns,
    and its business hints. Returning nothing is the abstention signal: a question that links
    to no table is out of scope, and the agent refuses instead of inventing a query.
    """
    q = _words(question)
    linked = []
    for table in SCHEMA:
        vocab = _words(table + " " + " ".join(SCHEMA[table]) + " " + TABLE_HINTS[table])
        if q & vocab:
            linked.append(table)
    return linked


# --- SQL generation with execution-guided self-correction -----------------------------

def _classify(question):
    q = question.lower()
    if any(k in q for k in ("delete", "drop", "remove all", "truncate", "wipe")):
        return "destructive"
    if "region" in q and any(k in q for k in ("revenue", "sales", "most", "top", "highest", "best")):
        return "region_revenue"
    if any(k in q for k in ("revenue", "sales", "total spend")):
        return "total_revenue"
    if "customer" in q and any(k in q for k in ("how", "count", "number", "many")):
        return "count_customers"
    if "order" in q and any(k in q for k in ("deliver",)):
        return "count_delivered_orders"
    if "order" in q and any(k in q for k in ("how", "count", "number", "many")):
        return "count_orders"
    return "unknown"


_SQL_BANK = {
    "total_revenue":
        "SELECT SUM(oi.quantity * p.unit_price) AS total_revenue "
        "FROM order_items oi JOIN products p ON oi.product_id = p.product_id",
    "count_customers":
        "SELECT COUNT(*) AS customer_count FROM customers",
    "count_delivered_orders":
        "SELECT COUNT(*) AS delivered_orders FROM orders WHERE status = 'delivered'",
    "count_orders":
        "SELECT COUNT(*) AS order_count FROM orders",
    # A naive first attempt that references a revenue column that does not exist. Execution
    # returns "no such column: revenue", and the repair below rewrites it with the real join.
    "region_revenue_naive":
        "SELECT region, SUM(revenue) AS revenue FROM orders "
        "JOIN customers USING (customer_id) GROUP BY region ORDER BY revenue DESC LIMIT 1",
    "region_revenue_fixed":
        "SELECT c.region, SUM(oi.quantity * p.unit_price) AS revenue "
        "FROM order_items oi "
        "JOIN products p ON oi.product_id = p.product_id "
        "JOIN orders o ON oi.order_id = o.order_id "
        "JOIN customers c ON o.customer_id = c.customer_id "
        "GROUP BY c.region ORDER BY revenue DESC LIMIT 1",
    # Honors a destructive request literally, so the guardrail is exercised on generated SQL.
    "destructive":
        "DELETE FROM orders",
}


def generate_sql(question, tables, error=None):
    """Return a SQL string for the question. If error is set, rewrite to fix it (self-correct)."""
    model = _get_model()
    if model is not None:
        try:
            return _generate_with_model(model, question, tables, error)
        except Exception:
            pass
    return _generate_offline(question, error)


def _generate_offline(question, error):
    intent = _classify(question)
    if intent == "region_revenue":
        # Execution feedback drives the repair: a "no such column" error triggers the fixed join.
        if error and "no such column" in error.lower():
            return _SQL_BANK["region_revenue_fixed"]
        return _SQL_BANK["region_revenue_naive"]
    return _SQL_BANK.get(intent, "SELECT 1 WHERE 1 = 0")  # unknown intent returns no rows


_GEN_SYSTEM = (
    "You are an analytics copilot that writes SQLite SQL. Reply with EXACTLY one SQL query "
    "and nothing else, no prose, no code fences. Rules: a single read-only SELECT only; never "
    "write, update, delete, or alter anything; use only the tables and columns given. If the "
    "question cannot be answered from the schema, reply: SELECT 1 WHERE 1 = 0."
)


def _generate_with_model(model, question, tables, error):
    schema = schema_text(tables)
    fix = f"\nYour previous query failed with this database error, fix it:\n{error}" if error else ""
    user = f"SCHEMA:\n{schema}\n\nQUESTION: {question}{fix}"
    resp = model.invoke([("system", _GEN_SYSTEM), ("human", user)])
    text = (getattr(resp, "content", "") or "").strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).replace("```", "").strip()
    m = re.search(r"(?is)\b(select|with)\b.*", text)
    return m.group(0).strip() if m else "SELECT 1 WHERE 1 = 0"


# --- answer synthesis -----------------------------------------------------------------

def synthesize_answer(question, columns, rows):
    """Turn the query result into a sentence. Returns None when the result answers nothing."""
    if not rows:
        return None
    model = _get_model()
    if model is not None:
        try:
            return _answer_with_model(model, question, columns, rows)
        except Exception:
            pass
    return _answer_offline(columns, rows)


def _answer_offline(columns, rows):
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0].replace('_', ' ')}: {rows[0][0]}."
    if len(rows) == 1:
        parts = ", ".join(f"{c.replace('_', ' ')} = {v}" for c, v in zip(columns, rows[0]))
        return f"Top result: {parts}."
    head = "; ".join(", ".join(str(v) for v in r) for r in rows[:5])
    return f"{len(rows)} rows ({', '.join(columns)}): {head}."


_ANS_SYSTEM = (
    "You answer a business question from a SQL result. One short sentence, state the number "
    "plainly, and do not invent anything the rows do not contain."
)


def _answer_with_model(model, question, columns, rows):
    result = f"COLUMNS: {columns}\nROWS: {rows[:20]}"
    resp = model.invoke([("system", _ANS_SYSTEM), ("human", f"QUESTION: {question}\n{result}")])
    return (getattr(resp, "content", "") or "").strip() or _answer_offline(columns, rows)
