"""The model layer, provider-agnostic: turn a semi-structured document into typed fields.

In this system the model does one job, extraction: read a (mock) invoice or expense and
return structured fields (vendor, invoice id, amount, currency, date, category). The policy
decision itself is deterministic and lives in policy.py, because the rules are knowable and
you want them exact and auditable. This file gives you two implementations behind one
function:

- a deterministic OFFLINE parser so the whole graph runs with no API key, which is what
  makes this example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model
  and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not
  change the graph to change the model.

Selecting a model (any one of these):
    export FINOPS_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export FINOPS_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export FINOPS_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If FINOPS_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or
langchain-google-genai.
"""
import json
import os
import re

# The fields a complete record needs before a policy decision can be trusted. Extraction
# confidence below is the share of these that were found, which is what routes a thin or
# garbled document to a human instead of a guess.
REQUIRED_FIELDS = ("vendor", "invoice_id", "amount", "date", "category")

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline parser."""
    model = os.environ.get("FINOPS_MODEL")
    provider = os.environ.get("FINOPS_PROVIDER")
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


def extract_fields(document: str) -> dict:
    """Extract typed fields from a document. Adds a "_confidence" score. Provider-agnostic."""
    model = _get_model()
    fields = None
    if model is not None:
        try:
            fields = _extract_with_model(model, document)
        except Exception:
            fields = None  # fall back to the offline parser if the provider call fails
    if fields is None:
        fields = _extract_offline(document)
    fields["_confidence"] = _confidence(fields)
    return fields


def _confidence(fields: dict) -> float:
    found = sum(1 for f in REQUIRED_FIELDS if fields.get(f) not in (None, "", []))
    return round(found / len(REQUIRED_FIELDS), 3)


# --- offline deterministic parser (no provider) ---------------------------------------

# Labelled patterns first (Vendor: ..., Invoice: INV-123, Amount: USD 1250.00), then a few
# unlabelled fallbacks so a freeform line like "USD 240 for team lunch" still yields amount
# and a category. A real system swaps this parser for a layout-aware extractor.
_LABELLED = {
    "vendor": r"vendor\s*[:=]\s*([^|\n,]+)",
    "invoice_id": r"invoice(?:\s*id)?\s*[:=]?\s*(inv-\w+)",
    "amount": r"(?:amount|total)\s*[:=]\s*(?:usd|eur|gbp|inr|\$)?\s*([\d,]+(?:\.\d{1,2})?)",
    "date": r"(\d{4}-\d{2}-\d{2})",
    "category": r"category\s*[:=]\s*([a-z][a-z \-]*)",
}
_CURRENCY = r"\b(usd|eur|gbp|inr)\b"
_AMOUNT_FALLBACK = r"(?:usd|eur|gbp|inr|\$)\s*([\d,]+(?:\.\d{1,2})?)"
_CATEGORY_KEYWORDS = {
    "meals": ("lunch", "dinner", "meal", "restaurant", "cafe", "coffee"),
    "travel": ("flight", "hotel", "airfare", "taxi", "mileage"),
    "software": ("software", "license", "subscription", "saas", "cloud"),
    "alcohol": ("bar", "wine", "beer", "alcohol"),
}


def _extract_offline(document: str) -> dict:
    text = document.lower()
    fields = {}
    for key, pat in _LABELLED.items():
        m = re.search(pat, text)
        if m:
            fields[key] = m.group(1).strip()
    if "amount" in fields:
        fields["amount"] = float(fields["amount"].replace(",", ""))
    else:
        m = re.search(_AMOUNT_FALLBACK, text)
        if m:
            fields["amount"] = float(m.group(1).replace(",", ""))
    cur = re.search(_CURRENCY, text)
    fields["currency"] = cur.group(1).upper() if cur else ("USD" if "$" in document else None)
    if "category" not in fields:
        for cat, words in _CATEGORY_KEYWORDS.items():
            if any(w in text for w in words):
                fields["category"] = cat
                break
    if fields.get("category"):
        fields["category"] = fields["category"].strip().lower()
    return fields


# --- real path: any provider via init_chat_model --------------------------------------

_EXTRACT_SYSTEM = (
    "You extract fields from a financial document (an invoice or an expense). "
    "Return ONLY a JSON object with these keys and nothing else:\n"
    '{"vendor": str|null, "invoice_id": str|null, "amount": number|null, '
    '"currency": str|null, "date": "YYYY-MM-DD"|null, "category": str|null}\n'
    "Use null for any field not clearly present. Do not guess amounts or ids. "
    "category is a single lowercase word such as software, meals, travel, hardware, office, alcohol."
)


def _extract_with_model(model, document: str) -> dict:
    resp = model.invoke([("system", _EXTRACT_SYSTEM), ("human", document)])
    raw = (getattr(resp, "content", "") or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    data = json.loads(raw[start:end + 1])
    fields = {k: v for k, v in data.items() if v not in (None, "")}
    if "amount" in fields:
        fields["amount"] = float(fields["amount"])
    if fields.get("category"):
        fields["category"] = str(fields["category"]).strip().lower()
    if fields.get("currency"):
        fields["currency"] = str(fields["currency"]).strip().upper()
    return fields
