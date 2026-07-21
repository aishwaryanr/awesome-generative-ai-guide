"""Mock insurance-submission documents and a field extractor.

In production, `extract` is a layout-aware parser (Docling, LlamaParse, a Document AI
service) followed by an extraction model that reads PDFs, scans, forms, and tables and
returns typed fields WITH a per-field confidence. Here the document is a small text blob
and the extractor is deterministic, so the whole system runs offline with no API key and
no external service. The shape is the same: document in, typed fields plus confidence out.

The `scan_quality: low` marker simulates a poor scan that lowers OCR confidence on every
field, which is what routes a submission to a human underwriter in the verify step.
"""

# Fields the underwriting policy needs before it can decide. A missing one routes to a human.
REQUIRED_FIELDS = ("applicant_name", "property_value", "requested_coverage",
                   "year_built", "prior_claims", "construction", "flood_zone")

# Built-in submissions. A real one is a PDF or a scan; this is the text an extractor returns.
SAMPLES = {
    "clean-approve":
        "applicant_name: Jane Okafor; property_value: 420000; requested_coverage: 300000; "
        "year_built: 1998; prior_claims: 0; construction: masonry; flood_zone: no; "
        "ssn: 123-45-6789",
    "flood-decline":
        "applicant_name: Ravi Patel; property_value: 500000; requested_coverage: 400000; "
        "year_built: 2005; prior_claims: 1; construction: masonry; flood_zone: yes; "
        "flood_endorsement: no; ssn: 222-33-4444",
    "over-insurance-decline":
        "applicant_name: Mia Chen; property_value: 300000; requested_coverage: 450000; "
        "year_built: 2010; prior_claims: 0; construction: frame; flood_zone: no",
    "high-impact-refer":
        "applicant_name: Acme Holdings LLC; property_value: 1200000; requested_coverage: 750000; "
        "year_built: 2001; prior_claims: 0; construction: masonry; flood_zone: no",
    "low-confidence-refer":
        "applicant_name: Sam Rivera; property_value: 380000; requested_coverage: 260000; "
        "year_built: 1994; prior_claims: 0; construction: masonry; flood_zone: no; "
        "scan_quality: low",
    "missing-field-refer":
        "applicant_name: Lto Nguyen; property_value: 350000; year_built: 1990; "
        "prior_claims: 0; construction: masonry; flood_zone: no",
    "old-construction-refer":
        "applicant_name: Ada Bloom; property_value: 260000; requested_coverage: 180000; "
        "year_built: 1928; prior_claims: 0; construction: masonry; flood_zone: no",
}

CONF_FLOOR = 0.85  # a field below this is treated as an unreliable read and sent to a human


def _mask_ssn(ssn: str) -> str:
    """Show only the last 4 digits. PII is masked at the extraction boundary, so nothing
    downstream (memory, policy, the audit trail) ever holds the number in the clear."""
    digits = [c for c in ssn if c.isdigit()]
    last4 = "".join(digits[-4:]) if len(digits) >= 4 else "".join(digits)
    return f"***-**-{last4}"


def extract(document: str) -> dict:
    """Parse a (mock) document into typed fields with a per-field confidence in [0, 1].

    Returns {"fields": {name: {"value": ..., "confidence": ...}}, "pii_masked": {...}}.
    A real pipeline swaps this for a layout-aware parse plus an extraction model; the
    return shape (typed values plus confidence, PII already masked) stays the same.
    """
    raw = {}
    for part in document.split(";"):
        if ":" in part:
            key, val = part.split(":", 1)
            raw[key.strip().lower()] = val.strip()

    scan_low = raw.get("scan_quality", "high").lower() == "low"
    base_conf = 0.70 if scan_low else 0.98  # a poor scan lowers confidence on every field

    fields = {}

    def put(name, caster=None):
        if raw.get(name, "") == "":
            return
        value = raw[name]
        if caster is not None:
            try:
                value = caster(value)
            except ValueError:
                fields[name] = {"value": None, "confidence": 0.40}  # present but unreadable
                return
        fields[name] = {"value": value, "confidence": base_conf}

    put("applicant_name")
    put("property_value", int)
    put("requested_coverage", int)
    put("year_built", int)
    put("prior_claims", int)
    put("construction")
    put("flood_zone")
    put("flood_endorsement")

    # PII is detected and masked here, at the boundary, and never returned in the clear.
    pii_masked = {}
    if raw.get("ssn"):
        pii_masked["ssn"] = _mask_ssn(raw["ssn"])

    return {"fields": fields, "pii_masked": pii_masked}
