"""An append-only, tamper-evident audit trail.

Every decision writes one record. Each record carries the hash of the record before it, so
the log is a hash chain: change any past record and every hash after it stops matching, which
is what makes tampering detectable (verify_chain returns False). This is the minimum an
auditor or a regulator needs, who decided what, when, on what evidence, and that the trail was
not edited after the fact. In production this is an append-only store (a WORM bucket, an
immutable database table, or a ledger), not an in-memory list.

The same log also carries the set of processed invoice ids, which is how policy.py catches a
duplicate before it becomes a double payment.
"""
import hashlib
import json
import time

AUDIT_LOG = []
PROCESSED_IDS = set()

_PAYLOAD_KEYS = ("seq", "ts", "document_id", "decision", "reasons", "amount", "category", "confidence")


def _digest(prev_hash: str, payload: dict) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()


def write_record(document_id, decision, reasons, fields) -> dict:
    """Append one immutable, hash-chained decision record and return it."""
    prev = AUDIT_LOG[-1]["hash"] if AUDIT_LOG else "genesis"
    payload = {
        "seq": len(AUDIT_LOG),
        "ts": time.time(),
        "document_id": document_id,
        "decision": decision,
        "reasons": list(reasons),
        "amount": fields.get("amount"),
        "category": fields.get("category"),
        "confidence": fields.get("_confidence"),
    }
    record = dict(payload)
    record["prev_hash"] = prev
    record["hash"] = _digest(prev, payload)
    AUDIT_LOG.append(record)
    if document_id:
        PROCESSED_IDS.add(document_id)
    return record


def verify_chain() -> bool:
    """Recompute every hash from the start. False means a record was altered."""
    prev = "genesis"
    for rec in AUDIT_LOG:
        payload = {k: rec[k] for k in _PAYLOAD_KEYS}
        if rec["prev_hash"] != prev or rec["hash"] != _digest(prev, payload):
            return False
        prev = rec["hash"]
    return True


def is_duplicate(document_id) -> bool:
    return document_id in PROCESSED_IDS


def reset():
    """Clear the log and the processed-id set (used between test runs)."""
    AUDIT_LOG.clear()
    PROCESSED_IDS.clear()
