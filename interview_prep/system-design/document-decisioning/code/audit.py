"""The audit trail: an append-only, tamper-evident record of every decision.

A regulated decision has to be reconstructable long after the fact: what was decided, on what
evidence, by which model, and when. This is a small hash-chained log that stands in for a
write-once (WORM) store. Each record carries the hash of the previous one, so any later edit
to any record breaks the chain and `verify_chain` catches it. PII is already masked upstream
(documents.py), so nothing sensitive is written here in the clear.

In production this is an append-only, access-controlled store (WORM object storage, an
immutable database, or a ledger) with retention rules, not an in-memory list.
"""
import hashlib
import json
import time

AUDIT_LOG = []  # in production: an append-only, access-controlled, retained store
_GENESIS = "0" * 64


def record(entry: dict) -> dict:
    """Append one decision to the trail and return the stored record (with its hash)."""
    prev = AUDIT_LOG[-1]["hash"] if AUDIT_LOG else _GENESIS
    body = {**entry, "seq": len(AUDIT_LOG), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                       time.gmtime()),
            "prev_hash": prev}
    digest = hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()
    rec = {**body, "hash": digest}
    AUDIT_LOG.append(rec)
    return rec


def verify_chain() -> bool:
    """Recompute every hash and confirm the chain is intact (nothing was altered)."""
    prev = _GENESIS
    for rec in AUDIT_LOG:
        body = {k: val for k, val in rec.items() if k != "hash"}
        digest = hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()
        if digest != rec["hash"] or rec["prev_hash"] != prev:
            return False
        prev = rec["hash"]
    return True
