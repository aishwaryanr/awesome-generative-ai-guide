"""An append-only, tamper-evident audit trail.

Every screening decision is written as one entry that carries the hash of the entry before
it, so the log forms a chain. Change any past entry and every hash after it stops matching,
which makes tampering detectable. This is the compliance requirement made concrete: a
regulator can be shown that the record of decisions was not edited after the fact.

In production this is an append-only store (an immutable ledger, object storage with a
retention lock, or a database with write-once semantics) with the same hash chain over it.
Here it is an in-memory list so the example runs with no external services.
"""
import hashlib
import json

_GENESIS = "0" * 64


class AuditLog:
    def __init__(self):
        self._entries = []

    def _last_hash(self):
        return self._entries[-1]["hash"] if self._entries else _GENESIS

    def record(self, event, verdict, reason, evidence):
        prev = self._last_hash()
        body = {
            "seq": len(self._entries),
            "actor": event.get("actor"),
            "event": event.get("text"),
            "verdict": verdict,
            "reason": reason,
            "evidence": list(evidence),
            "prev": prev,
        }
        digest = hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        entry = {**body, "hash": digest}
        self._entries.append(entry)
        return entry

    def verify(self):
        """Recompute the chain and confirm no entry was altered after it was written."""
        prev = _GENESIS
        for e in self._entries:
            body = {k: e[k] for k in ("seq", "actor", "event", "verdict", "reason", "evidence", "prev")}
            expected = hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()
            if e["prev"] != prev or e["hash"] != expected:
                return False
            prev = e["hash"]
        return True

    def __len__(self):
        return len(self._entries)
