"""One independent hash chain per operation; metadata only."""
import datetime
import json
import uuid
from pathlib import Path
from .io import canonical, require
import hashlib

class EvidenceLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event, payload):
        previous = "0" * 64
        if self.path.exists():
            previous = json.loads(self.path.read_text().splitlines()[-1])["hash"]
        record = {"previous_hash": previous, "event": event, "payload": payload}
        record["hash"] = hashlib.sha256(canonical(record).encode()).hexdigest()
        with self.path.open("a") as handle:
            handle.write(canonical(record) + "\n")
        return record["hash"]

    def verify(self):
        return verify_evidence(self.path)

def verify_evidence(path):
    previous, count = "0" * 64, 0
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        stored = record.pop("hash")
        require(record["previous_hash"] == previous, "Evidence link verification failed.")
        require(hashlib.sha256(canonical(record).encode()).hexdigest() == stored,
                "Evidence digest verification failed.")
        previous, count = stored, count + 1
    return {"records": count, "head": previous}

def new_operation(runtime_root, kind):
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(runtime_root)
    run_dir = root / "results/classifier-runs" / (kind + "-" + run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    log = EvidenceLog(root / "logs/classifier-modular" / (kind + "-" + run_id + ".jsonl"))
    return run_id, timestamp, run_dir, log
