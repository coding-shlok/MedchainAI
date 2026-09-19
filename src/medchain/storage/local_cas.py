"""Content-addressed local store: the IPFS stand-in.

Same interface as the IPFS client, so experiments run with no daemon and the
chain logic is identical either way. The CID is the content digest, which is the
only property the on-chain audit trail actually depends on.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Dict

import numpy as np


class LocalCAS:
    def __init__(self, root="artifacts/cas"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cid(payload: bytes) -> str:
        return "bafy" + hashlib.sha256(payload).hexdigest()[:52]

    def put_bytes(self, payload: bytes) -> str:
        cid = self._cid(payload)
        (self.root / cid).write_bytes(payload)
        return cid

    def get_bytes(self, cid: str) -> bytes:
        path = self.root / cid
        if not path.exists():
            raise KeyError(f"CID not found: {cid}")
        return path.read_bytes()

    def put_state(self, state: Dict[str, np.ndarray]) -> str:
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v, np.float32) for k, v in state.items()})
        return self.put_bytes(buf.getvalue())

    def get_state(self, cid: str) -> Dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.get_bytes(cid))) as z:
            return {k: z[k] for k in z.files}

    def verify(self, cid: str) -> bool:
        try:
            return self._cid(self.get_bytes(cid)) == cid
        except KeyError:
            return False
