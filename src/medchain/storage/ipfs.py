"""Real IPFS backend, falling back to LocalCAS when no daemon is reachable, so a
missing daemon degrades the deployment realism of a run, never its correctness."""
from __future__ import annotations

import io
from typing import Dict, Optional

import numpy as np

from .local_cas import LocalCAS


class IPFSStore:
    def __init__(self, api_addr: str = "/ip4/127.0.0.1/tcp/5001",
                 fallback_root: str = "artifacts/cas"):
        self.fallback = LocalCAS(fallback_root)
        self._client: Optional[object] = None
        try:
            import ipfshttpclient  # type: ignore
            self._client = ipfshttpclient.connect(api_addr)
        except Exception:
            self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def put_state(self, state: Dict[str, np.ndarray]) -> str:
        if self._client is None:
            return self.fallback.put_state(state)
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: np.asarray(v, np.float32) for k, v in state.items()})
        return self._client.add_bytes(buf.getvalue())  # type: ignore[attr-defined]

    def get_state(self, cid: str) -> Dict[str, np.ndarray]:
        if self._client is None:
            return self.fallback.get_state(cid)
        with np.load(io.BytesIO(self._client.cat(cid))) as z:  # type: ignore[attr-defined]
            return {k: z[k] for k in z.files}
