"""Chain semantics and the auditor's five integrity checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from medchain.audit.verifier import ChainAuditor
from medchain.chain.mock_chain import MockChain
from medchain.fl.serialization import state_hash
from medchain.storage.local_cas import LocalCAS


def test_duplicate_submission_rejected(tmp_path):
    chain = MockChain()
    chain.register_participant("h1", "Hospital 1", stake=100)
    chain.start_round(0, "orch")
    chain.submit_update(0, "h1", "0xabc", "cid1", 100)
    with pytest.raises(ValueError, match="duplicate"):
        chain.submit_update(0, "h1", "0xdef", "cid2", 100)


def test_inactive_participant_cannot_submit():
    chain = MockChain()
    chain.register_participant("h1", "Hospital 1", stake=10)
    chain.slash("h1", amount=10, reason="test")
    chain.start_round(0, "orch")
    with pytest.raises(ValueError, match="not registered or inactive"):
        chain.submit_update(0, "h1", "0xabc", "cid1", 100)


def test_auditor_detects_tampered_artifact(tmp_path):
    chain = MockChain()
    store = LocalCAS(tmp_path / "cas")
    chain.register_participant("h1", "H1", stake=100)
    chain.start_round(0, "orch")
    chain.submit_update(0, "h1", "0xabc", "cid1", 100)

    real_state = {"W": np.ones((2, 2), dtype=np.float32)}
    cid = store.put_state(real_state)
    chain.finalize_round(0, state_hash(real_state), cid, ["h1"], [], {"h1": 10_000})

    report = ChainAuditor(chain, store).verify()
    assert report.passed

    # tamper: overwrite the stored artifact after the chain committed to its hash
    tampered = {"W": np.zeros((2, 2), dtype=np.float32)}
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **tampered)
    (tmp_path / "cas" / cid).write_bytes(buf.getvalue())

    report2 = ChainAuditor(chain, store).verify()
    assert not report2.passed
    assert any(f.check == "C1" for f in report2.findings)


def test_auditor_detects_broken_lineage(tmp_path):
    chain = MockChain()
    store = LocalCAS(tmp_path / "cas")
    chain.register_participant("h1", "H1", stake=100)

    s0 = {"W": np.zeros((2, 2), dtype=np.float32)}
    chain.start_round(0, "orch", parent_hash="")
    chain.submit_update(0, "h1", "0x1", store.put_state(s0), 100)
    chain.finalize_round(0, state_hash(s0), store.put_state(s0), ["h1"], [], {"h1": 10_000})

    s1 = {"W": np.ones((2, 2), dtype=np.float32)}
    # forged parent hash: does not match round 0's actual model hash
    chain.start_round(1, "orch", parent_hash="0xFORGED")
    chain.submit_update(1, "h1", "0x2", store.put_state(s1), 100)
    chain.finalize_round(1, state_hash(s1), store.put_state(s1), ["h1"], [], {"h1": 10_000})

    report = ChainAuditor(chain, store).verify()
    assert not report.passed
    assert any(f.check == "C2" for f in report.findings)


def test_auditor_detects_weight_conservation_violation(tmp_path):
    chain = MockChain()
    store = LocalCAS(tmp_path / "cas")
    chain.register_participant("h1", "H1", stake=100)
    chain.register_participant("h2", "H2", stake=100)
    chain.start_round(0, "orch")
    s = {"W": np.zeros((2, 2), dtype=np.float32)}
    chain.submit_update(0, "h1", "0x1", store.put_state(s), 100)
    chain.submit_update(0, "h2", "0x2", store.put_state(s), 100)
    # weights sum to 1.5, not 1.0 -- a dishonest aggregator over-weighting its cohort
    chain.finalize_round(0, state_hash(s), store.put_state(s), ["h1", "h2"], [],
                         {"h1": 10_000, "h2": 5_000})

    report = ChainAuditor(chain, store).verify()
    assert not report.passed
    assert any(f.check == "C4" for f in report.findings)


def test_gas_summary_tracks_every_call():
    chain = MockChain()
    chain.register_participant("h1", "H1", stake=100)
    chain.start_round(0, "orch")
    chain.submit_update(0, "h1", "0x1", "cid", 100)
    summary = chain.gas_summary()
    assert summary["registerParticipant"]["calls"] == 1
    assert summary["_TOTAL"]["total_gas"] > 0


def test_influence_by_participant_sums_reasonably(tmp_path):
    chain = MockChain()
    store = LocalCAS(tmp_path / "cas")
    chain.register_participant("h1", "H1", stake=100)
    chain.register_participant("h2", "H2", stake=100)
    for r in range(3):
        chain.start_round(r, "orch", parent_hash=chain.rounds[r - 1].global_model_hash if r else "")
        s = {"W": np.full((2, 2), r, dtype=np.float32)}
        chain.submit_update(r, "h1", f"0x{r}a", store.put_state(s), 100)
        chain.submit_update(r, "h2", f"0x{r}b", store.put_state(s), 100)
        chain.finalize_round(r, state_hash(s), store.put_state(s), ["h1", "h2"], [],
                             {"h1": 7_000, "h2": 3_000})
    influence = ChainAuditor(chain, store).influence_by_participant(2)
    assert abs(influence["h1"] - 0.7) < 1e-6
    assert abs(influence["h2"] - 0.3) < 1e-6
