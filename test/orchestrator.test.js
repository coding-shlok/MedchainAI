const { expect } = require("chai");
const { ethers } = require("hardhat");

/**
 * These tests encode the INTEGRITY CLAIMS the paper makes. Each maps to a check
 * in src/medchain/audit/verifier.py, so if a claim in the text is not testable
 * here, the claim is too vague to publish.
 */
describe("MedChainAI orchestrator", function () {
  let registry, orchestrator, token, admin, h1, h2, h3;

  beforeEach(async function () {
    [admin, h1, h2, h3] = await ethers.getSigners();
    registry = await (await ethers.getContractFactory("ParticipantRegistry")).deploy();
    orchestrator = await (await ethers.getContractFactory("FLOrchestrator"))
      .deploy(await registry.getAddress());
    token = await (await ethers.getContractFactory("IncentiveToken")).deploy();

    const stake = ethers.parseEther("1");
    await registry.connect(h1).register("General Hospital", { value: stake });
    await registry.connect(h2).register("St Mary's", { value: stake });
    await registry.connect(h3).register("City Clinic", { value: stake });
  });

  const H = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

  it("registers participants with cold-start reputation of 0.5", async function () {
    const p = await registry.participants(h1.address);
    expect(p.reputationBps).to.equal(5000);
    expect(p.active).to.equal(true);
    expect(await registry.count()).to.equal(3);
  });

  it("rejects duplicate submissions in a round (C: weight multiplication)", async function () {
    await orchestrator.startRound(0, ethers.ZeroHash);
    await orchestrator.connect(h1).submitUpdate(0, H("u1"), "cid1", 100);
    await expect(
      orchestrator.connect(h1).submitUpdate(0, H("u1b"), "cid1b", 100)
    ).to.be.revertedWith("duplicate submission");
  });

  it("rejects finalization whose weights do not sum to 1 (C4)", async function () {
    await orchestrator.startRound(0, ethers.ZeroHash);
    await orchestrator.connect(h1).submitUpdate(0, H("u1"), "cid1", 100);
    await orchestrator.connect(h2).submitUpdate(0, H("u2"), "cid2", 100);
    await expect(
      orchestrator.finalizeRound(0, H("m0"), "mcid0", [h1.address, h2.address], [3000, 3000])
    ).to.be.revertedWith("weights must sum to 1");
  });

  it("rejects accepting an address that never submitted (C3)", async function () {
    await orchestrator.startRound(0, ethers.ZeroHash);
    await orchestrator.connect(h1).submitUpdate(0, H("u1"), "cid1", 100);
    await expect(
      orchestrator.finalizeRound(0, H("m0"), "mcid0", [h3.address], [10000])
    ).to.be.revertedWith("accepted a non-submitter");
  });

  it("verifies an intact hash lineage and detects a broken one (C2)", async function () {
    const m0 = H("model0"), m1 = H("model1");
    await orchestrator.startRound(0, ethers.ZeroHash);
    await orchestrator.connect(h1).submitUpdate(0, H("u1"), "cid1", 100);
    await orchestrator.finalizeRound(0, m0, "mcid0", [h1.address], [10000]);

    await orchestrator.startRound(1, m0);
    await orchestrator.connect(h1).submitUpdate(1, H("u2"), "cid2", 100);
    await orchestrator.finalizeRound(1, m1, "mcid1", [h1.address], [10000]);
    expect(await orchestrator.verifyLineage(1)).to.equal(true);

    // A round whose parent link does not match its predecessor = rewritten history
    await orchestrator.startRound(2, H("forged-parent"));
    await orchestrator.connect(h1).submitUpdate(2, H("u3"), "cid3", 100);
    await orchestrator.finalizeRound(2, H("model2"), "mcid2", [h1.address], [10000]);
    expect(await orchestrator.verifyLineage(2)).to.equal(false);
  });

  it("blocks a fully slashed participant from submitting", async function () {
    await registry.slash(h2.address, ethers.parseEther("1"), "reputation below threshold");
    expect(await registry.isActive(h2.address)).to.equal(false);
    await orchestrator.startRound(0, ethers.ZeroHash);
    await expect(
      orchestrator.connect(h2).submitUpdate(0, H("u2"), "cid2", 100)
    ).to.be.revertedWith("inactive participant");
  });

  it("refuses to pay the same round twice (reorg double-mint)", async function () {
    await token.distributeRewards(0, [h1.address, h2.address], [600, 400]);
    expect(await token.balanceOf(h1.address)).to.equal(600);
    await expect(
      token.distributeRewards(0, [h1.address], [1000])
    ).to.be.revertedWith("round already paid");
  });
});
