const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("deployer:", deployer.address);

  const Registry = await hre.ethers.getContractFactory("ParticipantRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();

  const Orchestrator = await hre.ethers.getContractFactory("FLOrchestrator");
  const orchestrator = await Orchestrator.deploy(await registry.getAddress());
  await orchestrator.waitForDeployment();

  const Token = await hre.ethers.getContractFactory("IncentiveToken");
  const token = await Token.deploy();
  await token.waitForDeployment();

  const out = {
    network: hre.network.name,
    ParticipantRegistry: await registry.getAddress(),
    FLOrchestrator: await orchestrator.getAddress(),
    IncentiveToken: await token.getAddress(),
  };
  console.log(JSON.stringify(out, null, 2));

  // The Python chain client reads this file, so deploy addresses never get
  // pasted into configs by hand.
  const fs = require("fs");
  fs.mkdirSync("artifacts/deploy", { recursive: true });
  fs.writeFileSync("artifacts/deploy/addresses.json", JSON.stringify(out, null, 2));
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
