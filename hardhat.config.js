require("@nomicfoundation/hardhat-toolbox");

/**
 * Gas reporting is ON by default. The measured per-round gas cost is a required
 * table in the paper -- "is this actually deployable" is the first question a
 * systems reviewer asks about any blockchain-FL design, and the usual reason
 * such papers are rejected.
 */
module.exports = {
  solidity: {
    version: "0.8.24",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  networks: {
    hardhat: { chainId: 31337 },
    localhost: { url: "http://127.0.0.1:8545", chainId: 31337 },
  },
  gasReporter: {
    enabled: true,
    currency: "USD",
    // Permissioned consortium chains have no real gas price. Report gas UNITS as
    // the primary number and treat any fiat figure as illustrative only.
    gasPrice: 20,
  },
};
