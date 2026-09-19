// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./ParticipantRegistry.sol";

/// @title FLOrchestrator
/// @notice Round lifecycle + tamper-evident provenance of the global model.
/// @dev WHAT IS DELIBERATELY NOT HERE: model weights. Only 32-byte digests and
///      short CID strings are stored. Storing a ResNet-50 on-chain would cost
///      roughly 10^9 gas per round and is the single fastest way to get a
///      blockchain-FL paper rejected as impractical. Payloads live in IPFS; the
///      chain stores the commitment that makes them non-repudiable.
contract FLOrchestrator {
    struct Submission {
        address participant;
        bytes32 updateHash;
        string  cid;
        uint64  numSamples;
        uint32  weightBps;
        bool    accepted;
        bool    exists;
        uint256 blockNumber;
    }

    struct Round {
        bytes32 globalModelHash;
        string  globalModelCid;
        bytes32 parentHash;      // hash-linked to the previous round: the audit chain
        address aggregator;
        uint256 startedBlock;
        uint256 finalizedBlock;
        bool    started;
        bool    finalized;
        address[] submitters;
    }

    ParticipantRegistry public immutable registry;
    address public admin;
    uint256 public currentRound;

    mapping(uint256 => Round) private rounds;
    mapping(uint256 => mapping(address => Submission)) private submissions;

    event RoundStarted(uint256 indexed round, address aggregator, bytes32 parentHash);
    event UpdateSubmitted(uint256 indexed round, address indexed participant,
                          bytes32 updateHash, string cid, uint64 numSamples);
    event RoundFinalized(uint256 indexed round, bytes32 modelHash, string cid,
                         bytes32 parentHash, uint256 accepted, uint256 rejected);

    modifier onlyAdmin() {
        require(msg.sender == admin, "not admin");
        _;
    }

    constructor(address registryAddr) {
        registry = ParticipantRegistry(registryAddr);
        admin = msg.sender;
    }

    function startRound(uint256 index, bytes32 parentHash) external onlyAdmin {
        require(!rounds[index].started, "round exists");
        Round storage r = rounds[index];
        r.started = true;
        r.parentHash = parentHash;
        r.aggregator = msg.sender;
        r.startedBlock = block.number;
        currentRound = index;
        emit RoundStarted(index, msg.sender, parentHash);
    }

    /// @notice Publish a commitment to a local update.
    /// @dev One submission per participant per round. Without this check a client
    ///      could submit repeatedly and multiply its aggregation weight -- a
    ///      cheap and complete break of any weighting scheme, ours included.
    function submitUpdate(uint256 round, bytes32 updateHash, string calldata cid,
                          uint64 numSamples) external {
        Round storage r = rounds[round];
        require(r.started && !r.finalized, "round not open");
        require(registry.isActive(msg.sender), "inactive participant");
        require(!submissions[round][msg.sender].exists, "duplicate submission");

        submissions[round][msg.sender] = Submission({
            participant: msg.sender, updateHash: updateHash, cid: cid,
            numSamples: numSamples, weightBps: 0, accepted: false,
            exists: true, blockNumber: block.number
        });
        r.submitters.push(msg.sender);
        emit UpdateSubmitted(round, msg.sender, updateHash, cid, numSamples);
    }

    function finalizeRound(uint256 round, bytes32 modelHash, string calldata cid,
                           address[] calldata accepted, uint32[] calldata weightsBps)
        external onlyAdmin
    {
        require(accepted.length == weightsBps.length, "length mismatch");
        Round storage r = rounds[round];
        require(r.started && !r.finalized, "round not open");

        uint256 total;
        for (uint256 i = 0; i < accepted.length; i++) {
            Submission storage s = submissions[round][accepted[i]];
            require(s.exists, "accepted a non-submitter");
            s.accepted = true;
            s.weightBps = weightsBps[i];
            total += weightsBps[i];
        }
        // Enforced on-chain so a dishonest aggregator cannot silently under- or
        // over-weight the cohort. 50 bps of slack absorbs float->bps rounding.
        require(accepted.length == 0 || (total >= 9_950 && total <= 10_050), "weights must sum to 1");

        r.globalModelHash = modelHash;
        r.globalModelCid = cid;
        r.finalized = true;
        r.finalizedBlock = block.number;
        emit RoundFinalized(round, modelHash, cid, r.parentHash,
                            accepted.length, r.submitters.length - accepted.length);
    }

    function getRound(uint256 index)
        external view returns (bytes32, string memory, bytes32, uint256, bool)
    {
        Round storage r = rounds[index];
        return (r.globalModelHash, r.globalModelCid, r.parentHash, r.finalizedBlock, r.finalized);
    }

    function getSubmission(uint256 round, address who)
        external view returns (bytes32, string memory, uint64, uint32, bool)
    {
        Submission storage s = submissions[round][who];
        return (s.updateHash, s.cid, s.numSamples, s.weightBps, s.accepted);
    }

    function submitterCount(uint256 round) external view returns (uint256) {
        return rounds[round].submitters.length;
    }

    /// @notice Verify the hash chain from round 0 to `upTo` in one call.
    /// @dev This is the provenance claim, executable by anyone. If it returns
    ///      false, some round's parent link was broken -- history was rewritten.
    function verifyLineage(uint256 upTo) external view returns (bool) {
        bytes32 prev = bytes32(0);
        for (uint256 i = 0; i <= upTo; i++) {
            Round storage r = rounds[i];
            if (!r.finalized) return false;
            if (i > 0 && r.parentHash != prev) return false;
            prev = r.globalModelHash;
        }
        return true;
    }
}
