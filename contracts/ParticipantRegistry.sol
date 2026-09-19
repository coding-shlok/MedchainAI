// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ParticipantRegistry
/// @notice Permissioned membership + stake + reputation for the federation.
/// @dev Reputation is stored in BASIS POINTS (0..10000) because Solidity has no
///      floats. The Python ReputationLedger quantises identically (mock_chain.to_bps)
///      so simulation and deployment agree to within 1 bps.
contract ParticipantRegistry {
    uint16 public constant BPS = 10_000;

    struct Participant {
        string  institution;
        uint16  reputationBps;
        uint256 stake;
        bool    registered;
        bool    active;
        uint256 registeredBlock;
    }

    address public admin;
    mapping(address => Participant) public participants;
    address[] public participantList;

    event ParticipantRegistered(address indexed who, string institution, uint256 stake);
    event ReputationUpdated(address indexed who, uint16 oldBps, uint16 newBps, uint256 round);
    event ParticipantSlashed(address indexed who, uint256 amount, string reason);
    event ParticipantDeactivated(address indexed who);

    modifier onlyAdmin() {
        require(msg.sender == admin, "not admin");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    /// @notice Join the federation by staking. Stake is what slashing burns, and
    ///         is the only thing making a Sybil identity costly -- without it an
    ///         attacker mints unlimited fresh identities and every reputation
    ///         scheme, including ours, is defeated for free.
    function register(string calldata institution) external payable {
        require(!participants[msg.sender].registered, "already registered");
        require(msg.value > 0, "stake required");
        participants[msg.sender] = Participant({
            institution: institution,
            reputationBps: 5_000,          // cold start = 0.5, matches ReputationConfig.initial
            stake: msg.value,
            registered: true,
            active: true,
            registeredBlock: block.number
        });
        participantList.push(msg.sender);
        emit ParticipantRegistered(msg.sender, institution, msg.value);
    }

    function updateReputation(address who, uint16 newBps, uint256 round) external onlyAdmin {
        require(participants[who].registered, "not registered");
        require(newBps <= BPS, "bps out of range");
        uint16 old = participants[who].reputationBps;
        participants[who].reputationBps = newBps;
        emit ReputationUpdated(who, old, newBps, round);
    }

    function slash(address who, uint256 amount, string calldata reason) external onlyAdmin {
        Participant storage p = participants[who];
        require(p.registered, "not registered");
        uint256 burned = amount > p.stake ? p.stake : amount;
        p.stake -= burned;
        emit ParticipantSlashed(who, burned, reason);
        if (p.stake == 0 && p.active) {
            p.active = false;
            emit ParticipantDeactivated(who);
        }
    }

    function isActive(address who) external view returns (bool) {
        return participants[who].active;
    }

    function count() external view returns (uint256) {
        return participantList.length;
    }
}
