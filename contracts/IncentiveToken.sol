// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title IncentiveToken
/// @notice Minimal ERC-20-style contribution token minted per round.
/// @dev Minted proportional to (reputation x data volume) -- see
///      ReputationLedger.reward_shares. Neither factor alone is safe: reputation
///      alone lets a one-image client earn as much as a ten-thousand-image one,
///      and volume alone is the classic FL incentive failure where fabricating
///      samples pays. Sample counts are self-reported, so this is only as strong
///      as the registry's admission control -- state that limitation in the paper.
contract IncentiveToken {
    string public constant name = "MedChain Contribution Token";
    string public constant symbol = "MCT";
    uint8  public constant decimals = 18;

    uint256 public totalSupply;
    address public minter;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => bool) public roundPaid;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event RewardsDistributed(uint256 indexed round, uint256 pool, uint256 recipients);

    constructor() {
        minter = msg.sender;
    }

    /// @dev `roundPaid` makes distribution idempotent. A retried transaction
    ///      after a chain reorg would otherwise double-mint the entire pool.
    function distributeRewards(uint256 round, address[] calldata to, uint256[] calldata amounts)
        external
    {
        require(msg.sender == minter, "not minter");
        require(to.length == amounts.length, "length mismatch");
        require(!roundPaid[round], "round already paid");
        roundPaid[round] = true;

        uint256 pool;
        for (uint256 i = 0; i < to.length; i++) {
            balanceOf[to[i]] += amounts[i];
            pool += amounts[i];
            emit Transfer(address(0), to[i], amounts[i]);
        }
        totalSupply += pool;
        emit RewardsDistributed(round, pool, to.length);
    }

    function transfer(address to, uint256 value) external returns (bool) {
        require(balanceOf[msg.sender] >= value, "insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
}
