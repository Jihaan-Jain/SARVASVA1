// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract GoldNFT is ERC721, ERC721URIStorage, Ownable {
    uint256 public _nextTokenId;

    struct GoldDetails {
        uint256 weightInGrams;
        uint256 purity; // e.g., 22 for 22K
        uint256 estimatedValue;
        bool isLocked;
        address lender;
    }

    mapping(uint256 => GoldDetails) public goldTokenDetails;

    constructor() ERC721("Temple Sona Gold", "TSG") Ownable(msg.sender) {}

    function mintGoldNFT(
        address owner,
        uint256 weightInGrams,
        uint256 purity,
        string memory tokenURI, // This will be the IPFS hash for image/metadata
        uint256 estimatedValue
    ) public onlyOwner returns (uint256) {
        uint256 tokenId = _nextTokenId++;
        _safeMint(owner, tokenId);
        _setTokenURI(tokenId, tokenURI);

        goldTokenDetails[tokenId] = GoldDetails(
            weightInGrams,
            purity,
            estimatedValue,
            false,
            address(0)
        );

        return tokenId;
    }

    function lockForLoan(
        uint256 tokenId,
        uint256 loanAmount,
        address lender
    ) public returns (bool) {
        require(_exists(tokenId), "Token does not exist");
        require(ownerOf(tokenId) == msg.sender, "Not the owner");
        require(!goldTokenDetails[tokenId].isLocked, "Token already locked");

        goldTokenDetails[tokenId].isLocked = true;
        goldTokenDetails[tokenId].lender = lender;
        // You can add logic here to interact with a loan contract
        // For now, we just lock it.
        return true;
    }

    function unlockAfterRepayment(uint256 tokenId) public returns (bool) {
        require(_exists(tokenId), "Token does not exist");
        // Only the lender or contract owner can unlock
        require(msg.sender == goldTokenDetails[tokenId].lender || msg.sender == owner(), "Not authorized");

        goldTokenDetails[tokenId].isLocked = false;
        goldTokenDetails[tokenId].lender = address(0);
        return true;
    }

    function isLocked(uint256 tokenId) public view returns (bool) {
        require(_exists(tokenId), "Token does not exist");
        return goldTokenDetails[tokenId].isLocked;
    }

    function getGoldDetails(uint256 tokenId) public view returns (
        address owner,
        uint256 weight,
        uint256 purity,
        string memory imageHash, // tokenURI
        uint256 value,
        bool locked
    ) {
        require(_exists(tokenId), "Token does not exist");
        GoldDetails memory details = goldTokenDetails[tokenId];
        return (
            ownerOf(tokenId),
            details.weightInGrams,
            details.purity,
            tokenURI(tokenId),
            details.estimatedValue,
            details.isLocked
        );
    }

    // Required for ERC721URIStorage
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    // We need to override this function from ERC721
    function supportsInterface(bytes4 interfaceId) public view override(ERC721, ERC721URIStorage) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}