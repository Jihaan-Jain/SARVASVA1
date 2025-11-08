import { ethers } from "hardhat";

async function main() {
  const goldNFT = await ethers.deployContract("GoldNFT");

  await goldNFT.waitForDeployment();

  console.log(`GoldNFT contract deployed to: ${goldNFT.target}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
