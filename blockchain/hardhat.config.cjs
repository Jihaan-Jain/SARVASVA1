require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.20",
  networks: {
    "polygon-mumbai": {
      url: process.env.POLYGON_MUMBAI_RPC_URL,
      accounts: [process.env.WALLET_PRIVATE_KEY],
    },
  },
};
