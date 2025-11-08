import React, { useState, useEffect } from "react";
import axios from "axios";
import "./theme.css";

// Main App Component
export default function App() {
  const [screen, setScreen] = useState("dashboard"); // 'dashboard', 'add', 'certificate'
  const [certificateData, setCertificateData] = useState(null);

  const renderScreen = () => {
    switch (screen) {
      case "add":
        return (
          <AddGoldForm
            setScreen={setScreen}
            setCertificateData={setCertificateData}
          />
        );
      case "certificate":
        return <GoldCertificate data={certificateData} setScreen={setScreen} />;
      default:
        return <GoldVaultDashboard setScreen={setScreen} />;
    }
  };

  return (
    <div
      className="min-h-screen p-6"
      style={{ background: "linear-gradient(180deg,#071029 0%, #0b1220 100%)" }}
    >
      <div className="app-shell">
        <nav className="app-nav">
          <div className="brand">
            <h1>🪙 Temple Sona</h1>
            <small className="muted">Gold Vault & Digital Certificates</small>
          </div>

          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            {screen !== "dashboard" && (
              <button
                onClick={() => setScreen("dashboard")}
                className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg"
              >
                Back to Vault
              </button>
            )}
          </div>
        </nav>

        <main className="mt-6 page-card">{renderScreen()}</main>
      </div>
    </div>
  );
}

// Screen 1: Gold Vault Dashboard
function GoldVaultDashboard({ setScreen }) {
  const [vault, setVault] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch vault data
    axios
      .get("/api/gold/vault/user123") // 'user123' is a mock user
      .then((res) => {
        setVault(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading your Gold Vault...</div>;

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl">My Gold Vault</h2>
        <button
          onClick={() => setScreen("add")}
          className="bg-yellow-500 hover:bg-yellow-600 text-gray-900 font-bold py-3 px-6 rounded-lg text-lg"
        >
          ➕ Add New Gold
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {vault?.gold_items.map((item) => (
          <GoldItemCard key={item.token_id} item={item} />
        ))}
      </div>
    </div>
  );
}

// Reusable card for the dashboard
function GoldItemCard({ item }) {
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-4 flex">
      <img
        src={`https://via.placeholder.com/150/FFD700/000000?text=${item.type}`}
        alt={item.type}
        className="w-32 h-32 rounded-lg object-cover mr-4"
      />
      <div>
        <h3 className="text-xl font-bold">
          {item.type} (Token: {item.token_id})
        </h3>
        <p className="text-gray-400">
          {item.weight}g | {item.purity}K Purity
        </p>
        <p className="text-2xl font-bold text-green-400 mt-2">
          Value: ₹{item.value.toLocaleString("en-IN")}
        </p>
        {item.locked ? (
          <p className="text-red-500 font-bold">🔒 LOCKED (Loan Active)</p>
        ) : (
          <p className="text-lg text-yellow-300">
            Can Borrow: ₹{item.can_borrow.toLocaleString("en-IN")}
          </p>
        )}
      </div>
    </div>
  );
}

// Screen 2: Add Gold Item
function AddGoldForm({ setScreen, setCertificateData }) {
  const [file, setFile] = useState(null);
  const [goldType, setGoldType] = useState("Necklace");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleMint = async () => {
    if (!file) {
      setError("Please upload a photo of your gold.");
      return;
    }
    setLoading(true);
    setError("");

    try {
      // Step 1: Upload image (this also mocks AI detection)
      const formData = new FormData();
      formData.append("image", file);

      const uploadRes = await axios.post("/api/gold/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Step 2: Tokenize (mint) the NFT
      const tokenizeData = {
        user_id: "user123",
        gold_type: goldType,
        weight: uploadRes.data.detected_weight,
        purity: uploadRes.data.detected_purity,
        image_hash: uploadRes.data.image_hash,
      };

      const tokenizeRes = await axios.post("/api/gold/tokenize", tokenizeData);

      // Success! Show certificate
      setCertificateData({ ...tokenizeRes.data, goldType });
      setScreen("certificate");
    } catch (err) {
      console.error(err);
      setError("Minting failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-gray-800 p-8 rounded-lg">
      <h2 className="text-2xl font-bold mb-6">Create Digital Certificate</h2>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          1. Upload Gold Photo
        </label>
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-yellow-500 file:text-gray-900 hover:file:bg-yellow-600"
        />
      </div>

      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          2. Select Gold Type
        </label>
        <select
          value={goldType}
          onChange={(e) => setGoldType(e.target.value)}
          className="w-full bg-gray-700 text-white p-3 rounded-lg"
        >
          <option>Necklace</option>
          <option>Ring</option>
          <option>Bangle</option>
          <option>Earring</option>
          <option>Chain</option>
          <option>Coin</option>
        </select>
      </div>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      <button
        onClick={handleMint}
        disabled={loading}
        className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-6 rounded-lg text-lg disabled:bg-gray-500"
      >
        {loading ? "Minting on Polygon..." : "Create Digital Certificate"}
      </button>
    </div>
  );
}

// Screen 3: Digital Certificate
function GoldCertificate({ data, setScreen }) {
  const polygonScanUrl = `https://mumbai.polygonscan.com/tx/${data.transaction_hash}`;

  return (
    <div className="max-w-lg mx-auto bg-gradient-to-br from-yellow-300 to-yellow-600 p-1 rounded-lg shadow-2xl">
      <div className="bg-gray-800 p-8 rounded-lg">
        <h2 className="text-3xl font-bold text-center mb-4 text-yellow-400">
          Digital Gold Certificate
        </h2>

        <img
          src={`https://via.placeholder.com/300/FFD700/000000?text=${data.goldType}`}
          alt={data.goldType}
          className="w-full h-48 object-cover rounded-lg mb-4 border-4 border-yellow-500"
        />

        <p className="text-center text-lg">
          <strong>Token ID:</strong> {data.token_id}
        </p>
        <p className="text-center text-2xl font-bold my-4">
          Value: ₹{data.estimated_value.toLocaleString("en-IN")}
        </p>
        <p className="text-center text-xl text-green-400 mb-6">
          Borrowing Capacity: ₹{data.borrowing_capacity.toLocaleString("en-IN")}
        </p>

        <div className="text-center bg-green-600 p-2 rounded text-white font-bold">
          ✓ Secured on Polygon Blockchain
        </div>

        <a
          href={polygonScanUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="block w-full text-center bg-gray-700 hover:bg-gray-600 p-3 rounded-lg mt-4"
        >
          View Transaction on PolygonScan
        </a>

        <button
          onClick={() => alert("Loan modal would open!")}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg mt-4"
        >
          Apply for Loan
        </button>
      </div>
    </div>
  );
}
