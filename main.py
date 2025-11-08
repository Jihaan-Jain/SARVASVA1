from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import requests
import json
import base64
from io import BytesIO
from dotenv import load_dotenv
import logging
from groq import Groq
import re
import os
import json
import time

# Load environment variables
load_dotenv()
lang = "en-IN"  # Default language code

# Flask app setup
app = Flask(__name__, static_folder='static', template_folder="templates")
CORS(app)

# ... after app = Flask(__name__) and CORS(app) ...

# --- BLOCKCHAIN (TEMPLE SONA) SETUP ---
try:
    # Load Contract ABI (Application Binary Interface)
    # We need to create this file first!
    with open('blockchain/artifacts/contracts/GoldNFT.sol/GoldNFT.json') as f:
        contract_json = json.load(f)
        GOLD_NFT_CONTRACT_ABI = contract_json['abi']

    POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL")
    BACKEND_WALLET_PRIVATE_KEY = os.getenv("BACKEND_WALLET_PRIVATE_KEY")
    GOLD_NFT_CONTRACT_ADDRESS = os.getenv("GOLD_NFT_CONTRACT_ADDRESS")

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0) # Required for PoA chains like Polygon

    backend_account = w3.eth.account.from_key(BACKEND_WALLET_PRIVATE_KEY)
    w3.eth.default_account = backend_account.address

    gold_nft_contract = w3.eth.contract(address=GOLD_NFT_CONTRACT_ADDRESS, abi=GOLD_NFT_CONTRACT_ABI)

    print("✅ Successfully connected to Polygon and loaded GoldNFT contract.")
    print(f"Backend wallet address: {backend_account.address}")

except Exception as e:
    print(f"🔥 FAILED TO INITIALIZE BLOCKCHAIN: {e}")
    print("Blockchain features will be disabled.")
    w3 = None
    gold_nft_contract = None
# --- END BLOCKCHAIN SETUP ---

# ... rest of your existing global variables (sessions, etc.) ...

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Upload folder for audio files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max file size

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Ensure API keys are set
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY is missing. Please set it in the environment variables.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in the environment variables.")


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
conversation_sessions = {}
TRANSLATE_API_URL = "https://api.sarvam.ai/translate"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """ Serve the frontend HTML file """
    # Assuming your HTML file is named index.html in the 'templates' folder
    return render_template("index.html")


@app.route('/set-language', methods=['POST'])
def set_language():
    """Set the default language for the application."""
    global lang

    data = request.json
    new_lang = data.get("language_code", "").strip()

    if not new_lang:
        return jsonify({"error": "Language code is required"}), 400

    print(f"Language set to: {new_lang}")
    lang = new_lang
    return jsonify({"message": f"Language changed to {lang}"}), 200


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot responses using Groq's LLaMA model."""
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        role = data.get("role", "user").strip()
        session_id = data.get("session_id", "default")
        reset = data.get("reset", False)

        if not user_message and user_message != "initial_greeting":
            return jsonify({"error": "Message cannot be empty"}), 400

        # Initialize or reset session if needed
        if reset or session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                "messages": [],
                "question_count": 0,
                "asked_questions": set(),
                "prompt_added": False,
                "unknown_answers": set(),
                "assessment_provided": False
            }

        session = conversation_sessions[session_id]

        # Add system prompt only once per conversation
        if not session["prompt_added"]:
            # NOTE: System prompt is long, keep it concise here for readability
            system_prompt = '''You are a friendly, professional, and highly knowledgeable loan assistant. Your goal is to help users understand their loan eligibility in an interactive and engaging manner. Start by warmly greeting the user and asking for the basic details like name and age etc and then ask required details step by step (e.g., type of loan, loan amount, tenure of loan, age, income, credit score, etc.) instead of requesting everything at once. Ask brief, clear questions to avoid overwhelming the user. Don't forget send only plain text no stars or any other special characters in the text.'''
            session["messages"].append({"role": "system", "content": system_prompt})
            session["prompt_added"] = True

        # Handle initial greeting separately
        if user_message == "initial_greeting":
            user_message_to_llm = "Start the conversation with a warm greeting and ask the first step question."
        else:
            user_message_to_llm = user_message
            session["messages"].append({"role": role, "content": user_message})

            if role == "user":
                session["question_count"] += 1

        # Prediction logic... (kept the same as your original code for continuity)
        lower_message = user_message_to_llm.lower()
        if (session["question_count"] >= 15 or lower_message.find("eligib") >= 0) and not session["assessment_provided"]:
            # The complex prediction prompt is omitted here for brevity but assumes your original logic handles it
            prediction_instruction = "Based on all information gathered so far, provide a final loan eligibility assessment. Ensure to consider type of loan, amount of loan, and loan tenure as mandatory factors. Provide full details on eligibility, required documents, and next steps."
            session["messages"].append({"role": "system", "content": prediction_instruction})
            session["assessment_provided"] = True

        chat_completion = client.chat.completions.create(
            messages=session["messages"],
            model="llama-3.3-70b-versatile"
        )

        bot_response = chat_completion.choices[0].message.content
        session["messages"].append({"role": "assistant", "content": bot_response})

        return jsonify({
            "response": bot_response,
            "questions_asked": session["question_count"],
            "session_id": session_id,
            "assessment_provided": session["assessment_provided"]
        })

    except Exception as e:
        logging.error(f"Unexpected error in /chat: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


def perform_translation(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format):
    """Perform translation request to Sarvam AI API (Helper for /translate)"""
    # ... [Your existing perform_translation logic remains here]
    try:
        payload = {
            "input": input_text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": speaker_gender,
            "mode": mode,
            "model": "mayura:v1",
            "enable_preprocessing": False,
            "output_script": output_script,
            "numerals_format": numerals_format
        }

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }

        response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers)
        response_data = response.json()
        
        if "translated_text" in response_data:
            return jsonify({
                "translated_text": response_data["translated_text"],
                "request_id": response_data.get("request_id", "unknown"),
                "source_language_code": response_data.get("source_language_code", "unknown")
            })

        return jsonify({
            "error": response_data.get("error", {}).get("message", "Translation failed"),
            "request_id": response_data.get("error", {}).get("request_id", "unknown"),
            "details": response_data
        }), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "API request failed", "details": str(e)}), 500


def translate_long_text(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format):
    """Handle translation of texts longer than 1000 characters by splitting into chunks (Helper for /translate)"""
    # ... [Your existing translate_long_text logic remains here]
    sentences = re.split(r'(?<=[.!?])\s+', input_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 950:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    translated_chunks = []
    for chunk in chunks:
        response = perform_translation(
            chunk, 
            source_lang, 
            target_lang, 
            speaker_gender, 
            mode, 
            output_script, 
            numerals_format
        )
        
        response_data = response.get_json() if hasattr(response, 'get_json') else response
        if "translated_text" in response_data:
            translated_chunks.append(response_data["translated_text"])
        else:
            return response
    
    full_translation = " ".join(translated_chunks)
    
    return jsonify({
        "translated_text": full_translation,
        "chunked_translation": True,
        "chunks_count": len(chunks)
    })


@app.route('/translate', methods=['POST'])
def translate_text():
    """API to translate text using Sarvam AI."""
    try:
        data = request.json
        input_text = data.get("input")
        source_lang = data.get("source_language_code", "").strip()
        target_lang = data.get("target_language_code", "").strip()
        speaker_gender = data.get("speaker_gender", "Female")
        mode = data.get("mode", "formal")
        output_script = data.get("output_script", "fully-native")
        numerals_format = data.get("numerals_format", "international")

        if not input_text or not input_text.strip():
            return jsonify({"error": "Input text is required"}), 400

        if len(input_text) > 1000:
            return translate_long_text(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format)
        
        return perform_translation(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format)

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """ Convert Speech to Text """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        audio_file.save(file_path)

        if os.stat(file_path).st_size == 0:
            os.remove(file_path)
            return jsonify({'error': 'Uploaded file is empty'}), 400

        global lang
        current_lang = lang
        logging.info(f"Using language for STT: {current_lang}")

        # Call Speech-to-Text API
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'audio/wav')}
            data = {
                'model': 'saarika:v2',
                'language_code': current_lang,
                'with_timestamps': 'false',
                'with_diarization': 'false',
                'num_speakers': '1'
            }
            headers = {'api-subscription-key': SARVAM_API_KEY}
            response = requests.post('https://api.sarvam.ai/speech-to-text', headers=headers, data=data, files=files)
            response.raise_for_status()

            result = response.json()
            logging.info(f"Speech-to-text response: {result}")

        if 'transcript' not in result:
            return jsonify({'error': 'No transcript found in response'}), 500

        transcription_text = result['transcript']
        detected_language = result.get('language_code', current_lang)

        response_data = {
            'transcription': transcription_text,
            'language_code': detected_language
        }

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logging.error(f"Speech-to-text API request failed: {str(e)}")
        return jsonify({'error': f'API request failed: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in STT: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert Text to Speech using Sarvam AI."""
    try:
        data = request.json
        text_list = data.get("inputs", [])
        if not text_list or not isinstance(text_list, list) or not text_list[0].strip():
            return jsonify({"error": "Text is required"}), 400

        text = text_list[0]
        
        currLang = data.get("target_language_code")
        # Ensure 'lang' global variable is the source if not specified in the request
        source_lang = data.get("source_language_code", lang)

        LANGUAGE_CONFIG = {
            'en-IN': {"model": "bulbul:v1", "chunk_size": 500, "silence_bytes": 2000, "speaker": "meera"},
            'hi-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'ta-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'te-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'kn-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'ml-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'mr-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'bn-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'gu-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"},
            'pa-IN': {"model": "bulbul:v1", "chunk_size": 300, "silence_bytes": 3000, "speaker": "meera"}
        }

        config = LANGUAGE_CONFIG.get(currLang, LANGUAGE_CONFIG['en-IN'])
        model = config["model"]
        chunk_size = config["chunk_size"]
        silence_bytes = config["silence_bytes"]
        speaker = config["speaker"]

        # 1. Translate text if source and target languages differ
        if source_lang != currLang:
            translate_payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": currLang,
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "bulbul:v1"
            }
            translate_headers = {
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY
            }
            try:
                translate_response = requests.post(TRANSLATE_API_URL, json=translate_payload, headers=translate_headers)
                if translate_response.status_code == 200:
                    translate_result = translate_response.json()
                    text = translate_result.get("translated_text", text)
                else:
                    logging.warning(f"Translation failed for TTS with status {translate_response.status_code}")
            except Exception as e:
                logging.error(f"Translation error in TTS: {str(e)}")
                # Continue with original text if translation fails

        # 2. Process text in chunks for TTS
        audio_data_combined = BytesIO()
        silence_chunk = b"\x00" * silence_bytes
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk in text_chunks:
            if not chunk.strip():
                continue

            request_body = {
                "inputs": [chunk],
                "target_language_code": currLang,
                "speaker": speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": model
            }
            if currLang == "en-IN":
                request_body["eng_interpolation_wt"] = 123

            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            }

            response = requests.post("https://api.sarvam.ai/text-to-speech", headers=headers, json=request_body)
            if response.status_code != 200:
                logging.error(f"TTS API error for chunk: {response.text}")
                continue

            result = response.json()
            if "audios" in result and result["audios"]:
                audio_data_combined.write(base64.b64decode(result["audios"][0]))
                audio_data_combined.write(silence_chunk)

        if audio_data_combined.getbuffer().nbytes <= silence_bytes:
            return jsonify({"error": "Failed to generate audio"}), 500

        audio_data_combined.seek(0)
        return send_file(audio_data_combined, mimetype="audio/mpeg")

    except requests.exceptions.RequestException as e:
        logging.error(f"TTS API request failed: {str(e)}")
        return jsonify({"error": "API request failed", "details": str(e)}), 500

    except Exception as e:
        logging.error(f"Unexpected error in TTS: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
# -----------------------------------------------
# 🪙 TEMPLE SONA (GOLD) API ENDPOINTS
# -----------------------------------------------

@app.route('/api/gold/price', methods=['GET'])
def get_gold_price():
    """
    Mocks a live gold price API.
    """
    # In a real app, you'd fetch this from a live API
    return jsonify({
        "price_per_gram_22k": 6500, # Mock price
        "price_per_gram_24k": 7000,
        "last_updated": "2025-11-08T10:30:00Z"
    })

@app.route('/api/gold/upload', methods=['POST'])
def upload_gold_image():
    """
    Mocks image upload, AI detection, and IPFS pinning.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image = request.files['image']

    # In a real app:
    # 1. Upload image to AWS S3 or Pinata (IPFS)
    # 2. Get the URL/hash
    # 3. Run AI detection

    # For this demo, we MOCK everything:
    print(f"Received image: {image.filename}")

    # Mock IPFS hash
    mock_ipfs_hash = f"QmXyz...{int(time.time())}" # Unique mock hash
    mock_image_url = f"https://gateway.pinata.cloud/ipfs/{mock_ipfs_hash}"

    # Mock AI detection
    mock_weight = 25.5
    mock_purity = 22

    return jsonify({
        "image_url": mock_image_url,
        "image_hash": mock_ipfs_hash, # This is our tokenURI
        "detected_weight": mock_weight,
        "detected_purity": mock_purity
    })

@app.route('/api/gold/tokenize', methods=['POST'])
def tokenize_gold():
    """
    Mints the Gold NFT on the blockchain.
    """
    if not gold_nft_contract:
        return jsonify({"error": "Blockchain service not initialized"}), 500

    data = request.json
    user_id = data.get('user_id') # You'd get this from user auth
    weight = float(data.get('weight'))
    purity = int(data.get('purity'))
    image_hash = data.get('image_hash') # This is the "tokenURI"

    # Mock owner address (in real app, use authenticated user's wallet)
    # For now, we mint it TO our backend wallet for demo purposes
    owner_address = backend_account.address 

    # Mock value calculation
    estimated_value = weight * 6500 # Using mock 22k price
    borrowing_capacity = estimated_value * 0.7

    try:
        # Prepare the transaction
        tx = gold_nft_contract.functions.mintGoldNFT(
            owner_address,
            int(weight * 100), # Store weight in mg or basis points
            purity,
            image_hash,
            int(estimated_value)
        ).build_transaction({
            'from': backend_account.address,
            'nonce': w3.eth.get_transaction_count(backend_account.address),
            'gas': 500000,
            'gasPrice': w3.eth.gas_price
        })

        # Sign and send the transaction
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=BACKEND_WALLET_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        # Get the token ID from the event logs (this is a bit advanced)
        # A simpler way is to have the contract return it, which ours does.
        # But let's get it from the receipt for robustness
        # This part is tricky. Let's assume the last token ID for simplicity for now.
        # A better way: emit an event. Our contract returns it.
        # Let's just... guess the token ID for the demo.
        # token_id = gold_nft_contract.functions._nextTokenId().call() - 1 
        # This is cleaner but still racy.

        # Let's just return the hash for now.

        return jsonify({
            "message": "Gold tokenized successfully!",
            "token_id": "GLD-SOON", # Mocking this, getting return value from tx is complex
            "nft_address": GOLD_NFT_CONTRACT_ADDRESS,
            "transaction_hash": tx_hash.hex(),
            "estimated_value": estimated_value,
            "borrowing_capacity": borrowing_capacity
        })

    except Exception as e:
        print(f"🔥 Token minting failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gold/vault/<user_id>', methods=['GET'])
def get_gold_vault(user_id):
    """
    Mocks fetching a user's gold items.
    In a real app, you would query the blockchain for all tokens
    owned by the user's address.
    """
    # This is hard because we minted TO the backend.
    # We will return MOCK data for the demo, as requested.
    demo_gold_items = [
      {
        "token_id": "GLD-2024-001",
        "type": "Necklace",
        "weight": 25.3,
        "purity": 22,
        "value": 164450,
        "can_borrow": 115115,
        "image_url": "https://gateway.pinata.cloud/ipfs/QmXyz...mock1",
        "blockchain_hash": "0x7a9f...3d2c",
        "locked": False
      },
      {
        "token_id": "GLD-2024-002", 
        "type": "Bangles",
        "weight": 18.0,
        "purity": 22,
        "value": 117000,
        "can_borrow": 81900,
        "image_url": "https://gateway.pinata.cloud/ipfs/QmXyz...mock2",
        "blockchain_hash": "0x8b3e...4f1a",
        "locked": True
      }
    ]

    return jsonify({
        "gold_items": demo_gold_items,
        "total_value": 281450,
        "total_borrowing_capacity": 197015
    })

@app.route('/api/gold/borrow', methods=['POST'])
def borrow_against_gold():
    """
    Mocks applying for a loan and locking the NFT.
    """
    data = request.json
    token_id = data.get('token_id')
    loan_amount = data.get('loan_amount')

    # In a real app, you would call the `lockForLoan` function
    # on the smart contract.

    print(f"Mock loan application for {token_id} for {loan_amount}")

    return jsonify({
        "loan_id": f"LOAN-{int(time.time())}",
        "monthly_emi": 6289, # Mocked
        "interest_rate": 8.5,
        "gold_locked": True,
        "approval_status": "pending"
    })


if __name__ == '__main__':
    # Changed host and removed ssl_context for standard local development
    app.run(host='127.0.0.1', port=5000, debug=True)