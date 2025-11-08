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
from pdf2image import convert_from_bytes
# --- TESSERACT OCR IMPORTS & CONFIGURATION (Free Local Solution) ---
import pytesseract
from PIL import Image
from io import BytesIO

# --- CRUCIAL FOR WINDOWS: Set the command path for the Tesseract executable ---
# You MUST verify this path matches your Tesseract installation directory.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
# --------------------------------------------------------------------------

# Load environment variables
load_dotenv()
lang = "en-IN" 

# Flask app setup
app = Flask(__name__, static_folder='static', template_folder="templates")
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Upload folder for audio files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 
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

# NOTE: Google Cloud Vision client initialization code REMOVED.

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """ Serve the frontend HTML file """
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

        lower_message = user_message_to_llm.lower()
        if (session["question_count"] >= 15 or lower_message.find("eligib") >= 0) and not session["assessment_provided"]:
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

        if request.path != '/translate':
            return {
                "error": response_data.get("error", {}).get("message", "Translation failed"),
                "request_id": response_data.get("error", {}).get("request_id", "unknown"),
                "details": response_data
            }
            
        return jsonify({
            "error": response_data.get("error", {}).get("message", "Translation failed"),
            "request_id": response_data.get("error", {}).get("request_id", "unknown"),
            "details": response_data
        }), 500

    except requests.exceptions.RequestException as e:
        if request.path != '/translate':
            return {"error": "API request failed", "details": str(e)}
            
        return jsonify({"error": "API request failed", "details": str(e)}), 500


def translate_long_text(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format):
    """Handle translation of texts longer than 1000 characters by splitting into chunks (Helper for /translate)"""
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
    is_api_call = request.path == '/translate'

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
        
        if is_api_call:
            response_data = response.get_json()
            if "translated_text" in response_data:
                translated_chunks.append(response_data["translated_text"])
            else:
                return response
        else:
            if "translated_text" in response:
                translated_chunks.append(response["translated_text"])
            else:
                return response
    
    full_translation = " ".join(translated_chunks)
    
    result_dict = {
        "translated_text": full_translation,
        "chunked_translation": True,
        "chunks_count": len(chunks)
    }
    
    if is_api_call:
        return jsonify(result_dict)
    
    return result_dict


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


# --- REPLACED FUNCTION: Now uses Tesseract (Local, Free) ---
# --- Corrected and Simplified perform_ocr Function ---

def perform_ocr(file_bytes, file_type):
    """Perform OCR using Tesseract, handling PDF to image conversion via Poppler."""
    raw_text = ""
    
    # We only check the file_type (MIME type from request). NO document_file.filename check.
    if 'pdf' in file_type.lower():
        try:
            # NOTE: If Poppler is not in the system PATH, you MUST specify poppler_path
            # IMPORTANT: The path below is illustrative. Replace it with your ACTUAL Poppler bin path.
            # Replace the path below with your correct path to the Poppler bin folder:
            poppler_path_win = r'C:\Users\jainj\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin' 
            
            images = convert_from_bytes(file_bytes, poppler_path=poppler_path_win) 
            
            for image in images:
                # Process each page (image) with Tesseract
                raw_text += pytesseract.image_to_string(image, lang='eng', config='--psm 3') + "\n\n"
        
        except Exception as e:
            # Re-raise the error to include details about the Poppler path failure
            raise Exception(f"PDF Handling Error: Poppler/PDF2Image failed. Ensure Poppler is installed and poppler_path is set in perform_ocr. Details: {e}")
            
    else: # Process as a standard image (PNG, JPEG)
        try:
            image = Image.open(BytesIO(file_bytes))
            raw_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        
        except Exception as e:
            raise Exception(f"Image Reading Error: Pillow/Tesseract failed. Details: {e}")

    if not raw_text.strip():
        raise Exception("OCR failed to extract any text from the document.")
        
    return raw_text


# NEW: Document Reader Endpoint
@app.route('/read-document', methods=['POST'])
def read_document():
    """Handle document upload, OCR, LLM simplification, and translation/TTS setup."""
    
    # NOTE: Check for vision_client initialization REMOVED.
    
    if 'document' not in request.files:
        return jsonify({'error': 'No document file uploaded'}), 400

    document_file = request.files['document']
    target_lang = request.form.get('language_code', 'en-IN')

    if document_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read file content into memory
    file_bytes = document_file.read()
    
    try:
        # --- 1. Perform OCR to extract text (using Tesseract) ---
        # document_file.content_type provides the MIME type (e.g., 'application/pdf')
        raw_text = perform_ocr(file_bytes, document_file.content_type) 
        
        if not raw_text.strip():
            return jsonify({'error': 'Could not extract text from the document. Please ensure the image/PDF is clear.'}), 400

        # --- 2. LLM Simplification (Groq) ---
        system_prompt_llm = f"""You are an expert financial explainer for first-time loan applicants, working in 'Explain Like I'm 18 Mode'. Your task is to analyze the following loan document text.
        
        **Your output MUST be formatted using standard Markdown syntax (e.g., ## for headings, * for lists, ** for bolding) to ensure clarity.**
        
        ## 💰 Summary of Key Loan Terms
        1. Summarize the **Key Terms** (Interest Rate, Tenure, EMI, Prepayment Penalty) in a bulleted list, using simple analogies.
        
        ## ⚠️ Risks and Commitments Explained
        2. Provide a **simple explanation** of the main risks and commitments in the document.
        
        3. Convert all financial jargon into plain-language and relatable examples.
        
        4. The final response must be in English for the next translation step.
        
        Document Text:
        ---
        {raw_text[:4000]}
        ---
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt_llm}],
            model="llama-3.3-70b-versatile"
        )

        english_explanation = chat_completion.choices[0].message.content
        
        if not english_explanation.strip():
            raise Exception("LLM failed to generate an explanation.")

        # --- 3. Translate to Target Vernacular Language (Sarvam AI) ---
        vernacular_explanation = english_explanation
        if target_lang != "en-IN":
            translation_result = translate_long_text(
                english_explanation,
                source_lang="en-IN",
                target_lang=target_lang,
                speaker_gender="Female",
                mode="formal",
                output_script="fully-native",
                numerals_format="international"
            )
            
            if "translated_text" in translation_result:
                vernacular_explanation = translation_result["translated_text"]
            else:
                logging.warning(f"Translation failed for document explanation, using English. Details: {translation_result.get('error')}")


        # --- 4. Return the result ---
        return jsonify({
            "raw_text": raw_text,
            "english_explanation": english_explanation,
            "vernacular_explanation": vernacular_explanation
        })

    except Exception as e:
        logging.error(f"Error in document reader: {str(e)}")
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500


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

        # --- UPDATED CONFIGURATION FOR VALID SARVAM SPEAKERS/MODELS ---
        LANGUAGE_CONFIG = {
            'en-IN': {"model": "bulbul:v2", "chunk_size": 500, "silence_bytes": 2000, "speaker": "anushka"},
            'hi-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "abhilash"},
            'ta-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "vidya"},
            'te-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "teja"},
            'kn-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "kavya"},
            'ml-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "arya"},
            'mr-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "sakshi"},
            'bn-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "ishita"},
            'gu-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "kiran"},
            'pa-IN': {"model": "bulbul:v2", "chunk_size": 300, "silence_bytes": 3000, "speaker": "ranjit"}
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)