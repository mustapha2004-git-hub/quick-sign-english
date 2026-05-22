"""
darija_server.py — Local AI server for SignBridge  (port 3002)

Endpoints:
  POST /api/translate-and-speak  — English → Darija text + plays audio
  POST /api/speak                — Darija text → plays audio
  GET  /api/health               — returns {"status":"ok"} once models are loaded
"""

import os
import traceback
import threading
import sounddevice as sd
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or None

app = Flask(__name__)
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading English→Darija translation model (BounharAbdelaziz/Terjman-Large-v2.0)…")
trans_tokenizer = AutoTokenizer.from_pretrained(
    "BounharAbdelaziz/Terjman-Large-v2.0", token=HF_TOKEN
)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(
    "BounharAbdelaziz/Terjman-Large-v2.0", token=HF_TOKEN
)
trans_model.eval()

print("Loading Darija TTS model (HAMMALE/speecht5-darija)…")
tts_processor = SpeechT5Processor.from_pretrained(
    "HAMMALE/speecht5-darija", token=HF_TOKEN
)
tts_model = SpeechT5ForTextToSpeech.from_pretrained(
    "HAMMALE/speecht5-darija", token=HF_TOKEN
)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", token=HF_TOKEN)
tts_model.eval()
vocoder.eval()

# Speaker embedding — use a real x-vector so TTS produces audible output
# (zero embeddings often result in silence with SpeechT5)
try:
    from datasets import load_dataset
    print("Loading speaker embeddings from CMU Arctic dataset…")
    _xvec_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(_xvec_ds[7306]["xvector"]).unsqueeze(0)
    print("Speaker embeddings loaded.")
except Exception as _e:
    print(f"Could not load xvectors ({_e}), using fallback embedding.")
    # Normalised random embedding — better than zeros for SpeechT5
    _rng = torch.Generator().manual_seed(42)
    _raw = torch.randn((512,), generator=_rng)
    speaker_embeddings = (_raw / _raw.norm()).unsqueeze(0)

print("All models loaded — server ready on http://localhost:3002")

# ── Helpers ───────────────────────────────────────────────────────────────────
TTS_MAX_CHARS = 200  # SpeechT5 crashes on very long token sequences


def translate_to_darija(english_text: str) -> str:
    input_ids = trans_tokenizer(
        english_text, return_tensors="pt", truncation=True, max_length=512
    ).input_ids
    with torch.no_grad():
        output_ids = trans_model.generate(input_ids, max_new_tokens=256)
    return trans_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def speak_darija(darija_text: str) -> None:
    """Runs in a background thread — logs all errors so nothing is silently swallowed."""
    try:
        text = darija_text[:TTS_MAX_CHARS]
        print(f"[TTS] Generating speech for: {text!r}")

        inputs = tts_processor(text=text, return_tensors="pt", truncation=True)
        ids = inputs["input_ids"]
        print(f"[TTS] Token count: {ids.shape[1]}")

        if ids.shape[1] == 0:
            print("[TTS] Empty token sequence — skipping.")
            return

        with torch.no_grad():
            speech = tts_model.generate_speech(ids, speaker_embeddings, vocoder=vocoder)

        audio = speech.detach().cpu().numpy().astype(np.float32)
        print(f"[TTS] Audio shape: {audio.shape}, max amplitude: {np.abs(audio).max():.4f}")

        if np.abs(audio).max() < 1e-4:
            print("[TTS] Warning: audio appears to be silent (very low amplitude).")

        sd.play(audio, samplerate=16000)
        sd.wait()   # Keep thread alive until audio finishes
        print("[TTS] Playback complete.")

    except Exception:
        print("[TTS] Error during TTS/playback:")
        traceback.print_exc()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/translate-and-speak", methods=["POST"])
def translate_and_speak():
    try:
        english_text = (request.json or {}).get("text", "")
        if not english_text:
            return jsonify({"error": "No text provided"}), 400

        print(f"[Translate] {english_text!r}")
        darija_text = translate_to_darija(english_text)
        print(f"[Darija]    {darija_text!r}")

        threading.Thread(target=speak_darija, args=(darija_text,), daemon=False).start()

        return jsonify({"darija": darija_text})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Translation failed"}), 500


@app.route("/api/speak", methods=["POST"])
def speak():
    try:
        darija_text = (request.json or {}).get("text", "")
        if not darija_text:
            return jsonify({"error": "No text provided"}), 400

        threading.Thread(target=speak_darija, args=(darija_text,), daemon=False).start()
        return jsonify({"status": "playing"})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "TTS failed"}), 500


if __name__ == "__main__":
    app.run(port=3002, debug=False, use_reloader=False)
