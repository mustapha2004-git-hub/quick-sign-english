# SignBridge — Claude Context

## What this app does
BSL (British Sign Language) → English → Moroccan Darija → audio.
Use case: deaf person hands phone to a Moroccan seller; seller sees the translation spoken in Darija.

## How to run (THREE terminal windows)
```
# Window 1 — React app (port 8080)
npm run dev

# Window 2 — Sign→English proxy via Groq (port 3001)
node proxy.js

# Window 3 — Local AI: English→Darija + Darija TTS (port 3002)
# First-time only: pip install -r requirements.txt
python darija_server.py
```

## Stack
- React 19 + TypeScript + TanStack Start + Tailwind
- Vite dev server on port 8080
- `@mediapipe/tasks-vision` — GestureRecognizer for hand tracking + gesture classification
- Node.js proxy (`proxy.js`) on port 3001 — sign→English via Groq (`meta-llama/llama-4-scout-17b-16e-instruct`)
- Python Flask server (`darija_server.py`) on port 3002 — local models:
  - `BounharAbdelaziz/Terjman-Large-v2.0` — English→Moroccan Darija (seq2seq)
  - `HAMMALE/speecht5-darija` + `microsoft/speecht5_hifigan` — Darija TTS via sounddevice

## Key files
| File | Purpose |
|---|---|
| `src/components/SignBridgeApp.tsx` | Entire UI + MediaPipe loop + sign recognition |
| `proxy.js` | Node HTTP server — `/api/translate-signs`, `/api/translate-landmarks`, `/api/translate`, `/api/darija` |
| `public/templates.json` | Optional: pre-trained sign templates (cosine similarity matching). Not present by default. |
| `make_templates.py` | Script to generate templates.json from Kaggle ASL Signs dataset |
| `.env` | HF_TOKEN (HuggingFace, unused now), VITE_GEMINI_API_KEY |

## Sign recognition pipeline (in order of priority)
1. **GestureRecognizer** (MediaPipe ML model) — 7 gestures: Thumb_Up→good, Open_Palm→hello/stop, Victory→peace, Pointing_Up→me, ILoveYou→i love you, Closed_Fist→a, Thumb_Down→bad
2. **Template matching** (cosine similarity vs `templates.json`) — if file exists
3. **Rule-based fallback** (`ruleBasedSign()`) — finger state logic

## Translation flow
- **Camera mode**: record 5s → collect GestureRecognizer sign names → `/api/translate-signs` → Groq → English → `/api/darija` → Groq → Darija → TTS
- **Sentence builder mode**: sign words one-by-one (hold 1s each, drop hands between) → same translation flow
- **Upload mode**: extract 3 frames → `/api/translate` → Groq vision → English → Darija

## API keys (in proxy.js — hardcoded)
- Groq key is in `proxy.js` line 3. If it stops working, get a new one at console.groq.com

## Known issues / notes
- `ar-MA` TTS voice rarely available; app falls back to any `ar-*` voice automatically
- MediaPipe loads from CDN on first load (~10-15s); status dot: red=loading, yellow=ready, green=hands detected
- Canvas overlay is CSS-mirrored (`scale-x-[-1]`) same as video — text drawn with `ctx.scale(-1,1)` to stay readable
- The app only does static pose recognition — dynamic/motion signs (like "thank you" which involves movement) are not well recognized
- For more signs: find a pre-trained `.tflite` or `.onnx` model on GitHub/HuggingFace that takes 21 hand landmark points as input
