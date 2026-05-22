import { useCallback, useEffect, useRef, useState } from "react";
import { Volume2, RotateCcw, Loader2, Upload, Copy, Check, Square, Type, Ear } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { GestureRecognizer, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

type Screen = "record" | "results";
type Mode = "camera" | "upload";

// ── Template-based recognizer (loaded from /public/templates.json) ──────────
let _templates: Record<string, number[]> | null = null;
let _templatesLoading = false;

function loadTemplates() {
  if (_templates || _templatesLoading) return;
  _templatesLoading = true;
  fetch("/templates.json")
    .then(r => r.ok ? r.json() : null)
    .then(data => { if (data) _templates = data; })
    .catch(() => {});
}

function normalizeHand(lm: Array<{ x: number; y: number; z: number }>, use3D = false): number[] | null {
  if (use3D) {
    // 3D normalization → 63 features (21 landmarks × 3 coords)
    let pts = lm.map(p => [p.x, p.y, p.z]);
    const [wx, wy, wz] = pts[0];
    pts = pts.map(p => [p[0] - wx, p[1] - wy, p[2] - wz]);
    const scale = Math.hypot(pts[9][0], pts[9][1], pts[9][2]); // wrist → middle MCP
    if (scale < 1e-6) return null;
    return pts.map(p => [p[0] / scale, p[1] / scale, p[2] / scale]).flat();
  }
  // 2D normalization → 42 features (backward compatible with make_templates.py)
  let pts = lm.map(p => [p.x, p.y]);
  const wx = pts[0][0], wy = pts[0][1];
  pts = pts.map(p => [p[0] - wx, p[1] - wy]);
  const scale = Math.hypot(pts[9][0], pts[9][1]); // wrist → middle MCP
  if (scale < 1e-6) return null;
  return pts.map(p => [p[0] / scale, p[1] / scale]).flat();
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

// ── Rule-based fallback (works without templates.json) ───────────────────────
type FingerState = "ext" | "half" | "bent";

function ruleBasedSign(lm: Array<{ x: number; y: number; z: number }>): string {
  const tipIds = [4, 8, 12, 16, 20];
  const pipIds = [3, 7, 11, 15, 19];
  const mcpIds = [2, 5, 9, 13, 17];
  const state = (fi: number): FingerState => {
    const tip = lm[tipIds[fi]], pip = lm[pipIds[fi]], mcp = lm[mcpIds[fi]];
    if (fi === 0) { const d = Math.abs(tip.x - lm[0].x); return d > 0.12 ? "ext" : d > 0.06 ? "half" : "bent"; }
    return tip.y < mcp.y ? "ext" : pip.y < mcp.y ? "half" : "bent";
  };
  const [th, ix, mi, ri, pi] = [0, 1, 2, 3, 4].map(state) as FingerState[];
  const e: FingerState = "ext", b: FingerState = "bent";
  const pinch = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.06;
  const atFace = lm[0].y < 0.38;
  if (ix === e && mi === e && ri === e && pi === e) return atFace ? "hello" : "stop";
  if (th === e && ix === e && mi === b && ri === b && pi === e) return "i love you";
  if (th === e && ix === e && mi === b && ri === b && pi === b) return "l";
  if (th === e && ix === b && mi === b && ri === b && pi === e) return "y";
  if (ix === e && mi === b && ri === b && pi === b) return "me";
  if (th === e && ix === b && mi === b && ri === b && pi === b) return "good";
  if (th !== e && ix === e && mi === e && ri === b && pi === b) return "peace";
  if (th !== e && ix === e && mi === e && ri === e && pi === b) return "3";
  if (pinch && mi === e && ri === e && pi === e) return "ok";
  if (th !== e && ix === b && mi === b && ri === b && pi === e) return "i";
  if (ix === b && mi === b && ri === b && pi === b) return "a";
  return "";
}

const CONFIDENCE_THRESHOLD = 0.93;

// Hoisted to module scope — avoids recreation on every animation frame (~60x/sec)
const GESTURE_MAP: Record<string, string> = {
  Thumb_Up:    "good",
  Thumb_Down:  "bad",
  Open_Palm:   "hello",  // refined at runtime based on wrist position
  Victory:     "peace",
  Pointing_Up: "me",
  ILoveYou:    "i love you",
  Closed_Fist: "a",
};

function recognizeHandSign(lm: Array<{ x: number; y: number; z: number }>): string {
  // Try template matching first
  if (_templates) {
    // Auto-detect 2D (42) vs 3D (63) based on first template's length
    const firstTmpl = Object.values(_templates)[0];
    const use3D = firstTmpl ? firstTmpl.length === 63 : false;
    const norm = normalizeHand(lm, use3D);
    if (norm) {
      let best = "", bestScore = 0;
      for (const [sign, tmpl] of Object.entries(_templates)) {
        if (tmpl.length !== norm.length) continue; // skip shape mismatch
        const score = cosineSimilarity(norm, tmpl);
        if (score > bestScore) { bestScore = score; best = sign; }
      }
      if (bestScore >= CONFIDENCE_THRESHOLD) return best;
    }
  }
  // Fall back to rule-based
  return ruleBasedSign(lm);
}

async function translateFromSigns(signs: string[]): Promise<string> {
  const unique = signs.filter((s, i) => i === 0 || s !== signs[i - 1]);
  const res = await fetch("http://localhost:3001/api/translate-signs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signs: unique }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.text;
}

async function translateFromFrames(frames: string[]): Promise<string> {
  const keyFrames = frames.length <= 3
    ? frames
    : [frames[0], frames[Math.floor(frames.length / 2)], frames[frames.length - 1]];

  // Use Google Gemini for vision-based translation (can actually see images, unlike text-only LLMs)
  const { GoogleGenAI } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });

  const contents = [
    ...keyFrames.map(frame => ({
      inlineData: {
        data: frame.replace(/^data:image\/jpeg;base64,/, ""),
        mimeType: "image/jpeg" as const
      }
    })),
    { text: `These are 3 frames (start, middle, end) from a short video of a person
     using British Sign Language (BSL).
     BSL uses two hands and has different grammar from English.
     Translate what the person is signing into natural English text.
     Return ONLY the translated English sentence, nothing else.
     If signs are unclear, return your best interpretation.` }
  ];

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents,
  });

  return response.text ?? "Could not interpret the signs.";
}

async function translateToDarija(englishText: string): Promise<string> {
  const res = await fetch("http://localhost:3002/api/translate-and-speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: englishText }),
  });
  if (!res.ok) return englishText;
  const data = await res.json();
  return data.darija || englishText;
}

function extractFramesFromVideo(file: File): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    const url = URL.createObjectURL(file);
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.load();
    video.onloadedmetadata = () => {
      const duration = video.duration;
      const times = [0.1, duration / 2, Math.max(duration - 0.1, 0.2)];
      const frames: string[] = [];
      const canvas = document.createElement("canvas");
      const captureAt = (index: number) => {
        if (index >= times.length) { URL.revokeObjectURL(url); resolve(frames); return; }
        video.currentTime = times[index];
        video.onseeked = () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext("2d")?.drawImage(video, 0, 0);
          frames.push(canvas.toDataURL("image/jpeg", 0.7));
          captureAt(index + 1);
        };
      };
      captureAt(0);
    };
    video.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not load video.")); };
  });
}

export function SignBridgeApp() {
  const [screen, setScreen] = useState<Screen>("record");
  const [mode, setMode] = useState<Mode>("camera");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translation, setTranslation] = useState("");
  const [darija, setDarija] = useState("");
  const [isDarijaLoading, setIsDarijaLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [mediapipeReady, setMediapipeReady] = useState(false);
  const [handsDetected, setHandsDetected] = useState(false);
  const [sentenceMode, setSentenceMode] = useState(false);
  const [sentenceWords, setSentenceWords] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const landmarkerRef = useRef<GestureRecognizer | null>(null);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);
  const animFrameRef = useRef<number>(0);
  const lastErrorLogRef = useRef<number>(0);
  const uploadedVideoUrlRef = useRef<string | null>(null);
  const recognizedSignsRef = useRef<string[]>([]);
  const currentSignRef = useRef<string>("");
  const [currentSign, setCurrentSign] = useState<string>("");
  const sentenceModeRef = useRef(false);
  const sentenceWordsRef = useRef<string[]>([]);
  const pendingSignRef = useRef<string>("");
  const pendingSignStartRef = useRef<number>(0);
  const lastAddedSignRef = useRef<string>("");

  // Load sign templates in background
  useEffect(() => { loadTemplates(); }, []);

  // Load MediaPipe on mount
  useEffect(() => {
    async function load() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const hl = await GestureRecognizer.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        landmarkerRef.current = hl;
        setMediapipeReady(true);
      } catch (e) {
        console.error("MediaPipe failed to load:", e);
      }
    }
    load();
  }, []);

  // Keep sentenceModeRef in sync
  useEffect(() => { sentenceModeRef.current = sentenceMode; }, [sentenceMode]);

  // Camera setup
  useEffect(() => {
    if (screen !== "record" || mode !== "camera") return;
    let active = true;
    let localStream: MediaStream | null = null;
    (async () => {
      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 640 } },
          audio: false,
        });
        if (!active) { localStream.getTracks().forEach(t => t.stop()); return; }
        setStream(localStream);
        if (videoRef.current) videoRef.current.srcObject = localStream;
      } catch {
        setCameraError("Unable to access camera. Please grant permission and reload.");
      }
    })();
    return () => {
      active = false;
      if (localStream) localStream.getTracks().forEach(t => t.stop());
      setStream(null);
    };
  }, [screen, mode]);

  // MediaPipe detection + drawing loop
  useEffect(() => {
    if (screen !== "record" || mode !== "camera") {
      cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const detect = () => {
      const video = videoRef.current;
      const overlay = overlayCanvasRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !overlay || !landmarker || video.readyState < 2) {
        animFrameRef.current = requestAnimationFrame(detect);
        return;
      }

      try {
        const results = landmarker.recognizeForVideo(video, performance.now());

        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 640;
        const ctx = overlay.getContext("2d");
        if (!ctx) { animFrameRef.current = requestAnimationFrame(detect); return; }
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const detected = results.landmarks.length > 0;
        setHandsDetected(detected);

        if (detected) {
          // Cache DrawingUtils per canvas context to avoid per-frame allocation
          if (!drawingUtilsRef.current) drawingUtilsRef.current = new DrawingUtils(ctx);
          const drawingUtils = drawingUtilsRef.current;
          let topSign = "";
          for (let hi = 0; hi < results.landmarks.length; hi++) {
            const landmarks = results.landmarks[hi];
            drawingUtils.drawConnectors(landmarks, GestureRecognizer.HAND_CONNECTIONS, {
              color: "#00FF00",
              lineWidth: 3,
            });
            drawingUtils.drawLandmarks(landmarks, {
              color: "#FF0000",
              fillColor: "#FF0000",
              radius: 5,
            });

            // GestureRecognizer result first, then template/rule fallback
            const gestureCategory = results.gestures?.[hi]?.[0];
            const gestureScore = gestureCategory?.score ?? 0;
            const gestureName = gestureCategory?.categoryName ?? "None";

            // Open_Palm context: face-level = "hello", lower = "stop"
            let mappedGesture = GESTURE_MAP[gestureName] ?? "";
            if (gestureName === "Open_Palm") {
              mappedGesture = landmarks[0].y < 0.38 ? "hello" : "stop";
            }

            const signLabel =
              (gestureScore > 0.75 && gestureName !== "None" ? mappedGesture : "") ||
              recognizeHandSign(landmarks);

            if (signLabel) {
              topSign = signLabel;
              const wx = landmarks[0].x * overlay.width;
              const wy = landmarks[0].y * overlay.height;
              ctx.save();
              ctx.translate(wx, wy - 40);
              ctx.scale(-1, 1); // undo CSS mirror so text reads correctly
              ctx.font = "bold 15px sans-serif";
              const tw = ctx.measureText(signLabel).width;
              ctx.fillStyle = "rgba(0,0,0,0.7)";
              ctx.fillRect(-tw / 2 - 8, -20, tw + 16, 26);
              ctx.fillStyle = "#FFFFFF";
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillText(signLabel, 0, -18);
              ctx.restore();
            }
          }

          if (topSign !== currentSignRef.current) {
            currentSignRef.current = topSign;
            setCurrentSign(topSign);
          }

          // Sentence building mode: hold a sign for 1 s to capture it
          const now = performance.now();
          if (sentenceModeRef.current) {
            if (topSign) {
              if (topSign !== pendingSignRef.current) {
                pendingSignRef.current = topSign;
                pendingSignStartRef.current = now;
              }
              if (topSign !== lastAddedSignRef.current && now - pendingSignStartRef.current > 1000) {
                const newWords = [...sentenceWordsRef.current, topSign];
                sentenceWordsRef.current = newWords;
                setSentenceWords([...newWords]);
                lastAddedSignRef.current = topSign;
              }
            } else {
              pendingSignRef.current = "";
              lastAddedSignRef.current = "";
            }
          }


        } else {
          if (currentSignRef.current !== "") {
            currentSignRef.current = "";
            setCurrentSign("");
          }
          if (sentenceModeRef.current) {
            pendingSignRef.current = "";
            lastAddedSignRef.current = "";
          }
        }
      } catch (err) {
        // Throttled error logging — max once per 3 seconds to avoid console spam
        const now = performance.now();
        if (now - lastErrorLogRef.current > 3000) {
          console.warn("[MediaPipe] Detection frame error:", err);
          lastErrorLogRef.current = now;
        }
      }

      animFrameRef.current = requestAnimationFrame(detect);
    };

    animFrameRef.current = requestAnimationFrame(detect);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [screen, mode, mediapipeReady]);

  const finishTranslation = useCallback(async (
    signs?: string[],
    fallbackFrames?: string[]
  ) => {
    setScreen("results");
    setIsTranslating(true);
    setTranslation("");
    setDarija("");

    let englishText = "";
    try {
      const signSequence = signs ?? recognizedSignsRef.current;
      if (signSequence.length > 0) {
        englishText = await translateFromSigns(signSequence);
      } else if (fallbackFrames && fallbackFrames.length > 0) {
        englishText = await translateFromFrames(fallbackFrames);
      } else {
        throw new Error("No data to translate");
      }
      setTranslation(englishText);
    } catch (e) {
      console.error(e);
      setTranslation("Sorry, translation failed. Please try again.");
      setIsTranslating(false);
      return;
    }
    setIsTranslating(false);

    setIsDarijaLoading(true);
    try {
      const darijaText = await translateToDarija(englishText);
      setDarija(darijaText);
    } catch (e) {
      console.error("Darija failed:", e);
    } finally {
      setIsDarijaLoading(false);
    }
  }, []);

  const startSentenceBuilding = useCallback(() => {
    sentenceWordsRef.current = [];
    pendingSignRef.current = "";
    lastAddedSignRef.current = "";
    setSentenceWords([]);
    setSentenceMode(true);
  }, []);

  const stopSentenceBuilding = useCallback(async () => {
    sentenceModeRef.current = false;
    setSentenceMode(false);
    pendingSignRef.current = "";
    lastAddedSignRef.current = "";
    const words = [...sentenceWordsRef.current];
    if (words.length === 0) return;
    await finishTranslation(words);
  }, [finishTranslation]);

  const handleVideoUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (uploadedVideoUrlRef.current) URL.revokeObjectURL(uploadedVideoUrlRef.current);
    uploadedVideoUrlRef.current = URL.createObjectURL(file);
    setUploadedFileName(file.name);
    try {
      const frames = await extractFramesFromVideo(file);
      await finishTranslation([], frames);
    } catch (e) {
      console.error(e);
      setScreen("results");
      setTranslation("Sorry, could not read the video file.");
    }
  }, [finishTranslation]);

  const speak = useCallback(async () => {
    const textToSpeak = darija || translation;
    if (!textToSpeak) return;
    try {
      await fetch("http://localhost:3002/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToSpeak }),
      });
    } catch (e) {
      console.error("TTS failed:", e);
    }
  }, [darija, translation]);

  const copyToClipboard = useCallback(() => {
    const text = darija ? `${translation}\n\n${darija}` : translation;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [translation, darija]);

  const recordAgain = useCallback(() => {
    window.speechSynthesis?.cancel();
    setTranslation("");
    setDarija("");
    setUploadedFileName(null);
    setHandsDetected(false);
    setSentenceWords([]);
    sentenceWordsRef.current = [];
    setSentenceMode(false);
    sentenceModeRef.current = false;
    setScreen("record");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col items-center px-4 py-6 sm:py-10 relative">
      <div className="w-full max-w-xl flex justify-end mb-4">
        <Link
          to="/speech-to-text"
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
        >
          <Ear className="w-4 h-4" />
          Hearing Mode (Speech to Text)
        </Link>
      </div>

      <header className="text-center mb-6 sm:mb-10">
        <h1 className="text-4xl sm:text-6xl font-bold tracking-tight">SignBridge</h1>
        <p className="mt-2 text-base sm:text-lg text-muted-foreground">
          British Sign Language Translator
        </p>
      </header>

      {screen === "record" ? (
        <section className="w-full max-w-xl flex flex-col items-center gap-6">

          {/* Mode toggle */}
          <div className="flex rounded-xl border border-border overflow-hidden w-full max-w-xs">
            <button type="button" onClick={() => setMode("camera")}
              className={`flex-1 py-2.5 text-sm font-medium transition-colors ${mode === "camera" ? "bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:text-foreground"}`}>
              Live Camera
            </button>
            <button type="button" onClick={() => setMode("upload")}
              className={`flex-1 py-2.5 text-sm font-medium transition-colors ${mode === "upload" ? "bg-primary text-primary-foreground" : "bg-card text-muted-foreground hover:text-foreground"}`}>
              Upload Video
            </button>
          </div>

          {mode === "camera" ? (
            <>
              {/* MediaPipe status */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className={`w-2 h-2 rounded-full ${mediapipeReady ? (handsDetected ? "bg-green-500" : "bg-yellow-500") : "bg-red-500 animate-pulse"}`} />
                {!mediapipeReady ? "Loading hand tracker..." : handsDetected ? "Hands detected" : "Show your hands to the camera"}
              </div>

              {/* Live sign label */}
              {currentSign && (
                <div className="text-base font-semibold px-5 py-1.5 rounded-full bg-primary text-primary-foreground tracking-wide">
                  {currentSign}
                </div>
              )}

              {/* Camera + overlay */}
              <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-card border border-border shadow-2xl">
                {cameraError ? (
                  <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-muted-foreground">
                    {cameraError}
                  </div>
                ) : (
                  <>
                    <video ref={videoRef} autoPlay playsInline muted
                      className="absolute inset-0 w-full h-full object-cover scale-x-[-1]" />
                    {/* Landmark overlay — mirrored same as video */}
                    <canvas ref={overlayCanvasRef}
                      className="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none" />
                  </>
                )}
              </div>

              {sentenceMode ? (
                <>
                  {/* Word chips */}
                  <div className="w-full min-h-[60px] rounded-xl bg-card border border-border p-3 flex flex-wrap gap-2 items-center">
                    {sentenceWords.length === 0 ? (
                      <span className="text-sm text-muted-foreground italic">Hold a sign for 1 second to add it...</span>
                    ) : (
                      sentenceWords.map((word, i) => (
                        <span key={i} className="px-3 py-1 rounded-full bg-primary text-primary-foreground text-sm font-bold">
                          {word}
                        </span>
                      ))
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground text-center -mt-3">
                    Drop hands between signs • Hold 1 second to capture each
                  </p>
                  <div className="flex gap-3 w-full">
                    <button type="button"
                      onClick={() => {
                        const newWords = sentenceWordsRef.current.slice(0, -1);
                        sentenceWordsRef.current = newWords;
                        setSentenceWords([...newWords]);
                      }}
                      disabled={sentenceWords.length === 0}
                      className="flex-1 min-h-[56px] rounded-xl bg-secondary text-secondary-foreground text-sm font-semibold flex items-center justify-center gap-2 active:scale-[0.98] transition-transform border border-border disabled:opacity-40">
                      ← Undo
                    </button>
                    <button type="button" onClick={stopSentenceBuilding}
                      disabled={sentenceWords.length === 0}
                      className="flex-[2] min-h-[56px] rounded-xl bg-primary text-primary-foreground text-base font-semibold flex items-center justify-center gap-2 active:scale-[0.98] transition-transform disabled:opacity-40">
                      <Square className="w-5 h-5 fill-current" />
                      Stop & Translate
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <button type="button" onClick={startSentenceBuilding}
                    disabled={!stream}
                    className="w-full min-h-[56px] rounded-xl bg-primary text-primary-foreground text-base font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform shadow-xl disabled:opacity-40">
                    <Type className="w-5 h-5" />
                    Build Sentence
                  </button>
                  <p className="text-xs text-muted-foreground">Sign words one by one to build a full sentence</p>
                </>
              )}
            </>
          ) : (
            <>
              <div onClick={() => fileInputRef.current?.click()}
                className="relative w-full aspect-square rounded-2xl overflow-hidden bg-card border-2 border-dashed border-border shadow-2xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:border-primary transition-colors">
                <Upload className="w-16 h-16 text-muted-foreground" />
                <div className="text-center px-4">
                  <p className="text-lg font-medium">{uploadedFileName ?? "Click to upload a BSL video"}</p>
                  <p className="text-sm text-muted-foreground mt-1">MP4, MOV, WebM supported</p>
                </div>
              </div>
              <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
              <button type="button" onClick={() => fileInputRef.current?.click()}
                className="w-full min-h-[60px] rounded-xl bg-primary text-primary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform shadow-xl">
                <Upload className="w-6 h-6" />
                Choose Video File
              </button>
            </>
          )}
        </section>
      ) : (
        <section className="w-full max-w-xl flex flex-col items-center gap-6 mt-4">

          {/* English */}
          <div className="w-full rounded-2xl bg-card border border-border p-6 shadow-xl">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3">
              English (BSL Translation)
            </p>
            {isTranslating ? (
              <div className="flex items-center gap-3 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin shrink-0" />
                <p>Translating signs...</p>
              </div>
            ) : (
              <p className="text-2xl sm:text-3xl font-medium leading-snug">{translation}</p>
            )}
          </div>

          {/* Darija */}
          <div className="w-full rounded-2xl bg-card border border-border p-6 shadow-xl">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3">
              Moroccan Darija (دارجة)
            </p>
            {isTranslating || isDarijaLoading ? (
              <div className="flex items-center gap-3 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin shrink-0" />
                <p>Translating to Darija...</p>
              </div>
            ) : (
              <p className="text-2xl sm:text-3xl font-medium leading-snug text-right" dir="rtl">
                {darija || "—"}
              </p>
            )}
          </div>

          {/* Buttons */}
          <div className="w-full flex flex-col gap-3">
            <button type="button" onClick={speak}
              disabled={isTranslating || isDarijaLoading || !translation}
              className="w-full min-h-[60px] rounded-xl bg-primary text-primary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform disabled:opacity-50">
              <Volume2 className="w-6 h-6" />
              Speak in Darija
            </button>
            <div className="flex gap-3">
              <button type="button" onClick={copyToClipboard} disabled={!translation}
                className="flex-1 min-h-[60px] rounded-xl bg-secondary text-secondary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform border border-border disabled:opacity-50">
                {copied ? <Check className="w-6 h-6" /> : <Copy className="w-6 h-6" />}
                {copied ? "Copied!" : "Copy"}
              </button>
              <button type="button" onClick={recordAgain}
                className="flex-1 min-h-[60px] rounded-xl bg-secondary text-secondary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform border border-border">
                <RotateCcw className="w-6 h-6" />
                Again
              </button>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
