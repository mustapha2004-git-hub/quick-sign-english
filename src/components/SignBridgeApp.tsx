import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Volume2, RotateCcw, Loader2 } from "lucide-react";

type Screen = "record" | "results";

async function translateSignLanguage(_frames: string[]): Promise<string> {
  // Placeholder — replace with real model call later.
  await new Promise((r) => setTimeout(r, 1200));
  return "Hello, I would like to buy this please";
}

export function SignBridgeApp() {
  const [screen, setScreen] = useState<Screen>("record");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translation, setTranslation] = useState<string>("");
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const framesRef = useRef<string[]>([]);

  // Start camera when on record screen
  useEffect(() => {
    if (screen !== "record") return;
    let active = true;
    let localStream: MediaStream | null = null;

    (async () => {
      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 640 } },
          audio: false,
        });
        if (!active) {
          localStream.getTracks().forEach((t) => t.stop());
          return;
        }
        setStream(localStream);
        if (videoRef.current) {
          videoRef.current.srcObject = localStream;
        }
      } catch (err) {
        console.error("Camera error", err);
        setCameraError(
          "Unable to access the camera. Please grant permission and reload.",
        );
      }
    })();

    return () => {
      active = false;
      if (localStream) localStream.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [screen]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    framesRef.current.push(canvas.toDataURL("image/jpeg", 0.7));
  }, []);

  const startRecording = useCallback(() => {
    if (isRecording || !stream) return;
    framesRef.current = [];
    setIsRecording(true);
    setCountdown(5);

    const frameInterval = window.setInterval(captureFrame, 200); // 5 fps
    let remaining = 5;
    const tick = window.setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        window.clearInterval(tick);
        window.clearInterval(frameInterval);
        setCountdown(null);
        setIsRecording(false);
        finishRecording();
      } else {
        setCountdown(remaining);
      }
    }, 1000);
  }, [isRecording, stream, captureFrame]);

  const finishRecording = useCallback(async () => {
    setScreen("results");
    setIsTranslating(true);
    setTranslation("");
    try {
      const text = await translateSignLanguage(framesRef.current);
      setTranslation(text);
    } catch (e) {
      console.error(e);
      setTranslation("Sorry, translation failed. Please try again.");
    } finally {
      setIsTranslating(false);
    }
  }, []);

  const speak = useCallback(() => {
    if (!translation) return;
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(translation);
    utt.rate = 1;
    utt.pitch = 1;
    utt.lang = "en-GB";
    window.speechSynthesis.speak(utt);
  }, [translation]);

  const recordAgain = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setTranslation("");
    setScreen("record");
  }, []);

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col items-center px-4 py-6 sm:py-10">
      <header className="text-center mb-6 sm:mb-10">
        <h1 className="text-4xl sm:text-6xl font-bold tracking-tight">SignBridge</h1>
        <p className="mt-2 text-base sm:text-lg text-muted-foreground">
          British Sign Language Translator
        </p>
      </header>

      {screen === "record" ? (
        <section className="w-full max-w-xl flex flex-col items-center gap-8">
          <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-card border border-border shadow-2xl">
            {cameraError ? (
              <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-muted-foreground">
                {cameraError}
              </div>
            ) : (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover scale-x-[-1]"
              />
            )}

            {isRecording && (
              <>
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-destructive/90 text-destructive-foreground px-3 py-1.5 rounded-full text-sm font-medium">
                  <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
                  REC
                </div>
                {countdown !== null && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-white text-9xl font-bold drop-shadow-[0_4px_24px_rgba(0,0,0,0.8)] tabular-nums">
                      {countdown}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>

          <button
            type="button"
            onClick={startRecording}
            disabled={isRecording || !stream}
            aria-label={isRecording ? "Recording" : "Start recording"}
            className={`relative h-24 w-24 sm:h-28 sm:w-28 rounded-full flex items-center justify-center transition-transform active:scale-95 disabled:opacity-70 ${
              isRecording
                ? "bg-destructive shadow-[0_0_0_0_rgba(239,68,68,0.7)] animate-pulse"
                : "bg-destructive hover:brightness-110 shadow-xl"
            }`}
          >
            {isRecording ? (
              <span className="block w-8 h-8 rounded-md bg-white" />
            ) : (
              <Mic className="w-10 h-10 text-white" strokeWidth={2.5} />
            )}
          </button>
          <p className="text-sm text-muted-foreground text-center -mt-4">
            {isRecording ? "Recording... hold steady" : "Tap to record 5 seconds"}
          </p>

          <canvas ref={canvasRef} className="hidden" />
        </section>
      ) : (
        <section className="w-full max-w-xl flex flex-col items-center gap-8 mt-4">
          <div className="w-full min-h-[200px] rounded-2xl bg-card border border-border p-6 sm:p-8 flex items-center justify-center text-center shadow-xl">
            {isTranslating ? (
              <div className="flex flex-col items-center gap-4 text-muted-foreground">
                <Loader2 className="w-10 h-10 animate-spin" />
                <p className="text-base">Translating your signs...</p>
              </div>
            ) : (
              <p className="text-2xl sm:text-3xl font-medium leading-snug">
                {translation}
              </p>
            )}
          </div>

          <div className="w-full flex flex-col gap-3">
            <button
              type="button"
              onClick={speak}
              disabled={isTranslating || !translation}
              className="w-full min-h-[60px] rounded-xl bg-primary text-primary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform disabled:opacity-50"
            >
              <Volume2 className="w-6 h-6" />
              Speak
            </button>
            <button
              type="button"
              onClick={recordAgain}
              className="w-full min-h-[60px] rounded-xl bg-secondary text-secondary-foreground text-lg font-semibold flex items-center justify-center gap-3 active:scale-[0.98] transition-transform border border-border"
            >
              <RotateCcw className="w-6 h-6" />
              Record Again
            </button>
          </div>
        </section>
      )}
    </main>
  );
}
