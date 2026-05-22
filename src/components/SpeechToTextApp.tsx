import { useState, useRef, useCallback } from "react";
import { Mic, Square, Loader2, Ear, ArrowLeft } from "lucide-react";
import { Link } from "@tanstack/react-router";

export function SpeechToTextApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        await transcribeAudio(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setError("Could not access microphone. Please ensure permissions are granted.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const transcribeAudio = async (audioBlob: Blob) => {
    setIsTranscribing(true);
    try {
      // Groq API requires a file, we can fake a filename
      const file = new File([audioBlob], "recording.webm", { type: "audio/webm" });

      const formData = new FormData();
      formData.append("file", file);
      // "whisper-large-v3" or "whisper-large-v3-turbo" are supported by Groq
      formData.append("model", "whisper-large-v3-turbo");
      formData.append("response_format", "json");
      formData.append("language", "en"); // Optional, but good for speed if we expect English

      const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${import.meta.env.VITE_GROQ_API_KEY}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.error?.message || "Failed to transcribe audio");
      }

      const data = await response.json();
      setTranscript((prev) => (prev ? `${prev}\n\n${data.text}` : data.text));
    } catch (err: any) {
      console.error("Transcription error:", err);
      setError(err.message || "An error occurred during transcription.");
    } finally {
      setIsTranscribing(false);
    }
  };

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col items-center px-4 py-6 sm:py-10">
      <div className="w-full max-w-2xl flex justify-start mb-4">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Sign Translator
        </Link>
      </div>

      <header className="text-center mb-8 sm:mb-12">
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-primary flex items-center justify-center gap-3">
          <Ear className="w-8 h-8 sm:w-12 sm:h-12" />
          Hearing Mode
        </h1>
        <p className="mt-3 text-sm sm:text-base text-muted-foreground max-w-md mx-auto">
          Speak into the microphone. Your speech will be converted to text so a deaf person can read it easily.
        </p>
      </header>

      <section className="w-full max-w-2xl flex flex-col items-center gap-8">
        <div className="w-full rounded-2xl bg-card border border-border p-6 shadow-xl min-h-[250px] flex flex-col">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-4">
            Live Transcription
          </p>
          
          <div className="flex-1 overflow-y-auto">
            {transcript ? (
              <p className="text-2xl sm:text-4xl font-medium leading-relaxed text-foreground whitespace-pre-wrap">
                {transcript}
              </p>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground italic">
                {isRecording ? "Listening..." : "Waiting for speech..."}
              </div>
            )}
            
            {isTranscribing && (
              <div className="flex items-center gap-3 text-muted-foreground mt-4">
                <Loader2 className="w-5 h-5 animate-spin shrink-0" />
                <p>Converting speech to text...</p>
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-4 rounded-lg bg-destructive/10 text-destructive text-sm font-medium">
                {error}
              </div>
            )}
          </div>
        </div>

        <button
          type="button"
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isTranscribing}
          className={`relative h-28 w-28 sm:h-32 sm:w-32 rounded-full flex items-center justify-center transition-transform active:scale-95 shadow-xl ${
            isRecording 
              ? "bg-destructive animate-pulse" 
              : "bg-primary hover:brightness-110"
          } ${isTranscribing ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {isRecording ? (
            <Square className="w-10 h-10 text-white fill-white" />
          ) : (
            <Mic className="w-12 h-12 text-white" strokeWidth={2.5} />
          )}
        </button>
        <p className="text-sm font-medium text-muted-foreground -mt-4">
          {isRecording ? "Tap to stop & transcribe" : "Tap to speak"}
        </p>
        
        {transcript && (
           <button
             type="button"
             onClick={() => setTranscript("")}
             className="text-sm font-semibold text-muted-foreground hover:text-foreground transition-colors underline"
           >
             Clear Transcript
           </button>
        )}
      </section>
    </main>
  );
}
