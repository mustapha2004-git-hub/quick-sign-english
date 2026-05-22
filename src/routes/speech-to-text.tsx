import { createFileRoute } from "@tanstack/react-router";
import { SpeechToTextApp } from "@/components/SpeechToTextApp";

export const Route = createFileRoute("/speech-to-text")({
  component: SpeechToTextApp,
});
