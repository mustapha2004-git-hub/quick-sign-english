import { createFileRoute } from "@tanstack/react-router";
import { SignBridgeApp } from "@/components/SignBridgeApp";

export const Route = createFileRoute("/")({
  component: SignBridgeApp,
});
