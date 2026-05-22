import { createServer } from "http";

const GROQ_API_KEY = "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";

async function callGroq(messages, maxTokens = 300) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
      messages,
      max_tokens: maxTokens,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

function formatLandmarkSamples(samples) {
  const fingerNames = ["thumb", "index", "middle", "ring", "pinky"];
  const tipIds  = [4, 8, 12, 16, 20];
  const pipIds  = [3, 7, 11, 15, 19];
  const mcpIds  = [2, 5, 9, 13, 17];

  function getFingerState(lm, fi) {
    const tip = lm[tipIds[fi]];
    const pip = lm[pipIds[fi]];
    const mcp = lm[mcpIds[fi]];
    if (fi === 0) {
      // Thumb: use x-distance from wrist
      const dist = Math.abs(tip.x - lm[0].x);
      if (dist > 0.12) return "extended";
      if (dist > 0.06) return "half";
      return "bent";
    }
    if (tip.y < mcp.y) return "extended";
    if (pip.y < mcp.y) return "half";
    return "bent";
  }

  function getPalmFacing(lm) {
    // Cross product of (index_mcp - pinky_mcp) × (middle_mcp - wrist)
    const [i, p, m, w] = [lm[5], lm[17], lm[9], lm[0]];
    const ax = i.x - p.x, ay = i.y - p.y, az = i.z - p.z;
    const bx = m.x - w.x, by = m.y - w.y, bz = m.z - w.z;
    const nz = ax * by - ay * bx;
    return nz > 0 ? "palm-toward-camera" : "palm-away";
  }

  function tipsClose(lm, fi1, fi2) {
    const a = lm[tipIds[fi1]], b = lm[tipIds[fi2]];
    return Math.hypot(a.x - b.x, a.y - b.y) < 0.05;
  }

  const lines = samples.map((sample, i) => {
    const phase = i === 0 ? "start" : i === samples.length - 1 ? "end" : `mid${i}`;
    if (!sample.landmarks || sample.landmarks.length === 0)
      return `[${phase}]: no hands`;

    const hands = sample.landmarks.map((lm, hi) => {
      const side = sample.handedness?.[hi]?.[0]?.categoryName ?? `Hand${hi + 1}`;
      const fingers = fingerNames.map((n, fi) => `${n}:${getFingerState(lm, fi)}`);
      const facing = getPalmFacing(lm);
      const w = lm[0];
      const px = w.x < 0.35 ? "left" : w.x > 0.65 ? "right" : "center";
      const py = w.y < 0.35 ? "face-level" : w.y > 0.65 ? "waist-level" : "chest-level";
      const extras = [];
      if (tipsClose(lm, 0, 1)) extras.push("thumb-index-touch");
      if (tipsClose(lm, 0, 2)) extras.push("thumb-middle-touch");
      return `${side}[${fingers.join(" ")} | ${facing} | ${px} ${py}${extras.length ? " | " + extras.join(" ") : ""}]`;
    });

    return `[${phase}] ${hands.join("  +  ")}`;
  });

  // Add movement description for dominant hand
  const first = samples[0], last = samples[samples.length - 1];
  if (first?.landmarks?.length && last?.landmarks?.length) {
    const fw = first.landmarks[0][0], lw = last.landmarks[0][0];
    const dx = lw.x - fw.x, dy = lw.y - fw.y;
    const moves = [];
    if (Math.abs(dx) > 0.06) moves.push(dx > 0 ? "moved-right" : "moved-left");
    if (Math.abs(dy) > 0.06) moves.push(dy > 0 ? "moved-down" : "moved-up");
    if (moves.length) lines.push(`[movement] dominant hand: ${moves.join(" ")}`);
  }

  return lines.join("\n");
}

const BSL_HINT = `
Common BSL signs for reference:
- Open hand (all extended), palm out, at face, waving = Hello / Goodbye
- Index finger pointing at self (chest) = I / Me
- Index pointing forward/at other person = You
- Fist at chest moving outward = Thank you
- Both hands open, palms up = What / What do you want?
- Thumb up = Good / Yes / OK
- Fist shaking side-to-side = No
- Index + middle extended (V shape) = 2 / Peace / Win
- All fingers extended, palm flat = Stop / Wait / 5
- Index finger pointing up = Yes / 1
- Flat hand moving to mouth = Eat / Food / Hungry
- Curved hand at mouth = Drink / Thirsty
- Both fists side by side, breaking apart = Open / Break
- Hand at forehead moving out = Sorry / Excuse me
- Pinched fingers at lips moving out = Please
`.trim();

createServer(async (req, res) => {
  // Allow any localhost origin (Vite dev server port varies: 5173, 3000, 8080, etc.)
  const origin = req.headers.origin || "";
  const allowedOrigin = origin.startsWith("http://localhost") ? origin : "*";
  res.setHeader("Access-Control-Allow-Origin", allowedOrigin);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  let body = "";
  req.on("data", (chunk) => (body += chunk));

  // Recognized sign names → English sentence
  if (req.method === "POST" && req.url === "/api/translate-signs") {
    req.on("end", async () => {
      try {
        const { signs } = JSON.parse(body);
        const unique = signs.filter((s, i) => i === 0 || s !== signs[i - 1]);
        console.log("Signs to translate:", unique);

        const text = await callGroq([
          {
            role: "system",
            content: "You are a BSL sentence assembler. You receive a list of English words that were already identified from BSL signs. Combine them into a natural English phrase or sentence. Do NOT change the meaning or substitute different words. Output ONLY the resulting English text, nothing else.",
          },
          {
            role: "user",
            content: `The signer produced these words (already translated from BSL): ${unique.join(", ")}
Write them as a natural English phrase. Keep the same words and meaning. Reply with ONLY the English phrase.`,
          },
        ], 60);

        console.log("Signs translation:", text);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ text: text.trim() }));
      } catch (e) {
        console.error("Signs translate error:", e.message);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // BSL landmarks → English (camera mode)
  if (req.method === "POST" && req.url === "/api/translate-landmarks") {
    req.on("end", async () => {
      try {
        const { samples } = JSON.parse(body);
        console.log(`Landmark translation: ${samples.length} samples`);

        const description = formatLandmarkSamples(samples);
        console.log("Landmarks:\n", description);

        const text = await callGroq([
          {
            role: "system",
            content: `You are a British Sign Language (BSL) expert. When given hand landmark data, you output ONLY the English translation — a word or short phrase (1–6 words). You never write explanations, uncertainty phrases like "likely" or "possibly", or sentences describing what the signer "might be" doing. Just the translation.`,
          },
          {
            role: "user",
            content: `Hand landmark data (finger states: extended/half/bent, palm direction, position, movement):

${description}

${BSL_HINT}

What is the person signing? Reply with ONLY the English word or phrase.`,
          },
        ]);

        console.log("Translation result:", text);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ text: text.trim() }));
      } catch (e) {
        console.error("Landmark translate error:", e.message);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // BSL frames → English (upload/fallback mode)
  if (req.method === "POST" && req.url === "/api/translate") {
    req.on("end", async () => {
      try {
        const { frames } = JSON.parse(body);

        const text = await callGroq([
          {
            role: "system",
            content: `You are a BSL interpreter. Output ONLY the English translation (1–6 words). No explanations, no hedging.`,
          },
          {
            role: "user",
            content: [
              ...frames.map((frame) => ({
                type: "image_url",
                image_url: { url: frame },
              })),
              {
                type: "text",
                text: `These are 3 frames from a BSL signing video (start, middle, end).
Look at the hand shapes and positions. What BSL sign is being made?
Reply with ONLY the English word or short phrase.`,
              },
            ],
          },
        ]);

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ text: text.trim() }));
      } catch (e) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // English → Moroccan Darija
  if (req.method === "POST" && req.url === "/api/darija") {
    req.on("end", async () => {
      try {
        const { text } = JSON.parse(body);
        console.log("Darija request for:", text);

        const darija = await callGroq([
          {
            role: "user",
            content: `You are a Moroccan Darija translator.
Translate this English text into Moroccan Darija written in Arabic script.
Be natural and conversational — write it how Moroccans actually speak.
Return ONLY the Arabic script translation, nothing else.

English: ${text}`,
          },
        ]);

        console.log("Darija result:", darija);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ darija: darija || text }));
      } catch (e) {
        console.error("Darija error:", e.message);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end("Not found");
}).listen(3001, () => {
  console.log("Translation proxy running on http://localhost:3001");
});
