import { createAPIFileRoute } from "@tanstack/start/api";

export const APIRoute = createAPIFileRoute("/api/translate")({
  POST: async ({ request }) => {
    const { frames } = await request.json();
    const token = process.env.HF_TOKEN;

    const body = {
      model: "meta-llama/Llama-3.2-11B-Vision-Instruct",
      messages: [
        {
          role: "user",
          content: [
            ...frames.map((frame: string) => ({
              type: "image_url",
              image_url: { url: frame },
            })),
            {
              type: "text",
              text: `These are 3 frames (start, middle, end) from a short video of a person
              using British Sign Language (BSL).
              BSL uses two hands and has different grammar from English.
              Translate what the person is signing into natural English text.
              Return only the translated English sentence, nothing else.
              If signs are unclear, return your best interpretation.`,
            },
          ],
        },
      ],
      max_tokens: 200,
    };

    const res = await fetch(
      "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      }
    );

    if (!res.ok) {
      const err = await res.text();
      return new Response(JSON.stringify({ error: err }), {
        status: res.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    const data = await res.json();
    const text =
      data.choices?.[0]?.message?.content ??
      "Could not translate. Please try again.";

    return new Response(JSON.stringify({ text }), {
      headers: { "Content-Type": "application/json" },
    });
  },
});
