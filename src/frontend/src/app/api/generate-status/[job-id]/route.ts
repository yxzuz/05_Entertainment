import { NextResponse } from "next/server";

const PY_BACKEND_URL = process.env.PY_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(_request: Request, context: { params: { "job-id": string } }) {
  const jobId = context.params["job-id"];
  if (!jobId) {
    return NextResponse.json({ error: "job_id is required." }, { status: 400 });
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let intervalId: ReturnType<typeof setInterval> | null = null;
      let closed = false;

      const closeStream = () => {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        if (!closed) {
          closed = true;
          controller.close();
        }
      };

      const emit = (payload: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      const pollStatus = async () => {
        try {
          const response = await fetch(`${PY_BACKEND_URL}/get_job_status/${jobId}`, {
            cache: "no-store"
          });

          if (!response.ok) {
            throw new Error(`Backend status polling failed: ${response.status}`);
          }

          const data = (await response.json()) as {
            status: string;
            progress?: number;
            image_url?: string | null;
            error?: string | null;
          };

          if (data.status === "completed") {
            emit({
              status: "completed",
              progress: 100,
              image_url: data.image_url || null,
              error: null
            });
            closeStream();
            return;
          }

          if (data.status === "failed") {
            emit({
              status: "failed",
              progress: data.progress || 0,
              image_url: null,
              error: data.error || "Generation failed."
            });
            closeStream();
            return;
          }

          emit({
            status: "generating",
            progress: data.progress || 0,
            image_url: null,
            error: null
          });
        } catch (error) {
          if (!closed) {
            controller.error(error);
          }
          closeStream();
        }
      };

      void pollStatus();
      intervalId = setInterval(() => {
        void pollStatus();
      }, 1000);

      return () => {
        if (intervalId) {
          clearInterval(intervalId);
        }
      };
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive"
    }
  });
}
