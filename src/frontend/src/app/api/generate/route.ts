import { NextResponse } from "next/server";

const PY_BACKEND_URL = process.env.PY_BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: Request) {
  let prompt = "";
  let numSteps = 50;

  try {
    const body = await request.json();
    prompt = (body?.prompt || "").trim();
    numSteps = Math.max(1, Math.min(body?.num_inference_steps || 50, 100));
  } catch (error) {
    console.error("Invalid JSON body:", error);
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (!prompt) {
    return NextResponse.json({ error: "Prompt is required." }, { status: 400 });
  }

  try {
    const backendResponse = await fetch(`${PY_BACKEND_URL}/generate_image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        prompt,
        num_inference_steps: numSteps
      })
    });

    if (!backendResponse.ok) {
      let backendError = backendResponse.statusText;
      try {
        const payload = await backendResponse.json();
        backendError = payload?.error || backendError;
      } catch {
        // Fall back to statusText when body is not JSON.
      }

      console.error("Backend error:", backendError);
      return NextResponse.json({ error: backendError }, { status: backendResponse.status });
    }

    const result = (await backendResponse.json()) as { job_id?: string };
    if (!result.job_id) {
      return NextResponse.json({ error: "Backend did not return job_id." }, { status: 500 });
    }

    return NextResponse.json({ job_id: result.job_id }, { status: 202 });
  } catch (error) {
    console.error("Proxy failure:", error);
    return NextResponse.json({ error: "Failed to reach Python backend." }, { status: 500 });
  }
}
