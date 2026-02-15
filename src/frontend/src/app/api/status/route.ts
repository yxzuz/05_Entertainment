import { NextResponse } from "next/server";

const PY_BACKEND_URL = process.env.PY_BACKEND_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${PY_BACKEND_URL}/status`, {
      cache: "no-store"
    });

    if (!response.ok) {
      return NextResponse.json(
        { status: "error", model_loaded: false, error: "Backend unreachable" },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Status check failed:", error);
    return NextResponse.json(
      { status: "error", model_loaded: false, error: "Backend connection failed" },
      { status: 503 }
    );
  }
}