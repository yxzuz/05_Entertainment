"use client";

import { useState, useEffect } from "react";

type JobStatus = "idle" | "pending" | "generating" | "completed" | "failed";

interface GeneratedImage {
  url: string;
  prompt: string;
  timestamp: Date;
}

export default function GeneratePage() {
  const [prompt, setPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<JobStatus>("idle");
  const [history, setHistory] = useState<GeneratedImage[]>([]);
  const [numSteps, setNumSteps] = useState(50);
  const [backendStatus, setBackendStatus] = useState<string | null>(null);

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus();
  }, []);

  async function checkBackendStatus() {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      setBackendStatus(data.model_loaded ? 'Model loaded' : 'Model not loaded');
    } catch {
      setBackendStatus('Backend unavailable');
    }
  }

  async function handleGenerate() {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setImageUrl(null);
    setError(null);
    setProgress(0);
    setStatus("pending");
    setLoading(true);

    const currentPrompt = prompt.trim();

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          prompt: currentPrompt,
          num_inference_steps: numSteps
        })
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({ error: "Failed to start generation." }));
        throw new Error(payload.error || "Failed to start generation.");
      }

      const { job_id: jobId } = (await response.json()) as { job_id?: string };
      if (!jobId) {
        throw new Error("No job_id returned from server.");
      }

      const eventSource = new EventSource(`/api/generate-status/${jobId}`);
      let startTime = Date.now();

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data) as {
          status: JobStatus;
          progress?: number;
          image_url?: string;
          error?: string | null;
        };

        if (typeof data.progress === "number") {
          setProgress(data.progress);
        }

        if (data.status === "completed") {
          setStatus("completed");
          setImageUrl(data.image_url || null);
          setLoading(false);
          
          // Add to history
          if (data.image_url) {
            const newImage: GeneratedImage = {
              url: data.image_url,
              prompt: currentPrompt,
              timestamp: new Date()
            };
            setHistory(prev => [newImage, ...prev.slice(0, 9)]); // Keep last 10
          }
          
          eventSource.close();
          return;
        }

        if (data.status === "failed") {
          setStatus("failed");
          setError(data.error || "Generation failed.");
          setLoading(false);
          eventSource.close();
          return;
        }

        setStatus("generating");
      };

      eventSource.onerror = () => {
        setStatus("failed");
        setError("Connection lost while monitoring generation.");
        setLoading(false);
        eventSource.close();
      };
    } catch (err) {
      setStatus("failed");
      setError(err instanceof Error ? err.message : "Unexpected error.");
      setLoading(false);
    }
  }

  function downloadImage(url: string, filename: string) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  const suggestedPrompts = [
    "a portrait of sks person, cyberpunk style",
    "a majestic lion in golden sunlight",
    "a futuristic cityscape at sunset",
    "a cozy cabin in the winter forest",
    "an astronaut floating in space"
  ];

  return (
    <div className="container py-5">
      <div className="row">
        <div className="col-lg-8">
          <div className="card shadow-sm">
            <div className="card-header bg-primary text-white">
              <h1 className="h4 mb-0">🎨 AI Image Generator</h1>
              {backendStatus && (
                <small className={`badge ms-2 ${
                  backendStatus.includes('loaded') ? 'bg-success' : 'bg-warning'
                }`}>
                  {backendStatus}
                </small>
              )}
            </div>
            <div className="card-body">
              <div className="mb-4">
                <label htmlFor="prompt" className="form-label fw-bold">
                  Describe your image
                </label>
                <textarea
                  id="prompt"
                  className="form-control"
                  rows={3}
                  placeholder="e.g. a portrait of sks person, cyberpunk style"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  disabled={loading}
                />
                <div className="form-text">
                  Be descriptive! Include style, mood, colors, and details.
                </div>
              </div>

              <div className="row mb-4">
                <div className="col-md-6">
                  <label htmlFor="steps" className="form-label fw-bold">
                    Quality (Steps: {numSteps})
                  </label>
                  <input
                    id="steps"
                    type="range"
                    className="form-range"
                    min="20"
                    max="100"
                    step="10"
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                    disabled={loading}
                  />
                  <div className="form-text">
                    Higher steps = better quality but slower generation
                  </div>
                </div>
              </div>

              <div className="mb-4">
                <div className="d-flex flex-wrap gap-2">
                  <button 
                    className="btn btn-primary btn-lg" 
                    onClick={handleGenerate} 
                    disabled={loading || !prompt.trim()}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" />
                        Generating...
                      </>
                    ) : (
                      '🎨 Generate Image'
                    )}
                  </button>
                  <button 
                    className="btn btn-outline-secondary" 
                    onClick={checkBackendStatus}
                    disabled={loading}
                  >
                    🔄 Refresh Status
                  </button>
                </div>
              </div>

              {loading && (
                <div className="mb-4">
                  <div className="progress mb-2" style={{height: '20px'}}>
                    <div 
                      className="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                      style={{ width: `${progress}%` }}
                    >
                      {progress}%
                    </div>
                  </div>
                  <div className="text-center">
                    <small className="text-muted">
                      Status: <span className="fw-bold text-info">{status}</span> • Progress: {progress}%
                    </small>
                  </div>
                </div>
              )}

              {error && (
                <div className="alert alert-danger" role="alert">
                  <strong>❌ Error:</strong> {error}
                </div>
              )}

              {imageUrl && (
                <div className="mt-4">
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <h5 className="mb-0">✨ Generated Image</h5>
                    <button 
                      className="btn btn-outline-success btn-sm"
                      onClick={() => downloadImage(imageUrl, `generated-${Date.now()}.png`)}
                    >
                      📥 Download
                    </button>
                  </div>
                  <div className="text-center">
                    <img 
                      src={imageUrl} 
                      alt="Generated result" 
                      className="img-fluid rounded shadow-lg" 
                      style={{maxHeight: '500px'}}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="col-lg-4">
          <div className="card shadow-sm mb-4">
            <div className="card-header">
              <h6 className="mb-0">💡 Quick Ideas</h6>
            </div>
            <div className="card-body">
              <div className="d-grid gap-2">
                {suggestedPrompts.map((suggestion, index) => (
                  <button
                    key={index}
                    className="btn btn-outline-secondary btn-sm text-start"
                    onClick={() => setPrompt(suggestion)}
                    disabled={loading}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {history.length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header">
                <h6 className="mb-0">📸 Recent Images ({history.length})</h6>
              </div>
              <div className="card-body">
                <div className="row g-2">
                  {history.map((item, index) => (
                    <div key={index} className="col-6">
                      <div className="card bg-light">
                        <img 
                          src={item.url} 
                          alt="Recent generation" 
                          className="card-img-top"
                          style={{height: '80px', objectFit: 'cover', cursor: 'pointer'}}
                          onClick={() => {
                            setImageUrl(item.url);
                            setPrompt(item.prompt);
                          }}
                        />
                        <div className="card-body p-2">
                          <small className="text-muted">
                            {item.prompt.substring(0, 30)}...
                          </small>
                          <br />
                          <small className="text-muted" style={{fontSize: '0.75em'}}>
                            {item.timestamp.toLocaleTimeString()}
                          </small>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
