import Link from "next/link";

export default function HomePage() {
  return (
    <div className="container py-5">
      <div className="row justify-content-center">
        <div className="col-lg-8 text-center">
          <div className="mb-5">
            <h1 className="display-4 fw-bold text-primary mb-4">
              🎨 AI Image Generator
            </h1>
            <p className="lead text-secondary mb-4">
              Transform your ideas into stunning images using our fine-tuned Stable Diffusion model.
              Generate portraits, landscapes, artwork and more with just a text description.
            </p>
          </div>

          <div className="row mb-5">
            <div className="col-md-4 mb-4">
              <div className="card h-100 border-0 shadow-sm">
                <div className="card-body text-center">
                  <div className="mb-3">
                    <span className="display-4">⚡</span>
                  </div>
                  <h5 className="card-title">Fast Generation</h5>
                  <p className="card-text text-muted">
                    Generate high-quality images in seconds with real-time progress tracking.
                  </p>
                </div>
              </div>
            </div>
            <div className="col-md-4 mb-4">
              <div className="card h-100 border-0 shadow-sm">
                <div className="card-body text-center">
                  <div className="mb-3">
                    <span className="display-4">🎯</span>
                  </div>
                  <h5 className="card-title">Custom Trained</h5>
                  <p className="card-text text-muted">
                    Our LoRA model is specially fine-tuned for portrait generation and artistic styles.
                  </p>
                </div>
              </div>
            </div>
            <div className="col-md-4 mb-4">
              <div className="card h-100 border-0 shadow-sm">
                <div className="card-body text-center">
                  <div className="mb-3">
                    <span className="display-4">🖼️</span>
                  </div>
                  <h5 className="card-title">High Quality</h5>
                  <p className="card-text text-muted">
                    Adjustable quality settings and support for detailed, creative prompts.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-4">
            <Link className="btn btn-primary btn-lg px-5 py-3" href="/generate">
              🚀 Start Generating Images
            </Link>
          </div>

          <div className="mt-5">
            <h4 className="mb-3">Example Prompts to Try:</h4>
            <div className="row text-start">
              <div className="col-md-6">
                <ul className="list-unstyled">
                  <li className="mb-2">✨ "a portrait of sks person, cyberpunk style"</li>
                  <li className="mb-2">🌅 "a majestic mountain landscape at sunrise"</li>
                  <li className="mb-2">🦁 "a photorealistic lion in golden light"</li>
                </ul>
              </div>
              <div className="col-md-6">
                <ul className="list-unstyled">
                  <li className="mb-2">🏠 "a cozy cabin in a winter forest"</li>
                  <li className="mb-2">🌊 "ocean waves crashing on rocky cliffs"</li>
                  <li className="mb-2">🎨 "abstract art with vibrant colors"</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
