import base64
import io
import logging
import threading
from urllib import response
import uuid

import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request
from peft import PeftModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

pipeline = None
job_status = {}
jobs_lock = threading.Lock()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model() -> None:
    global pipeline
    if pipeline is None:
        print("Loading Stable Diffusion model...")
        try:
            print("Attempting to load model on device:", get_device())
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5")
            pipeline.enable_attention_slicing()
            pipeline.enable_sequential_cpu_offload()
            print("Stable Diffusion model loaded successfully.")
        except Exception as exc:
            print("CRITICAL ERROR: Failed to load Stable Diffusion model:", exc)
            import traceback
            traceback.print_exc()
            # Even if loading fails, we set pipeline to None to avoid repeated load attempts.
            pipeline = None


def progress_callback(step: int, timestep: int, latents: torch.FloatTensor, job_id: str, total_steps: int) -> None:
    progress_percent = min(int((step / total_steps) * 100), 100)
    if job_id in job_status:
        job_status[job_id]['progress'] = progress_percent


def generate_image_task(job_id: str, prompt: str, num_inference_steps: int = 50) -> None:
    global job_status
    job_status[job_id] = {'status': 'generating', 'progress': 0,
                          'image_url': None, 'error': None}
    if pipeline is None:
        job_status[job_id]['status'] = 'failed'
        job_status[job_id]['error'] = 'Model not loaded'
        return

    print("starting image generation for job:", job_id)
    try:
        # Fix callback parameter name (callback_steps not callback_step)
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            callback=lambda step, timestep, latents: progress_callback(
                step, timestep, latents, job_id, num_inference_steps),
            callback_steps=1
        ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_base64 = f"data:image/png;base64,{image_str}"

        job_status[job_id]['status'] = 'completed'
        job_status[job_id]['progress'] = 100
        job_status[job_id]['image_url'] = image_base64

    except Exception as e:
        logging.error(f"Error generating image for job {job_id}: {e}")
        if job_id in job_status:
            job_status[job_id]['status'] = 'failed'
            job_status[job_id]['error'] = str(e)
        else:
            job_status[job_id] = {
                'status': 'failed',
                'progress': 0,
                'image_url': None,
                'error': str(e)
            }


@app.route("/generate_image", methods=["POST"])
def generate_image():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    num_steps = int(data.get("num_inference_steps", 50))
    num_steps = max(1, min(num_steps, 100))

    job_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=generate_image_task, args=(job_id, prompt, num_steps), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/get_job_status/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return jsonify({"error": "Job ID not found"}), 404
    return jsonify(status), 200


@app.route("/status", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None}), 200


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000, debug=True)
