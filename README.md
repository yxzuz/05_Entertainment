# Stable Diffusion LoRA Fine-tuning Project

This project implements LoRA (Low-Rank Adaptation) fine-tuning for Stable Diffusion models using custom image datasets, with both training capabilities and a full-stack web application.

## Project Structure

```
lecture 5/
├── src/
│   ├── frontend/         # Next.js frontend app
│   │   ├── package.json
│   │   ├── next.config.mjs
│   │   └── src/
│   │       ├── app/
│   │       └── components/
│   ├── python-backend/   # Flask backend app
│   │   └── app.py
│   ├── train.py          # Main training script
│   ├── evaluate.py       # CLIP score evaluation
│   ├── model.py          # Model utilities
│   └── predict.py        # Prediction utilities
├── data/                  # Training data
│   ├── *.jpeg            # Training images
│   └── metadata.json     # Image-prompt pairs
├── models/               # Saved models
│   └── my_lora_model/    # Output LoRA weights
├── logs/                 # Training logs
├── config/              # Configuration files
│   └── training_config.json
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration for backend
├── docker-compose.yml   # Multi-service orchestration
└── run_training.py      # Main entry point
```

## Quick Start with Docker (Recommended)

### Prerequisites
- Docker and Docker Compose installed

### Run the Full Application
```bash
# Clone the repository
git clone https://github.com/yxzuz/05_Entertainment.git
cd 05_Entertainment

# Start all services
docker-compose up --build
```

**Service URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

### Manual Setup (Alternative)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place images in the `data/` directory
   - Update `data/metadata.json` with image filenames and prompts

## Usage

### Quick Start

```bash
python run_training.py
```

### Custom Configuration

```bash
python run_training.py --config ./config/training_config.json
```

### Override Configuration Parameters

```bash
python run_training.py --override num_train_epochs=20 learning_rate=5e-5
```

### Evaluate Results

```bash
python src/evaluate.py --data_dir ./data --output_file ./logs/evaluation.json
```

## Full Lecture App (Backend + Frontend)

### Docker Deployment (Recommended)

The project includes Docker configuration for easy deployment:

```bash
# Start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

**Architecture:**
- **Backend**: Python Flask API (port 5000)
- **Frontend**: Next.js development server (port 3000)
- **Networking**: Services communicate via Docker network

### Manual Deployment

#### Python backend

```bash
cd src/python-backend
../../venv/bin/python app.py
```

Backend runs on `http://127.0.0.1:8000`.

#### Next.js frontend

```bash
cd src/frontend
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:3000`.

## Configuration

Edit `config/training_config.json` to modify training parameters:

- `pretrained_model_name_or_path`: Base Stable Diffusion model
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for training
- `lora_rank`: LoRA rank (4-16 recommended)
- `resolution`: Image resolution (512 recommended)
- `train_batch_size`: Batch size per device

## Data Format

The `metadata.json` file should contain:

```json
[
  {
    "file_name": "image_01.jpeg",
    "prompt": "Description of the image content"
  },
  ...
]
```

## Hardware Requirements

- GPU with at least 8GB VRAM (recommended)
- CPU training is possible but slower
- Apple Silicon Macs supported with MPS

## Output

- LoRA weights saved to `models/my_lora_model/`
- Training logs in `logs/`
- TensorBoard logs for monitoring training progress

## Monitoring Training

```bash
tensorboard --logdir ./logs
```

## Tips for Better Results

1. Use consistent, descriptive prompts
2. Ensure image quality is high
3. Start with lower LoRA ranks (4-8) for faster training
4. Monitor loss curves in TensorBoard
5. Use mixed precision (fp16) to save memory
