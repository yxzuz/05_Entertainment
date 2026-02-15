import torch
import clip
from PIL import Image
import argparse
import json
from pathlib import Path


def load_clip_model(device="cuda"):
    """Load CLIP model for evaluation"""
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    elif torch.backends.mps.is_available() and device == "mps":
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading CLIP model on {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def calculate_clip_score(image_path, text_prompt, model, preprocess, device):
    """Calculate CLIP score between image and text"""
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text_prompt]).to(device)

    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    clip_score = (image_features @ text_features.T).item()
    return clip_score


def evaluate_dataset(data_dir, metadata_file="metadata.json"):
    """Evaluate all images in the dataset using CLIP scores"""
    model, preprocess, device = load_clip_model()

    # Load metadata
    metadata_path = Path(data_dir) / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    scores = []
    results = []

    print(f"Evaluating {len(metadata)} images...")

    for item in metadata:
        image_path = Path(data_dir) / item["file_name"]
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            score = calculate_clip_score(
                image_path, item["prompt"], model, preprocess, device
            )
            scores.append(score)
            results.append({
                "file_name": item["file_name"],
                "prompt": item["prompt"],
                "clip_score": score
            })
            print(f"{item['file_name']}: {score:.4f}")

        except Exception as e:
            print(f"Error processing {item['file_name']}: {e}")
            continue

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage CLIP Score: {avg_score:.4f}")
        print(f"Min Score: {min(scores):.4f}")
        print(f"Max Score: {max(scores):.4f}")

        return {
            "results": results,
            "average_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores)
        }
    else:
        print("No valid scores calculated")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate images using CLIP scores")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing images and metadata.json"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.json",
        help="Name of the metadata file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save evaluation results"
    )

    args = parser.parse_args()

    results = evaluate_dataset(args.data_dir, args.metadata_file)

    if results and args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
