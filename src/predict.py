import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# For LoRa
from peft import PeftModel

# Main Inference Function


def predict(args):
    # Force CPU to avoid memory issues
    device = "cpu"
    print(f"Using device: {device}")

    # Load the base Stable Diffusion pipeline
    print(f"Loading base model: {args.pretrained_model_name_or_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,  # Use float32 for CPU
        safety_checker=None,  # Disable safety checker to prevent false positives
        requires_safety_checker=False
    )

    # Move to device
    pipeline.to(device)
    print(f'Model moved to device.{device}')

    # Enable memory optimizations
    pipeline.enable_attention_slicing()
    if hasattr(pipeline, 'enable_sequential_cpu_offload'):
        pipeline.enable_sequential_cpu_offload()

    # Load LoRa weights if provided
    if args.lora_model_path and os.path.exists(args.lora_model_path):
        print(f"Loading LoRa weights from: {args.lora_model_path}...")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet, args.lora_model_path)
        print("LoRa weights loaded and applied to the UNet.")
    elif args.lora_model_path and not os.path.exists(args.lora_model_path):
        print(
            f"Warning: LoRa model path {args.lora_model_path} does not exist. Skipping LoRa loading.")

    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt: ")
        if not prompt:
            print("No prompt provided. Exiting.")
            return
    print(f"Generating image for prompt: '{prompt}'")

    # Generate image
    with torch.no_grad():  # Inference should not compute gradients
        image = pipeline(prompt).images[0]

    # Save the generated image
    output_filename = args.output_path or "generated_image.png"
    image.save(output_filename)
    print(f"Image saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Inference Script.")
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to the pretrained Stable Diffusion model or model identifier from Hugging Face.")
    parser.add_argument("--prompt",
                        type=str,
                        default=None,  # Make prompt optional
                        help="Text prompt for image generation.If not provided, will prompt user input.")
    parser.add_argument("--lora_model_path",
                        type=str,
                        default="./models/my_lora_model",
                        help="Path to the LoRa model weights.")
    parser.add_argument("--output_path",
                        type=str,
                        default="generated_image.png",
                        help="Path to save the generated image. Defaults to 'generated_image.png'.")

    args = parser.parse_args()
    predict(args)
