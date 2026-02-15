#!/usr/bin/env python3
"""
Training script with configuration file support
"""

from train import train
import argparse
import json
import sys
from pathlib import Path

# Add src to path so we can import train module
sys.path.append(str(Path(__file__).parent / "src"))


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_args_from_config(config):
    """Convert config dict to argparse Namespace"""
    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA model with config file")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.json",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (format: key=value)"
    )

    cmd_args = parser.parse_args()

    # Load configuration
    config = load_config(cmd_args.config)
    print(f"Loaded configuration from {cmd_args.config}")

    # Apply overrides if provided
    if cmd_args.override:
        for override in cmd_args.override:
            if "=" not in override:
                print(f"Invalid override format: {override}")
                continue
            key, value = override.split("=", 1)
            # Try to parse as JSON, fallback to string
            try:
                config[key] = json.loads(value)
            except json.JSONDecodeError:
                config[key] = value
            print(f"Override: {key} = {config[key]}")

    # Convert config to args object
    args = create_args_from_config(config)

    # Ensure output directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting training with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Start training
    train(args)


if __name__ == "__main__":
    main()
