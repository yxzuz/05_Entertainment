import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# for LoRA
from peft import LoraConfig, get_peft_model

# for dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Force CPU usage to avoid MPS memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.backends.mps.is_available = lambda: False
torch.set_num_threads(1)

logger = get_logger(__name__)


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root,
                 tokenizer,
                 size=512,
                 center_crop=False,
                 encoder_hidden_states=None,
                 metadata_file="metadata.json"):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance data root {self.instance_data_root} does not exist.")

        self.instance_images_path = []
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
        for f in Path(instance_data_root).iterdir():
            if f.is_file() and f.suffix.lower() in image_extensions:
                self.instance_images_path.append(f)

        self.num_instance_images = len(self.instance_images_path)
        if self.num_instance_images == 0:
            raise ValueError(
                f"No images found in {self.instance_data_root}. Supported extensions are: {image_extensions}")

        self.metadata = {}

        # Load metadata file if available
        metadata_path = self.instance_data_root / metadata_file
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                loaded_metadata = json.load(f)
            # Convert list to dict for easier access
            self.metadata = {item["file_name"]: item["prompt"]
                             for item in loaded_metadata}
        else:
            # If no metadata file found, use default prompts
            for img_path in self.instance_images_path:
                self.metadata[img_path.name] = img_path.stem.replace(
                    "_", " ")  # simple default prompt based on file name

        # Preprocessing transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        instance_image_path = self.instance_images_path[index %
                                                        self.num_instance_images]
        instance_prompt = self.metadata.get(
            instance_image_path.name, "")  # Get prompt from metadata
        if not instance_prompt:
            instance_prompt = instance_image_path.stem.replace(
                "_", " ")  # Default prompt if none found

        image = Image.open(instance_image_path).convert("RGB")
        example = {}
        example["instance_images"] = self.image_transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        ).input_ids

        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # concat input_ids and pixel_values
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    return batch


def train(args):
    # Determine device for non-accelerator components
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device} for non-accelerator components")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Make one log on every process with the total number of steps
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info(f"***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # Load scheduler, tokenizer, and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Move models to device
    vae.to(device)
    text_encoder.to(device)

    # Convert Unet to LoRA model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the models to appropriate precision
    if args.mixed_precision == "fp16":
        vae = vae.to(dtype=torch.float16)
        text_encoder = text_encoder.to(dtype=torch.float16)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # We need to initialize the trackers we use for logging
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    logger.info(f"  Num examples = {len(train_dataset)}")

    global_step = 0
    first_epoch = 0

    # Potentially load EMA model
    if args.with_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(
                    device=vae.device, dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning
                input_ids = batch["input_ids"].to(device=text_encoder.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(),
                                  target.float(), reduction="mean")

                # Backpropagate the loss
                accelerator.backward(loss)

                # Optimize the model parameters
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Update EMA model
                if args.with_ema:
                    ema_unet.step(unet.parameters())

            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

    # Save the LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        # save only LoRA layers
        unet.save_pretrained(args.output_dir)
        logger.info(f"LoRA weights saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning for Stable Diffusion")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",  # or path to teacher's model
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models."
    )

    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="../data",
        help="A folder containing the training data images and metadata.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images",
    )

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the images before resizing to resolution",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )

    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )

    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="The logging directory for the experiment tracking.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to.",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        help="Whether to use mixed precision.",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )

    parser.add_argument(
        "--with_ema",
        action="store_true",
        help="Whether to use EMA model.",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The rank of the LoRA layers.",
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.instance_data_dir is None:
        raise ValueError("You must specify a instance_data_dir")
    if args.pretrained_model_name_or_path is None:
        raise ValueError("You must specify a pretrained_model_name_or_path")

    train(args)
