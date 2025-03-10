# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import accelerate
import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from discriminator import Discriminator
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm

from diffusers import VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available




import sys
sys.path.append(os.path.join(os.getcwd(), "dataloader"))
sys.path.append(os.path.join(os.getcwd(), "modules/vae"))
import washu_dataloader as wu_dl
from create_images_vae import image_as_grid

import hydra
from omegaconf import DictConfig
import omegaconf

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_perceptual_loss(pixel_values, fmap, timm_model, timm_model_resolution, timm_model_normalization):
    raise NotImplementedError("Perceptual loss is not implemented yet.")

def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()

#enforces lipschitz constraint
def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    bsz = gradients.shape[0]
    gradients = torch.reshape(gradients, (bsz, -1))
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def log_validation(model, args, dataset, accelerator, global_step):
    logger.info("Generating images...")
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    
    sample = dataset.__getitem__(0).unsqueeze(0).to(accelerator.device, dtype=dtype)
    mask = ~torch.isnan(sample)
    original_image_bt = fill_nan_with_min(sample) #this is same as first filling nan with zero and then normalizing. But we first normalize
    generated_image_bt = accelerator.unwrap_model(model)(original_image).sample
    model.train()
    
    original_image = image_as_grid(original_image_bt[0], dataset)
    generated_image = image_as_grid(generated_image_bt[0], dataset)
    masked_image = image_as_grid(generated_image_bt[0], dataset, mask=mask[0])
    # Log images
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {"input_images": [wandb.Image(original_image)],
                           "masked_reconstructed_images": [wandb.Image(masked_image)],
                            "reconstructed_images": [wandb.Image(generated_image)] },
                step=global_step,
            )
    torch.cuda.empty_cache()
    return generated_image_bt


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

def fill_nan_with_min(batch, min_vals, nan_mask):
    min_vals = torch.FloatTensor(min_vals).to(batch.device)
    min_vals = min_vals.view(1, -1, 1, 1)  # Reshape to match (B, C, H, W) broadcasting
    filled_batch = torch.where(nan_mask, min_vals, batch)
    return filled_batch

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    args = cfg.ae
    #########################
    # SETUP Accelerator     #
    #########################

    # Enable TF32 on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.train_batch_size

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################

    if accelerator.is_main_process:
        tracker_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        #tracker_config.pop("validation_images")
        accelerator.init_trackers(cfg.wandb.project, tracker_config)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed) 

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    if args.model_config_name_or_path is None and args.pretrained_model_name_or_path is None:
        # Taken from config of movq at kandinsky-community/kandinsky-2-2-decoder but without the attention layers
        model = VQModel(
            act_fn="silu",
            block_out_channels=[
                128,
                256,
                512,
            ],
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            in_channels=len(cfg.components),
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            norm_type="spatial",
            num_vq_embeddings=16384,
            out_channels=len(cfg.components),
            sample_size=32,
            scaling_factor=0.18215,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            vq_embed_dim=4,
        )
    elif args.pretrained_model_name_or_path is not None:
        model = VQModel.from_pretrained(args.pretrained_model_name_or_path)
    else:
        config = VQModel.load_config(args.model_config_name_or_path)
        model = VQModel.from_config(config)
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=VQModel, model_config=model.config)
    if args.discriminator_config_name_or_path is None:
        discriminator = Discriminator(in_channels=len(cfg.components))
    else:
        config = Discriminator.load_config(args.discriminator_config_name_or_path)
        discriminator = Discriminator.from_config(config)

    # Enable flash attention if asked
    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "vqmodel_ema"))
                vqmodel = models[0]
                discriminator = models[1]
                vqmodel.save_pretrained(os.path.join(output_dir, "vqmodel"))
                discriminator.save_pretrained(os.path.join(output_dir, "discriminator"))
                weights.pop()
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "vqmodel_ema"), VQModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            discriminator = models.pop()
            load_model = Discriminator.from_pretrained(input_dir, subfolder="discriminator")
            discriminator.register_to_config(**load_model.config)
            discriminator.load_state_dict(load_model.state_dict())
            del load_model
            vqmodel = models.pop()
            load_model = VQModel.from_pretrained(input_dir, subfolder="vqmodel")
            vqmodel.register_to_config(**load_model.config)
            vqmodel.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = (
            learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    discr_optimizer = optimizer_cls(
        list(discriminator.parameters()),
        lr=args.discr_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    
    args.train_batch_size * accelerator.num_processes
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    '''
    # DataLoaders creation:
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    assert args.image_column is not None
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}")
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    '''

    train_dataset = wu_dl.initialize_dataset(cfg.root_dir, cfg.grid_size, cfg.components)      
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )
    discr_lr_scheduler = get_scheduler(
        args.discr_lr_scheduler,
        optimizer=discr_optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=args.lr_warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler = accelerator.prepare(
        model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            accelerator.wait_for_everyone()
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    avg_gen_loss, avg_discr_loss = None, None
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        discriminator.train()
        for i, batch in enumerate(train_dataloader):
            batch = batch.to(accelerator.device, non_blocking=True)
            mask = ~torch.isnan(batch)
            pixel_values = fill_nan_with_min(batch, train_dataset.min_vals, ~mask)

            data_time_m.update(time.time() - end)
            generator_step = ((i // args.gradient_accumulation_steps) % 2) == 0
            # Train Step
            # The behavior of accelerator.accumulate is to
            # 1. Check if gradients are synced(reached gradient-accumulation_steps)
            # 2. If so sync gradients by stopping the not syncing process
            if generator_step:
                optimizer.zero_grad(set_to_none=True)
            else:
                discr_optimizer.zero_grad(set_to_none=True)
            # encode images to the latent space and get the commit loss from vq tokenization
            # Return commit loss
            fmap, commit_loss = model(pixel_values, return_dict=False)
            # Make the values corresponding to nan in the input min. we will only use mask to calculate l2,l1 loss after here. for discr. it is not necessary
            fmap = fill_nan_with_min(fmap, train_dataset.min_vals, ~mask)

            if generator_step:
                with accelerator.accumulate(model):
                    # reconstruction loss. Pixel level differences between input vs output
                    if args.vae_loss == "l2":
                        loss = F.mse_loss(pixel_values[mask], fmap[mask])
                    else:
                        loss = F.l1_loss(pixel_values[mask], fmap[mask])
                    
                    # generator loss
                    #logger.info(f"fmap shape: {fmap.shape}")
                    #logger.info(f"fmap[mask] shape: {fmap[mask].shape}")
                    #logger.info(f"mask shape: {mask.shape}")
                    
                    gen_loss = -discriminator(fmap).mean()
                    last_dec_layer = accelerator.unwrap_model(model).decoder.conv_out.weight
                    norm_grad_wrt_rec_loss = grad_layer_wrt_loss(loss, last_dec_layer).norm(p=2)
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)

                    adaptive_weight = norm_grad_wrt_rec_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)
                    adaptive_weight = adaptive_weight.clamp(max=1e4)
                    loss += commit_loss
                    loss += adaptive_weight * gen_loss * args.gen_loss_weight
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_gen_loss = accelerator.gather(loss.repeat(args.train_batch_size)).float().mean()
                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    # log gradient norm before zeroing it
                    if (
                        accelerator.sync_gradients
                        and global_step % args.log_grad_norm_steps == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(model, accelerator, global_step)
            else:
                # Return discriminator loss
                with accelerator.accumulate(discriminator):
                    fmap.detach_()
                    pixel_values.requires_grad_()
                    real = discriminator(pixel_values)
                    fake = discriminator(fmap)
                    loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()
                    gp = gradient_penalty(pixel_values, real)
                    loss += gp
                    avg_discr_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    accelerator.backward(loss)

                    if args.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

                    discr_optimizer.step()
                    discr_lr_scheduler.step()
                    if (
                        accelerator.sync_gradients
                        and global_step % args.log_grad_norm_steps == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(discriminator, accelerator, global_step)
            batch_time_m.update(time.time() - end)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if args.use_ema:
                    ema_model.step(model.parameters())
            if accelerator.sync_gradients and not generator_step and accelerator.is_main_process:
                # wait for both generator and discriminator to settle
                # Log metrics
                if global_step % args.log_steps == 0:
                    samples_per_second_per_gpu = (
                        args.gradient_accumulation_steps * args.train_batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_discr_loss": avg_discr_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if avg_gen_loss is not None:
                        logs["step_gen_loss"] = avg_gen_loss.item()
                    accelerator.log(logs, step=global_step)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                # Save model checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Generate images
                if global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the VQGAN parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    log_validation(model, args, train_dataset, accelerator, global_step)
                    if args.use_ema:
                        # Switch back to the original VQGAN parameters.
                        ema_model.restore(model.parameters())
            end = time.time()
            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        discriminator = accelerator.unwrap_model(discriminator)
        if args.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(os.path.join(args.output_dir, "vqmodel"))
        discriminator.save_pretrained(os.path.join(args.output_dir, "discriminator"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
