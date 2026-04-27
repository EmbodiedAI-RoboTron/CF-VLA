"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import functools
import logging
import os
import platform

import etils.epath as epath
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import jax
import numpy as np
import tqdm_loggable.auto as tqdm
import swanlab as wandb
import deepspeed

import openpi
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils
import openpi.training.checkpoints as _checkpoints


def _create_optimizer(model, config):
    """Create AdamW optimizer for trainable parameters.
    
    Args:
        model: PyTorch model (can be wrapped by DDP)
        config: TrainConfig with optimizer and lr_schedule settings
    
    Returns:
        optimizer: torch.optim.AdamW optimizer
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    
    return optimizer


def _freeze_parameters(model, freeze_filter, is_main=True):
    """Freeze parameters in the model based on a filter function or pattern.
    
    Args:
        model: PyTorch model
        freeze_filter: Can be:
            - callable: function that takes (name, param) and returns True to freeze
            - str or list[str]: parameter name pattern(s) to freeze (supports wildcards)
            - dict: mapping of module names to freeze
        is_main: whether this is the main process (for logging)
    
    Examples:
        # Freeze by name pattern
        freeze_filter = ["vision_encoder.*", "*.bias"]
        
        # Freeze by function
        def freeze_filter(name, param):
            return "vision" in name or "embed" in name
        
        # Freeze specific layers
        freeze_filter = {
            "vlm.vision_tower": True,
            "vlm.language_model.layers.0": True,
        }
    """
    import re
    
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # Check if parameter should be frozen
        if callable(freeze_filter):
            # Function-based filter
            should_freeze = freeze_filter(name, param)
        elif isinstance(freeze_filter, (list, tuple)):
            # Pattern-based filter (list of patterns)
            for pattern in freeze_filter:
                if re.match(pattern, name):
                    should_freeze = True
                    break
        elif isinstance(freeze_filter, str):
            # Single pattern
            if re.match(freeze_filter, name):
                should_freeze = True
        elif isinstance(freeze_filter, dict):
            # Dict-based filter
            for prefix, freeze in freeze_filter.items():
                if name.startswith(prefix) and freeze:
                    should_freeze = True
                    break
        
        # Apply freezing
        if should_freeze:
            param.requires_grad = False
            frozen_params.append((name, param.numel()))
        else:
            param.requires_grad = True
            trainable_params.append((name, param.numel()))
    
    # Logging
    if is_main:
        total_frozen = sum(count for _, count in frozen_params)
        total_trainable = sum(count for _, count in trainable_params)
        total = total_frozen + total_trainable
        
        logging.info(f"=" * 70)
        logging.info(f"Parameter Freezing Summary")
        logging.info(f"=" * 70)
        logging.info(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
        logging.info(f"Frozen parameters: {total_frozen:,} ({total_frozen/1e6:.2f}M) - {total_frozen/total*100:.1f}%")
        logging.info(f"Trainable parameters: {total_trainable:,} ({total_trainable/1e6:.2f}M) - {total_trainable/total*100:.1f}%")
        
        if frozen_params and len(frozen_params) <= 20:
            logging.info(f"\nFrozen layers:")
            for name, count in frozen_params:
                logging.info(f"  ❄️  {name}: {count:,}")
        elif frozen_params:
            logging.info(f"\nFrozen layers (showing first 10 of {len(frozen_params)}):")
            for name, count in frozen_params[:10]:
                logging.info(f"  ❄️  {name}: {count:,}")
        
        if trainable_params and len(trainable_params) <= 20:
            logging.info(f"\nTrainable layers:")
            for name, count in trainable_params:
                logging.info(f"  🔥 {name}: {count:,}")
        elif trainable_params:
            logging.info(f"\nTrainable layers (showing first 10 of {len(trainable_params)}):")
            for name, count in trainable_params[:10]:
                logging.info(f"  🔥 {name}: {count:,}")
        
        logging.info(f"=" * 70)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """
    Initialize wandb/swanlab for experiment tracking.
    Note: SwanLab does not support resume parameter like WandB,
    so we create a new run even when resuming training.
    """
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    
    # Get experiment name from environment or config
    experiment_name = os.getenv("SWANLAB_PROJECT", config.exp_name)
    if resuming:
        experiment_name = f"{experiment_name}_resumed"
        logging.info(f"Resuming training with new SwanLab experiment: {experiment_name}")
    
    # Initialize run (SwanLab compatible API)
    run = wandb.init(
        experiment_name=experiment_name,  # SwanLab uses 'experiment_name'
        project=config.project_name,
        config=dataclasses.asdict(config),
    )
    
    # Log source code if requested (may not be supported by SwanLab)
    if log_code and hasattr(wandb, 'run') and hasattr(wandb.run, 'log_code'):
        try:
            wandb.run.log_code(epath.Path(__file__).parent.parent)
        except Exception as e:
            logging.warning(f"Failed to log code: {e}")


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    
    # Get local_rank before initializing process group to set device correctly
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Set device before initializing process group to avoid warnings
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
        }
        # Note: device_id parameter is deprecated in newer PyTorch versions
        # The device is already set via torch.cuda.set_device() above
        # Initialize process group (device should be set before this call via torch.cuda.set_device)
        torch.distributed.init_process_group(**init_kwargs)

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        # Use barrier without device_ids parameter (deprecated in newer PyTorch versions)
        # The device should already be set correctly from setup_ddp()
        try:
            torch.distributed.barrier()
        except Exception as e:
            # If barrier fails, log but continue with cleanup
            logging.warning(f"Barrier failed during cleanup: {e}")
        finally:
            torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_lr_schedule(peak_lr: float, warmup_steps: int, decay_steps: int, end_lr: float):
    """Create a learning-rate schedule function (mirrors scripts/train.py style)."""
    def schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    return schedule


def init_train_state(
    config: _config.TrainConfig,
    device: torch.device,
    *,
    use_ddp: bool,
    resuming: bool,
    is_main: bool,
    world_size: int,
    effective_batch_size: int,
    checkpoint_manager: _checkpoints.TorchCheckpointManager | None = None,
    use_deepspeed: bool = False,
    deepspeed_config: dict = None,
):
    """Initialize model, optimizer, lr-schedule, and (optionally) DDP/DeepSpeed wrapper.

    Returns:
      model, optimizer, lr_schedule_fn, global_step, enable_gradient_checkpointing, model_engine (None if not using DeepSpeed)
    """
    # Build model
    model = config.model.create()
    
    # Ensure all processes finish model creation before DeepSpeed init
    if use_ddp and use_deepspeed:
        logging.info("Synchronizing after model creation before DeepSpeed init...")
        dist.barrier()
    
    # Apply parameter freezing based on freeze_filter (if defined)
    # if config.freeze_filter is not None and isinstance(config.freeze_filter, nnx.Nothing):
    _freeze_parameters(model, config.freeze_filter, is_main)
    
    # Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    global_step = 0
    model_engine = None
    
    # Build LR schedule (shared by both DeepSpeed and standard training)
    lr_schedule = build_lr_schedule(
        peak_lr=config.lr_schedule.peak_lr,
        warmup_steps=config.lr_schedule.warmup_steps,
        decay_steps=config.lr_schedule.decay_steps,
        end_lr=config.lr_schedule.decay_lr,
    )
    
    if use_deepspeed:
        # DeepSpeed initialization
        if use_ddp:
            logging.info("Waiting for all ranks before DeepSpeed initialization...")
            dist.barrier()
        logging.info("Initializing DeepSpeed...")
        
        # Update DeepSpeed config with our training parameters
        if deepspeed_config is not None:
            ds_config = deepspeed_config.copy()
            # DeepSpeed requires: train_batch_size = micro_batch_per_gpu * gradient_accumulation_steps * world_size
            # effective_batch_size is already batch_size // world_size (per GPU batch size)
            # So train_batch_size should be: effective_batch_size * gradient_accumulation_steps * world_size
            ds_config["train_batch_size"] = effective_batch_size * config.gradient_accumulation_steps * world_size
            ds_config["train_micro_batch_size_per_gpu"] = effective_batch_size
            ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
            ds_config["gradient_clipping"] = config.optimizer.clip_gradient_norm
            
            if is_main:
                logging.info(
                    f"DeepSpeed batch config: train_batch_size={ds_config['train_batch_size']}, "
                    f"micro_batch_per_gpu={ds_config['train_micro_batch_size_per_gpu']}, "
                    f"gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}, "
                    f"world_size={world_size}"
                )
            
            # Update optimizer params
            if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
                ds_config["optimizer"]["params"]["lr"] = config.lr_schedule.peak_lr
                ds_config["optimizer"]["params"]["betas"] = [config.optimizer.b1, config.optimizer.b2]
                ds_config["optimizer"]["params"]["eps"] = config.optimizer.eps
                ds_config["optimizer"]["params"]["weight_decay"] = config.optimizer.weight_decay
            
            # Remove scheduler from DeepSpeed config since we use custom lr_schedule
            # DeepSpeed will not initialize a scheduler if it's not in the config
            if "scheduler" in ds_config:
                if is_main:
                    logging.info("Removing DeepSpeed scheduler config - using custom lr_schedule instead")
                del ds_config["scheduler"]
            
        else:
            ds_config = None
        
        # Create optimizer before DeepSpeed to avoid FusedAdam compilation issues
        optimizer = _create_optimizer(model, config)
        
        model_engine, optim, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,  # Pass PyTorch optimizer to avoid FusedAdam
            config=ds_config,
        )
        
        if use_ddp:
            logging.info("Synchronizing after DeepSpeed initialization...")
            dist.barrier()
            logging.info("DeepSpeed initialization finished on all ranks")
        
        # Note: We use our own LR schedule for compatibility (defined above)
        model = model_engine.module
        
        # Resume from checkpoint (DeepSpeed)
        if resuming:
            if checkpoint_manager is None:
                raise ValueError("checkpoint_manager is required when resuming training")
            # Use our checkpoint manager which handles both DeepSpeed and PyTorch formats
            global_step = checkpoint_manager.load_latest(
                model=model_engine.module, 
                optimizer=optim, 
                device=device, 
                model_engine=model_engine
            )
            logging.info(f"Resumed DeepSpeed training from step {global_step}")
        
        if is_main:
            logging.info("DeepSpeed initialized successfully")
            logging.info(f"DeepSpeed ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
    else:
        # Standard PyTorch training
        model = model.to(device)
        
        # DDP wrap
        if use_ddp:
            world_size = torch.distributed.get_world_size()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                static_graph=world_size >= 8,
            )

        # Log parameter statistics
        model_to_optimize = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if is_main:
            total_params = sum(p.numel() for p in model_to_optimize.parameters())
            trainable_params_count = sum(p.numel() for p in model_to_optimize.parameters() if p.requires_grad)
            frozen_params_count = total_params - trainable_params_count
            logging.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            logging.info(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/1e6:.2f}M)")
            logging.info(f"Frozen parameters: {frozen_params_count:,} ({frozen_params_count/1e6:.2f}M)")
            logging.info(f"Trainable ratio: {trainable_params_count/total_params*100:.2f}%")
        
        # Create optimizer (shared logic via helper function)
        optim = _create_optimizer(model_to_optimize, config)
        # Note: LR schedule is defined above, shared with DeepSpeed branch

        # Resume (only for non-DeepSpeed; DeepSpeed handles resume internally)
        if resuming:
            if checkpoint_manager is None:
                raise ValueError("checkpoint_manager is required when resuming training")
            global_step = checkpoint_manager.load_latest(model, optim, device, model_engine=None)
            logging.info(f"Resumed training from step {global_step}")

    # Summary logging (match train.py style)
    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={world_size}")
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}, deepspeed={use_deepspeed}")
        logging.info(
            f"LR schedule: warmup={config.lr_schedule.warmup_steps}, peak_lr={config.lr_schedule.peak_lr:.2e}, decay_steps={config.lr_schedule.decay_steps}, end_lr={config.lr_schedule.decay_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")

    return model, optim, lr_schedule, global_step, model_engine


def train_step(
    config: _config.TrainConfig,
    *,
    model,
    optim,
    batch,
    device: torch.device,
    lr_schedule,
    global_step: int,
    model_engine=None,
    gradient_accumulation_steps: int = 1,  # enable gradient accumulation for effective larger batch size
    accumulation_step: int = 0,  # current accumulation step
):
    """Training step compatible with both standard PyTorch and DeepSpeed."""
    
    if model_engine is not None:
        # DeepSpeed training step
        model_engine.train()
        observation, actions = batch
        observation = jax.tree.map(lambda x: x.to(model_engine.local_rank), observation)
        actions = actions.to(torch.float32)
        actions = actions.to(model_engine.local_rank)

        # Forward
        losses = model_engine(observation, actions)
        if isinstance(losses, (list, tuple)):
            loss, aux_loss = losses[0].mean(), losses[1]
        else:
            loss, aux_loss = losses.mean(), {"action_loss": losses.mean()}
        
        # Update LR using our custom schedule (before backward/step)
        # Since we removed DeepSpeed scheduler, we need to manually update LR
        current_lr = lr_schedule(global_step)
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr
        
        # DeepSpeed backward and step
        model_engine.backward(loss)
        model_engine.step()
        
        # Get grad norm (DeepSpeed handles clipping internally)
        grad_norm = 0.0
        if hasattr(model_engine, 'get_global_grad_norm'):
            grad_norm_value = model_engine.get_global_grad_norm()
            if grad_norm_value is not None:
                grad_norm = grad_norm_value.item() if isinstance(grad_norm_value, torch.Tensor) else float(grad_norm_value)
        
        info = {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "lr": float(current_lr),
        }
        info.update(aux_loss)
        return info
    
    else:
        # Standard PyTorch training step
        model.train()
        observation, actions = batch
        observation = jax.tree.map(
            lambda x: x.to(device) if hasattr(x, "to") else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(device)

        # Update LR
        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        # Forward
        losses = model(observation, actions)
        if isinstance(losses, (list, tuple)):
            loss, aux_loss = losses[0].mean(), losses[1]
        else:
            loss, aux_loss = losses.mean(), {"action_loss": losses.mean()}

        # divide loss by accumulation steps before backward
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        # Backward ()
        loss.backward()

        # run optimizer step and gradient sync only after accumulation
        is_accumulation_complete = (accumulation_step + 1) % gradient_accumulation_steps == 0
        
        if is_accumulation_complete:
            # Gradient clipping ()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
            )
            # Optimizer step ( DDP )
            optim.step()
            optim.zero_grad(set_to_none=True)
        else:
            # during accumulation, skip optimizer step and keep gradients
            # gradients are accumulated in param.grad
            grad_norm = 0.0

        # Return info dict like train.py's ptrain_step
        # Ensure grad_norm is always a float
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        elif grad_norm is None:
            grad_norm = 0.0
        else:
            grad_norm = float(grad_norm)
        
        info = {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "lr": float(optim.param_groups[0]["lr"]),
        }
        info.update(aux_loss)
        return info


def main(config: _config.TrainConfig):
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Using{openpi.__file__}")

    # Check for DeepSpeed config
    use_deepspeed = config.deepspeed_config is not None
    deepspeed_config_dict = None
    
    if use_deepspeed:
        import json
        deepspeed_config_path = config.deepspeed_config
        if os.path.exists(deepspeed_config_path):
            with open(deepspeed_config_path, 'r') as f:
                deepspeed_config_dict = json.load(f)

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )
    
    # Enable memory optimizations for all GPU training (not needed for DeepSpeed as it manages this)
    if not use_deepspeed:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator config to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations and CUDA allocator configuration")
    
    # Initialize checkpoint manager and wandb
    manager = _checkpoints.TorchCheckpointManager(
        config.checkpoint_dir,
        save_interval=config.save_interval,
        num_train_steps=config.num_train_steps,
        wandb_enabled=config.wandb_enabled,
        overwrite=config.overwrite,
        resume=config.resume,
        is_main=is_main,
    )
    resuming = manager.resuming

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    if hasattr(config.model, "training"):  # used for tokenizer training mode
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, training=True))

    # Use the unified data loader with PyTorch framework
    data_loader = _data_loader.create_data_loader(
        config,
        framework="pytorch", 
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    data_config = data_loader.data_config()
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # if is_main and config.wandb_enabled:
    #     # Log images from first batch to sanity check.
    #     images_to_log = [
    #         wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #         for i in range(min(5, len(next(iter(batch[0].images.values())))))
    #     ]
    #     wandb.log({"camera_views": images_to_log}, step=0)

    model, optim, lr_schedule, global_step, model_engine = init_train_state(
        config,
        device,
        use_ddp=use_ddp,
        resuming=resuming,
        is_main=is_main,
        world_size=world_size,
        effective_batch_size=effective_batch_size,
        checkpoint_manager=manager,
        use_deepspeed=use_deepspeed,
        deepspeed_config=deepspeed_config_dict,
    )

    if is_main and hasattr(data_loader, "set_epoch"):
        data_loader.set_epoch(global_step // max(1, len(data_loader)))

    if is_main and torch.cuda.is_available():
        _checkpoints.TorchCheckpointManager.log_memory_usage(device, 0, "after_model_creation")

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        manager.load(model, optim, device, epath.Path(config.pytorch_weight_path), model_engine=model_engine)
        
    ptrain_step = functools.partial(train_step, config)
    start_step = global_step
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not is_main,
    )

    infos = []
    # gradient accumulation
    gradient_accumulation_steps = config.gradient_accumulation_steps  # read from config
    accumulation_counter = 0  # accumulation counter
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            if use_ddp and hasattr(data_loader, "set_epoch"):
                data_loader.set_epoch(step // max(1, len(data_loader)))
            data_iter = iter(data_loader)
            batch = next(data_iter)

        info = ptrain_step(
            model=model,
            optim=optim,
            batch=batch,
            device=device,
            lr_schedule=lr_schedule,
            global_step=step,
            model_engine=model_engine,
            gradient_accumulation_steps=gradient_accumulation_steps,
            accumulation_step=accumulation_counter,
        )
        accumulation_counter += 1
        if accumulation_counter >= gradient_accumulation_steps:
            accumulation_counter = 0
        if is_main: infos.append(info)
        if is_main and (step % config.log_interval == 0):
            keys = infos[0].keys()
            reduced_info = {k: sum(d[k] for d in infos) / len(infos) for k in keys if all(k in d for d in infos)}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if config.wandb_enabled: wandb.log(reduced_info, step=step)
            infos = []
            
        manager.save(model, optim, step, data_config, model_engine=model_engine)

    cleanup_ddp()


if __name__ == "__main__":
    init_logging()
    
    # DeepSpeed adds --local_rank argument, extract it and remove before parsing
    import sys
    import os
    local_rank = None
    new_argv = []
    skip_next = False
    
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        
        # Check for --local_rank=N or --local-rank=N
        if arg.startswith('--local_rank=') or arg.startswith('--local-rank='):
            local_rank = arg.split('=')[1]
            continue  # Skip this argument
        
        # Check for --local_rank N or --local-rank N
        elif (arg == '--local_rank' or arg == '--local-rank'):
            if i + 1 < len(sys.argv):
                local_rank = sys.argv[i + 1]
                skip_next = True  # Skip next argument (the value)
            continue  # Skip this argument
        
        new_argv.append(arg)
    
    sys.argv = new_argv
    
    if local_rank is not None:
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    main(_config.cli())
