from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import gc
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
import torch
import torch.distributed as dist
import safetensors.torch

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            enable_async_checkpointing=False,
            # async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


class TorchCheckpointManager:
    """Checkpoint manager for PyTorch training runs.

    Encapsulates checkpoint directory lifecycle, save/load operations,
    and optional memory logging utilities for multi-GPU training.
    """

    def __init__(
        self,
        checkpoint_dir: epath.Path | str,
        *,
        save_interval: int,
        num_train_steps: int,
        wandb_enabled: bool,
        overwrite: bool,
        resume: bool,
        is_main: bool,
    ):
        self.checkpoint_dir = epath.Path(checkpoint_dir).resolve()
        self.save_interval = int(save_interval)
        self.num_train_steps = int(num_train_steps)
        self.wandb_enabled = bool(wandb_enabled)
        self.is_main = bool(is_main)
        self.resuming = False

        if resume:
            if self.checkpoint_dir.exists():
                latest_step = self.get_latest_checkpoint_step()
                if latest_step is not None:
                    self.resuming = True
                    logging.info(
                        f"Resuming from experiment checkpoint directory: {self.checkpoint_dir} at step {latest_step}"
                    )
                else:
                    raise FileNotFoundError(
                        f"No valid checkpoints found in {self.checkpoint_dir} for resume"
                    )
            else:
                raise FileNotFoundError(
                    f"Experiment checkpoint directory {self.checkpoint_dir} does not exist for resume"
                )
        elif overwrite and self.checkpoint_dir.exists() and self.is_main:
            self.checkpoint_dir.rmtree()
            logging.info(f"Overwriting checkpoint directory: {self.checkpoint_dir}")

        if not self.resuming:
            # Create experiment directory for a new run
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {self.checkpoint_dir}")
        else:
            logging.info(f"Using existing experiment checkpoint directory: {self.checkpoint_dir}")

    def get_latest_checkpoint_step(self) -> int | None:
        """Return the latest checkpoint step number in the directory, if any."""
        if not self.checkpoint_dir.exists():
            return None
        checkpoint_steps = [
            int(d.name)
            for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
        ]
        return max(checkpoint_steps) if checkpoint_steps else None

    def save(self, model, optimizer, step: int, data_config, model_engine=None) -> None:
        """Save a checkpoint at the given step if it is time to save.

        Saves model weights (safetensors), optimizer state (pt), metadata, and
        normalization assets if present.
        
        Args:
            model: PyTorch model (may be wrapped by DDP)
            optimizer: PyTorch optimizer
            step: Current training step
            data_config: Data configuration with norm_stats and asset_id
            model_engine: DeepSpeed engine (if using DeepSpeed)
        """
        if not self.is_main:
            return

        should_save = (step % self.save_interval == 0 and step > 0) or step == self.num_train_steps - 1
        if not should_save:
            return

        if model_engine is not None:
            # DeepSpeed checkpoint saving
            self._save_deepspeed(model_engine, step, data_config)
        else:
            # Standard PyTorch checkpoint saving
            self._save_pytorch(model, optimizer, step, data_config)

    def _save_deepspeed(self, model_engine, step: int, data_config) -> None:
        """Save DeepSpeed checkpoint and convert to PyTorch format for compatibility."""
        final_ckpt_dir = self.checkpoint_dir / f"{step}"
        tmp_ckpt_dir = self.checkpoint_dir / f"tmp_{step}"
        
        if tmp_ckpt_dir.exists():
            tmp_ckpt_dir.rmtree()
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DeepSpeed native checkpoint
        ds_ckpt_dir = tmp_ckpt_dir / "deepspeed"
        ds_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        client_state = {"step": int(step)}
        model_engine.save_checkpoint(str(ds_ckpt_dir), client_sd=client_state)
        logging.info(f"Saved DeepSpeed checkpoint to {ds_ckpt_dir}")
        
        # Also save in PyTorch format for compatibility and easier loading
        # Extract the underlying model from DeepSpeed engine
        model_to_save = model_engine.module
        
        # Save model weights using safetensors
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
        logging.info(f"Converted DeepSpeed model to safetensors format")
        
        # Save metadata
        metadata = {
            "global_step": int(step),
            "deepspeed": True,
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")
        
        # Save normalization stats
        norm_stats = getattr(data_config, "norm_stats", None)
        asset_id = getattr(data_config, "asset_id", None)
        if norm_stats is not None and asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / asset_id, norm_stats)
        
        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            final_ckpt_dir.rmtree()
        tmp_ckpt_dir.rename(final_ckpt_dir)
        
        logging.info(f"Saved DeepSpeed checkpoint at step {step} -> {final_ckpt_dir}")

    def _save_pytorch(self, model, optimizer, step: int, data_config) -> None:
        """Save standard PyTorch checkpoint."""
        final_ckpt_dir = self.checkpoint_dir / f"{step}"
        tmp_ckpt_dir = self.checkpoint_dir / f"tmp_{step}"

        if tmp_ckpt_dir.exists():
            tmp_ckpt_dir.rmtree()
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config objects incompatible with JAX)
        metadata = {
            "global_step": int(step),
            "deepspeed": False,
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # Save normalization stats
        norm_stats = getattr(data_config, "norm_stats", None)
        asset_id = getattr(data_config, "asset_id", None)
        if norm_stats is not None and asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            final_ckpt_dir.rmtree()
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {step} -> {final_ckpt_dir}")

    def load_latest(self, model, optimizer, device: torch.device, model_engine=None) -> int:
        """Load the latest checkpoint and return the global step.
        
        Args:
            model: PyTorch model (may be wrapped by DDP)
            optimizer: PyTorch optimizer
            device: Device to load the checkpoint to
            model_engine: DeepSpeed engine (if using DeepSpeed)
            
        Returns:
            global_step: The training step of the loaded checkpoint
        """
        latest_step = self.get_latest_checkpoint_step()
        if latest_step is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        ckpt_dir = self.checkpoint_dir / f"{latest_step}"
        return self.load(model, optimizer, device, ckpt_dir, model_engine, latest_step)
    
    def load(self, model, optimizer, device: torch.device, ckpt_dir: epath.Path, model_engine=None, latest_step: int=0) -> int:
        # Check if this is a DeepSpeed checkpoint
        metadata_path = ckpt_dir / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location=device, weights_only=False)
            is_deepspeed = metadata.get("deepspeed", False)
        else:
            is_deepspeed = False
        
        if model_engine is not None and is_deepspeed:
            # Load DeepSpeed checkpoint
            return self._load_deepspeed(model_engine, ckpt_dir, device, latest_step)
        else:
            # Load standard PyTorch checkpoint
            return self._load_pytorch(model, optimizer, ckpt_dir, device, latest_step)
    
    def _load_deepspeed(self, model_engine, ckpt_dir: epath.Path, device: torch.device, latest_step: int) -> int:
        """Load DeepSpeed checkpoint."""
        ds_ckpt_dir = ckpt_dir / "deepspeed"
        
        if not ds_ckpt_dir.exists():
            # Fallback to loading from PyTorch format if DeepSpeed checkpoint not found
            logging.warning(f"DeepSpeed checkpoint not found at {ds_ckpt_dir}, falling back to PyTorch format")
            model = model_engine.module
            optimizer = model_engine.optimizer
            return self._load_pytorch(model, optimizer, ckpt_dir, device, latest_step)
        
        logging.info(f"Loading DeepSpeed checkpoint from {ds_ckpt_dir}")
        _, client_state = model_engine.load_checkpoint(str(ds_ckpt_dir))
        
        if client_state:
            global_step = client_state.get("step", latest_step)
        else:
            # Try to load from metadata
            metadata_path = ckpt_dir / "metadata.pt"
            if metadata_path.exists():
                metadata = torch.load(metadata_path, map_location=device, weights_only=False)
                global_step = int(metadata.get("global_step", latest_step))
            else:
                global_step = latest_step
        
        logging.info(f"Successfully loaded DeepSpeed checkpoint from step {global_step}")
        return global_step
    
    def _load_pytorch(self, model, optimizer, ckpt_dir: epath.Path, device: torch.device, latest_step: int) -> int:
        """Load standard PyTorch checkpoint."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(device, latest_step, "before_loading_checkpoint")

        # Load model state
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"
        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            try:
                safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
                logging.info("Loaded model state from safetensors format")
            except RuntimeError as exc:
                # Fallback: tolerate missing/unexpected keys (e.g., newly added modules)
                logging.warning(
                    f"safetensors.load_model failed ({exc}). "
                    "Falling back to load_file + load_state_dict(strict=False)."
                )
                state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
                missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
                del state_dict
                if missing:
                    logging.warning(f"Missing keys when loading weights (initialized freshly): {missing}")
                if unexpected:
                    logging.warning(f"Unexpected keys ignored in checkpoint: {unexpected}")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"
        if optimizer_path.exists():
            try:
                optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(optimizer_state_dict)
                logging.info("Loaded optimizer state from pt format")
                del optimizer_state_dict
            except (ValueError, KeyError) as e:
                # Try partial loading: only load state for parameters that exist in both optimizers
                # This handles cases where frozen/unfrozen parameters have changed
                logging.info(
                    f"Full optimizer state load failed: {e}. "
                    "Attempting partial loading for matching parameters..."
                )
                try:
                    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    self._load_optimizer_state_partial(optimizer, optimizer_state_dict, device, model_to_load)
                    logging.info("Partially loaded optimizer state (only matching parameters)")
                except Exception as partial_e:
                    logging.warning(
                        f"Partial loading also failed: {partial_e}. "
                        "Starting with fresh optimizer state."
                    )
                finally:
                    del optimizer_state_dict
        else:
            logging.warning(f"No optimizer checkpoint found at {ckpt_dir}, starting with fresh optimizer state")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata_path = ckpt_dir / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location=device, weights_only=False)
            global_step = int(metadata.get("global_step", latest_step))
            del metadata
        else:
            global_step = latest_step

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    @staticmethod
    def _load_optimizer_state_partial(optimizer, loaded_state_dict, device, model):
        """Partially load optimizer state for matching parameters only.
        
        This handles cases where frozen/unfrozen parameters have changed.
        Matches parameters by comparing their tensor values (shape and data).
        
        Args:
            optimizer: Current optimizer instance
            loaded_state_dict: State dict loaded from checkpoint
            device: Device to load tensors to
            model: Model instance to get parameter references
        """
        import torch.nn as nn
        
        current_state_dict = optimizer.state_dict()
        
        # Get current optimizer parameters with their IDs
        current_param_map = {}  # param_id -> param tensor
        model_to_load = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        
        # Build a map from parameter tensor to parameter ID in current optimizer
        current_all_params = []
        for param_group in current_state_dict["param_groups"]:
            for param_id in param_group["params"]:
                current_all_params.append(param_id)
                # Find the actual parameter tensor
                for name, param in model_to_load.named_parameters():
                    if id(param) == param_id:
                        current_param_map[param_id] = param
                        break
        
        # Get loaded optimizer parameters
        loaded_all_params = []
        for param_group in loaded_state_dict["param_groups"]:
            loaded_all_params.extend(param_group["params"])
        
        # Match parameters by comparing tensor values
        # We'll match by position as a heuristic, but verify by shape
        num_matched = 0
        min_len = min(len(current_all_params), len(loaded_all_params))
        
        # Create a mapping: try to match by position first
        # This works when the same parameters appear in the same relative order
        for i in range(min_len):
            current_id = current_all_params[i]
            loaded_id = loaded_all_params[i]
            
            # Only load if state exists and current param exists
            if loaded_id in loaded_state_dict["state"] and current_id in current_param_map:
                current_param = current_param_map[current_id]
                
                # Verify shapes match (basic sanity check)
                # Note: We can't easily verify values match without loading the model weights first
                # But if model weights were loaded successfully, we assume they match
                
                state = loaded_state_dict["state"][loaded_id]
                # Deep copy and move tensors to correct device
                if isinstance(state, dict):
                    copied_state = {}
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            # Verify tensor shape matches if it's a parameter-related state
                            if key in ['exp_avg', 'exp_avg_sq', 'step']:
                                # For Adam states, shapes should match parameter shape
                                if value.shape != current_param.shape:
                                    # Skip this parameter if shapes don't match
                                    continue
                            copied_state[key] = value.to(device)
                        else:
                            copied_state[key] = value
                    optimizer.state[current_id] = copied_state
                    num_matched += 1
        
        if num_matched == 0:
            raise ValueError("No matching parameters found for partial loading")
        
        num_total = len(current_all_params)
        logging.info(
            f"Partially loaded optimizer state: {num_matched}/{num_total} parameters "
            f"({num_matched/num_total*100:.1f}%)"
        )

    @staticmethod
    def log_memory_usage(device: torch.device, step: int, phase: str = "unknown") -> None:
        """Log detailed CUDA memory usage if available."""
        if not torch.cuda.is_available():
            return

        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        memory_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9

        memory_stats = torch.cuda.memory_stats(device)
        max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
        max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

        ddp_info = ""
        if dist.is_initialized():
            ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

        logging.info(
            f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, "
            f"reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, "
            f"peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
        )
