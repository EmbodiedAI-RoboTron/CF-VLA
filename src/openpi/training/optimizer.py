import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
from openpi.shared.registry import register_scheduler, register_optimizer_config

@runtime_checkable
class LRScheduleConfig(Protocol):
    """Learning rate schedule configuration protocol.
    
    This protocol requires a create method that takes no additional parameters
    and returns an optax.Schedule.
    """
    def create(self) -> optax.Schedule: ...


@register_scheduler("cosine_decay_schedule")
@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    init_lr: float = 0.0

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.init_lr,
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@register_scheduler("rsqrt_decay_schedule")
@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    """Optimizer configuration protocol.
    
    This protocol requires a create method that takes lr and weight_decay_mask parameters
    and returns an optax.GradientTransformation.
    """
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@register_optimizer_config("adamw")
@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@register_optimizer_config("sgd")
@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask), lr


# Type checking helper functions to avoid Protocol ambiguity
def is_lr_schedule_config(obj) -> bool:
    """Check if an object is specifically a learning rate schedule config.
    
    This function checks the actual class type rather than relying on Protocol
    to avoid ambiguity with OptimizerConfig.
    """
    return isinstance(obj, (CosineDecaySchedule, RsqrtDecaySchedule))


def is_optimizer_config(obj) -> bool:
    """Check if an object is specifically an optimizer config.
    
    This function checks the actual class type rather than relying on Protocol
    to avoid ambiguity with LRScheduleConfig.
    """
    return isinstance(obj, (AdamW, SGD))


def get_config_type(obj) -> str:
    """Get the specific type of a configuration object.
    
    Returns:
        str: The type name ('lr_schedule', 'optimizer', or 'unknown')
    """
    if is_lr_schedule_config(obj):
        return 'lr_schedule'
    elif is_optimizer_config(obj):
        return 'optimizer'
    else:
        return 'unknown'
