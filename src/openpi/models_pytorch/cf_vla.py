import dataclasses
import logging
import math
import copy
from typing import Sequence, Optional
import random
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from typing_extensions import override
import safetensors

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.shared.registry import register_model_config


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention. each token can attend only to itself and previous tokens.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour. the first 3 tokens can attend to each other; the last 3 use causal attention.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block. causal attention across 4 blocks; each block can attend within block and previous blocks.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


@register_model_config("pi0_2stg_pytorch")
@dataclasses.dataclass(frozen=True)
class Pi02stgPytorchConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    state_dim: int = 32  #  this value is not overridden at instantiation; update it when changing state/action dims
    action_dim: int = 32
    action_horizon: int = 50  # number of output actions (action chunk size)
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    input_size: Sequence[int] = (224, 224)  # translated comment
    model_kwargs: dict = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05_2STG_PYTORCH
        return _model.ModelType.PI0_2STG_PYTORCH

    @override
    def create(self) -> "Pi02stgPytorch":
        if self.state_dim < 32:  # openpi default state_dim is 32; clamp to 32 if smaller
            self.state_dim = 32
        return Pi02stgPytorch(self)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        return None, None

    def get_freeze_filter(self):
        return self.freeze_filter

    def load_pytorch(self, train_config, weight_path: str):
        logger = logging.getLogger(__name__)
        logger.info(f"train_config: {train_config}")
        model = Pi02stgPytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        # model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        return model


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -5.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        if generator is not None:
            sample = sample.to(generator.device)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2],
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )

    def nll(self, sample: torch.Tensor, mask: torch.Tensor = None, dims: Sequence[int] = [1, 2]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        # Use torch tensor constant to ensure proper dtype/device and numerical stability
        logtwopi = sample.new_tensor(math.log(2.0 * math.pi))
        # NLL = 0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
        nll_per_element = logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var
        if mask is not None:
            nll_per_element[mask] = torch.pow(sample - self.mean, 2)[mask] + (self.var - 0.0)[mask]
        
        return 0.5 * torch.mean(nll_per_element, dim=dims)

    def mode(self) -> torch.Tensor:
        return self.mean


class Pi02stgPytorch(nn.Module):
    def __init__(self, config: Pi02stgPytorchConfig):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.input_size = config.input_size
        self.model_kwargs = config.model_kwargs

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)
        self.action_out_proj_logvar = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")  # Python / graph

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        # try:
        #     from transformers.models.siglip import check

        #     if not check.check_whether_transformers_replace_is_installed_correctly():
        #         raise ValueError(msg)
        # except ImportError:
        #     raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI02TGPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI02TGPytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, 
            train=train, 
            image_resolution=self.input_size
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(self.model_kwargs.get("alpha", 1.5), self.model_kwargs.get("beta", 1.0), bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward_suffix(self, state, x_t, time, prefix_pad_masks, past_key_values, ref):

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        if suffix_embs.dtype != next(self.paligemma_with_expert.parameters()).dtype:
            suffix_embs = suffix_embs.to(dtype=next(self.paligemma_with_expert.parameters()).dtype)
        
        # attention_mask dtype embeddings dtype ， SDPA dtype
        if prefix_att_2d_masks_4d.dtype != suffix_embs.dtype:
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=suffix_embs.dtype)
        # Apply gradient checkpointing if enabled
        def forward_func(suffix_embs, suffix_att_2d_masks_4d, position_ids, adarms_cond, past_key_values):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=suffix_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                cache_update=False,
                adarms_cond=[None, adarms_cond],
            )
            return outputs[1]
        # Ensure eager attention also for suffix path
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        hidden_states = self._apply_checkpoint(
            forward_func, suffix_embs, prefix_att_2d_masks_4d, position_ids, adarms_cond, past_key_values
        )

        suffix_out = hidden_states[:, -self.config.action_horizon :]
        if suffix_out.dtype != next(self.action_out_proj.parameters()).dtype:
            suffix_out = suffix_out.to(dtype=next(self.action_out_proj.parameters()).dtype)
        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return torch.cat([self.action_out_proj(suffix_out), self.action_out_proj_logvar(suffix_out)], dim=-1)
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return DiagonalGaussianDistribution(v_t)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        if prefix_embs.dtype != next(self.paligemma_with_expert.parameters()).dtype:
            prefix_embs = prefix_embs.to(dtype=next(self.paligemma_with_expert.parameters()).dtype)
        
        # attention_mask dtype embeddings dtype ， SDPA dtype
        if prefix_att_2d_masks_4d.dtype != prefix_embs.dtype:
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)
        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids):
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return past_key_values
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        past_key_values = self._apply_checkpoint(forward_func, prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids)

        ref, noise_func = None, None
        if self.model_kwargs.get("use_1stg", False):
            if self.model_kwargs.get("1stg_rand_noise", False):
                one_noise = self.sample_noise(actions.shape, actions.device)
            else:
                one_noise = torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)  # 2*50*32
            one_time = torch.ones(actions.shape[0], dtype=torch.float32, device=actions.device)  # 2
            if self.model_kwargs.get("times_t1", 0) > 0:
                one_time *= self.model_kwargs.get("times_t1", 0)
                x_t = one_noise
            else:
                time_expanded = one_time[:, None, None]
                x_t = time_expanded * one_noise + (1 - time_expanded) * actions
            one_u_t = one_noise - actions

            posterior = self.forward_suffix(state, x_t, one_time, prefix_pad_masks, past_key_values, None)
            if one_u_t.dtype != posterior.mode().dtype:
                one_u_t = one_u_t.to(dtype=posterior.mode().dtype)

            if self.model_kwargs.get("1stg_loss_type", "kl") == "kl":
                other = torch.cat([self.model_kwargs.get("noise_scale", 1.0) * one_u_t, torch.log(torch.tensor(self.model_kwargs.get("noise_var", 1.0)) * torch.ones_like(one_u_t))], dim=-1)
                kl_loss = posterior.kl(DiagonalGaussianDistribution(other))
            elif self.model_kwargs.get("1stg_loss_type", "kl") == "mse":
                mse_loss = F.mse_loss(one_u_t, posterior.mode(), reduction="none")
                if self.model_kwargs.get("noise_var", 1.0) != 1.0:
                    target_logvar = torch.log(torch.tensor(self.model_kwargs.get("noise_var", 1.0))) * torch.ones_like(one_u_t)
                    mse_loss = mse_loss + self.model_kwargs.get("loss_logvar_weight", 1) * F.mse_loss(target_logvar, posterior.logvar, reduction="none")
                kl_loss = mse_loss.mean(dim=[1,2]) 
            elif self.model_kwargs.get("1stg_loss_type", "kl") == "nll":
                kl_loss = posterior.nll(one_u_t, mask=observation.extra_data.get('actions_mask', None))
            elif self.model_kwargs.get("1stg_loss_type", "kl") == None:
                kl_loss = torch.zeros(one_u_t.shape[0], device=one_u_t.device, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown noise loss type: {self.model_kwargs.get('1stg_loss_type', 'kl')}")
            if self.model_kwargs.get("1stg_output_mode", "mode") == "mode":
                noise_func = lambda: one_noise - posterior.mode()
            else:
                noise_func = lambda: one_noise - posterior.sample()

            if self.model_kwargs.get("only_1stg", False):
                return kl_loss

        action_losses = []
        for _ in range(self.model_kwargs.get("flow_num", 1)):
            # if self.model_kwargs.get("1stg_as_ref", False):
            #     ref = noise_func()
            if self.model_kwargs.get("use_0_noise", False):
                noise = torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)  # 2*50*32
            elif self.model_kwargs.get("1stg_as_noise", False):
                noise = noise_func()   # C+F stg12stg
            elif noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if self.model_kwargs.get("discrete_time_steps", 0)>0:
                time = torch.tensor(random.choices(range(self.model_kwargs.get("discrete_time_steps", 0), 0,-1), k=actions.shape[0]), dtype=torch.float32, device=actions.device)/self.model_kwargs.get("discrete_time_steps", 0)
            elif self.model_kwargs.get("times_list_train", None) is not None:
                time = torch.tensor(random.choices(self.model_kwargs.get("times_list_train", None), k=actions.shape[0]), dtype=torch.float32, device=actions.device)
            elif time is None:
                time = self.sample_time(actions.shape[0], actions.device)
            if self.model_kwargs.get("1stg_detach", False):
                noise = noise.detach()

            time_expanded = time[:, None, None]
            if self.model_kwargs.get("noisy_actions", "diffusion_mixing") == "clean_noise":
                x_t = noise
            elif self.model_kwargs.get("noisy_actions", "diffusion_mixing") == "noise_injection":
                x_t = time_expanded * noise + actions
            else:  # diffusion_mixing
                x_t = time_expanded * noise + (1 - time_expanded) * actions
            if self.model_kwargs.get("output_actions", False):
                u_t = -actions
            else:
                u_t = noise - actions
            
            posterior = self.forward_suffix(state, x_t, time, prefix_pad_masks, past_key_values, ref)
            v_t = posterior.mode()
            if u_t.dtype != v_t.dtype:
                u_t = u_t.to(dtype=v_t.dtype)

            action_loss = F.mse_loss(u_t, v_t, reduction="none")
            action_losses.append(action_loss)
        action_loss = torch.stack(action_losses).mean(dim=0)
        if self.model_kwargs.get("use_1stg", False):
            combined_loss = (
                self.model_kwargs.get('action_loss_weight', 1.0) * action_loss.mean() +
                self.model_kwargs.get('kl_loss_weight', 0.1) * kl_loss
            )
            
            # Return combined loss [B] and auxiliary loss dict for logging
            aux_loss = {
                "action_loss": action_loss.mean().item(),
                "kl_loss": kl_loss.mean().item(),
            }
            return combined_loss, aux_loss
        else:
            return action_loss.mean()

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=1) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # attention_mask dtype embeddings dtype ， SDPA dtype
        if prefix_att_2d_masks_4d.dtype != prefix_embs.dtype:
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        ref = None
        if self.model_kwargs.get("use_1stg", False):
            if self.model_kwargs.get("1stg_rand_noise", False):
                one_noise = self.sample_noise(actions_shape, device)
            else:
                one_noise = torch.zeros(actions_shape, dtype=torch.float32, device=device)  # 2*50*32
            one_num_steps = 1
            assert one_num_steps == 1, "one_num_steps must be 1"
            dt = -1.0 / one_num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = one_noise
            if self.model_kwargs.get("times_t1", 0) == 0:
                time = torch.tensor(1.0, dtype=torch.float32, device=device)
                while time >= -dt / 2:
                    expanded_time = time.expand(bsize)
                    posterior = self.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        expanded_time,
                    )
                    x_t = x_t + dt * posterior.mode()
                    time += dt
            else:
                time = torch.tensor(self.model_kwargs.get("times_t1", 0), dtype=torch.float32, device=device)
                expanded_time = time.expand(bsize)
                posterior = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt * posterior.mode()
            noise = x_t
            if self.model_kwargs.get("only_1stg", False):
                return noise
            # if self.model_kwargs.get("1stg_as_ref", False):
            #     ref = noise
            #     noise = None

        if self.model_kwargs.get("use_0_noise", False):
            noise = torch.zeros(actions_shape, dtype=torch.float32, device=device)  # 2*50*32
        elif noise is None:
            noise = self.sample_noise(actions_shape, device)
        if self.model_kwargs.get("discrete_time_steps", 0) > 0:
            num_steps = self.model_kwargs.get("discrete_time_steps", 0)
        # elif self.model_kwargs.get("noisy_actions", "diffusion_mixing") != "diffusion_mixing":
        #     num_steps = 1
        dt = -1.0 / num_steps # num_steps10
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        if self.model_kwargs.get("times_start_test", 0) > 0:
            time = time * self.model_kwargs.get("times_start_test", 0)
            dt = dt * self.model_kwargs.get("times_start_test", 0)  # translated comment
            if self.model_kwargs.get("noisy_actions", "diffusion_mixing") == "diffusion_mixing":
                x_t *= (1-self.model_kwargs.get("times_start_test", 0))
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            posterior = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            if self.model_kwargs.get("output_actions", False):
                x_t = -posterior.mode()
            else:
                x_t = x_t + self.model_kwargs.get("dt", dt) * posterior.mode()
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # attention_mask dtype embeddings dtype ， SDPA dtype
        if prefix_att_2d_masks_4d.dtype != suffix_embs.dtype:
            prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=suffix_embs.dtype)

        # Prepare attention masks
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            cache_update=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = torch.cat([self.action_out_proj(suffix_out), self.action_out_proj_logvar(suffix_out)], dim=-1)
        return DiagonalGaussianDistribution(v_t)
