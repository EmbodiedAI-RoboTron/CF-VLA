import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.shared.registry import register_weight_loader

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """

@register_weight_loader("NoOpWeightLoader")
@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@register_weight_loader("CheckpointWeightLoader")
@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@register_weight_loader("CheckpointWeightLoaderNState")
@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderNState(WeightLoader):
    params_path: str
    state_dim: int = 32

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        if self.state_dim != 32:
            expanded_kernel = np.zeros(params['state_proj']['kernel'].shape, dtype= params['state_proj']['kernel'].dtype)
            expanded_kernel[:32, :] = loaded_params['state_proj']['kernel']
            # randomly initialize the added 10 rows (Xavier or He)
            # Method 2a: Xavier initialization
            std = np.sqrt(2.0 / (1024 + 42))
            expanded_kernel[32:, :] = np.random.normal(0, std, (10, 1024))

            loaded_params['state_proj'] = {'kernel': expanded_kernel,
                                        'bias': loaded_params['state_proj']['bias']
                                    }
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@register_weight_loader("Pi0WMWeightLoaderFromPi0")
@dataclasses.dataclass(frozen=True)
class Pi0WMWeightLoaderFromPi0(WeightLoader):
    """Loads weights from the official Pi0WM checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        loaded_params['latent_action_time_mlp_in'] = loaded_params['action_time_mlp_in']
        loaded_params['latent_action_time_mlp_out'] = loaded_params['action_time_mlp_out']
        loaded_params['latent_in_proj'] = {'kernel': loaded_params['action_in_proj']['kernel'][:4],
                                    'bias': loaded_params['action_in_proj']['bias']
                                }
        loaded_params['latent_out_proj'] = {'kernel': loaded_params['action_out_proj']['kernel'][:, :4],
                                    'bias': loaded_params['action_out_proj']['bias'][:4]
                                }
        loaded_params['PaliGemma']['llm']['final_norm_2'] = loaded_params['PaliGemma']['llm']['final_norm_1']
        loaded_params['PaliGemma']['llm']['layers']['pre_ffw_norm_2'] = loaded_params['PaliGemma']['llm']['layers']['pre_ffw_norm_1']
        loaded_params['PaliGemma']['llm']['layers']['mlp_2'] = loaded_params['PaliGemma']['llm']['layers']['mlp_1']
        loaded_params['PaliGemma']['llm']['layers']['pre_attention_norm_2'] = loaded_params['PaliGemma']['llm']['layers']['pre_attention_norm_1']
        loaded_params['PaliGemma']['llm']['layers']['attn']['q_einsum_2'] = loaded_params['PaliGemma']['llm']['layers']['attn']['q_einsum_1']
        loaded_params['PaliGemma']['llm']['layers']['attn']['attn_vec_einsum_2'] = loaded_params['PaliGemma']['llm']['layers']['attn']['attn_vec_einsum_1']
        loaded_params['PaliGemma']['llm']['layers']['attn']['kv_einsum_2'] = loaded_params['PaliGemma']['llm']['layers']['attn']['kv_einsum_1']

        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
