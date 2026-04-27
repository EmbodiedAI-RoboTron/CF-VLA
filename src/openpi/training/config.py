"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
import sys
from typing import Any, Literal, Protocol, TypeAlias
import yaml
import copy
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
from omegaconf import OmegaConf

import openpi.models.model as _model
from openpi.shared.registry import get_model_config, get_data_config, register_data_config, get_weight_loader, get_scheduler, get_optimizer_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.calvin_policy as calvin_policy
import openpi.policies.unt_policy as unt_policy
import openpi.policies.agibot_policy as agibot_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

import os
MODEL_ZOO = os.getenv("MODEL_ZOO", "./modelzoo")

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None

@register_data_config("data_config")
@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # multi-dataset configuration
    multi_repos: dict[str, Any] | None = None
    hist_objs_config: dict[str, Any] | None = None  # historical observation object config


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    clip_state: int = -1
    clip_action: int = -1
    use_quantile_norm = False

    # multi-dataset configuration
    multi_repos: dict[str, Any] | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=True if model_config.model_type in [ModelType.PI05, ModelType.PI0_FAST] else self.use_quantile_norm,
            multi_repos=self.multi_repos,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig, data_config: DataConfigFactory) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0  \
                | _model.ModelType.PI0_PYTORCH  \
                | _model.ModelType.PI0_WM  \
                | _model.ModelType.PI0_Bar  \
                | _model.ModelType.PI0_2STG_PYTORCH  \
                | _model.ModelType.PI05_2STG_PYTORCH:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.PadClipStatesAndActions(
                            clip_state=data_config.clip_state, 
                            clip_action=data_config.clip_action,
                            model_state_dim=model_config.state_dim,
                            model_action_dim=model_config.action_dim,
                        ),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_WM_LAI:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.PadClipStatesAndActions(
                            clip_state=data_config.clip_state, 
                            clip_action=data_config.clip_action,
                            model_state_dim=model_config.state_dim,
                            model_action_dim=model_config.action_dim,
                        ),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_action_input=True,
                        ),
                    ],
                )
            case _model.ModelType.PI05 | _model.ModelType.PI05_PYTORCH:
                # assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.PadClipStatesAndActions(
                            clip_state=data_config.clip_state, 
                            clip_action=data_config.clip_action,
                            model_state_dim=model_config.state_dim,
                            model_action_dim=model_config.action_dim,
                        ),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _model.ModelType.VLA_GEMMA  \
                    | _model.ModelType.VLA_GEMMA_HIST  \
                    | _model.ModelType.VLA_GEMMA_OFFL  \
                    | _model.ModelType.VLA_WM_GEMMA  \
                    | _model.ModelType.VLA_VGGT_GEMMA  \
                    | _model.ModelType.VLA_VQA_GEMMA  \
                    | _model.ModelType.VLA_CLS_GEMMA  \
                    | _model.ModelType.VLA_REF_GEMMA  \
                    | _model.ModelType.VLA_2STG_GEMMA \
                    | _model.ModelType.Q_FUNCTION_GEMMA \
                    | _model.ModelType.VLA_3D_GEMMA:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(model_config.input_size[0], model_config.input_size[1]),
                        _transforms.PadClipStatesAndActions(
                            clip_state=data_config.clip_state, 
                            clip_action=data_config.clip_action,
                            model_state_dim=model_config.state_dim,
                            model_action_dim=model_config.action_dim,
                        ),
                        _transforms.TokenizePrompt(
                            _tokenizer.VlaGemmaTokenizer(model_config.max_token_len, model_config.model_id, tokenizer_kwargs=getattr(model_config, "tokenizer_kwargs", {})),
                        ),
                    ],
                )
            case _model.ModelType.VLA_QWEN | _model.ModelType.VLA_QWEN3_VL_OFFL:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(model_config.input_size[0], model_config.input_size[1]),
                        _transforms.PadClipStatesAndActions(
                            clip_state=data_config.clip_state, 
                            clip_action=data_config.clip_action,
                            model_state_dim=model_config.state_dim,
                            model_action_dim=model_config.action_dim,
                        ),
                        _transforms.TokenizePrompt(
                            _tokenizer.VlaQwenVLTokenizer(model_config.max_token_len, model_config.model_id, train=model_config.training),
                        ),
                    ],
                )
            case _:
                raise ValueError(f"Model type {model_config.model_type} not supported")


@register_data_config("fake")
@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@register_data_config("simple")
@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@register_data_config("lerobot_aloha")
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    extra_delta_transform: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.repack_transforms)
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config, self)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@register_data_config("lerobot_libero")
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    hist_objs_config: dict[str, Any] | None = None
    repack_transforms: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
    )
    delta_transform_type: Sequence[dict[str, Any]] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.repack_transforms)
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_config)],
            outputs=[libero_policy.LiberoOutputs(model_config)],
        )
        if self.delta_transform_type != None:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsByAbs(self.delta_transform_type)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActionsByAbs(self.delta_transform_type)],  # inverse operation of above
            )
        elif self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1) # 【T T T T T T F】
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            hist_objs_config=self.hist_objs_config,
            action_sequence_keys=self.action_sequence_keys,
        )


@register_data_config("lerobot_q_function")
@dataclasses.dataclass(frozen=True)
class LeRobotQFunctionDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    hist_objs_config: dict[str, Any] | None = None
    repack_transforms: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
    )
    delta_transform_type: Sequence[dict[str, Any]] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.repack_transforms)
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_config)],
            outputs=[],
        )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            hist_objs_config=self.hist_objs_config,
            action_sequence_keys=self.action_sequence_keys,
        )


@register_data_config("lerobot_data")
@dataclasses.dataclass(frozen=True)
class LeRobotDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    hist_objs_config: dict[str, Any] | None = None
    repack_transforms: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
    )
    delta_transform_type: Sequence[dict[str, Any]] | None = None
    data_type: str = "libero"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.repack_transforms)
            ]
        )
        if self.data_type == "libero":
            data_transforms = _transforms.Group(
                inputs=[libero_policy.LiberoInputs(model_config)],
                outputs=[libero_policy.LiberoOutputs(model_config)],
            )
        elif self.data_type == "calvin":
            data_transforms = _transforms.Group(
                inputs=[calvin_policy.CalvinInputs(model_config)],
                outputs=[calvin_policy.CalvinOutputs(model_config)],
            )
        elif self.data_type == "agibot":
            data_transforms = _transforms.Group(
                inputs=[agibot_policy.AgibotInputs(model_config)],
                outputs=[agibot_policy.AgibotOutputs(model_config)],
            )
        else:
            assert False, f"Unsupported data type: {self.data_type}"
        if self.delta_transform_type != None:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsByAbs(self.delta_transform_type)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActionsByAbs(self.delta_transform_type)],  # inverse operation of above
            )
        elif self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1) # 【T T T T T T F】
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            hist_objs_config=self.hist_objs_config,
            action_sequence_keys=self.action_sequence_keys,
        )


@register_data_config("lerobot_libero_vae")
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoVaeDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("actions",)
    latent_index: int | Sequence[int] = 0
    extra_delta_transform: bool = True
    hist_objs_config: dict[str, Any] | None = None
    repack_transforms: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.repack_transforms)
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoVaeInputs(model_type=model_config.model_type, latent_index=self.latent_index)],
            outputs=[libero_policy.LiberoVaeOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1) # 【T T T T T T F】
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            hist_objs_config=self.hist_objs_config,
			action_sequence_keys=self.action_sequence_keys,
        )


@register_data_config("pi0_joint_30_state_no_left")
@dataclasses.dataclass(frozen=True)
class Pi0Joint30StateNoLeftDataConfig(DataConfigFactory):
    clip_state: int = 30
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "state",  
                        "actions": "actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.UntNoLeftInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(model_config.action_dim-2, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
        )
 
 
@register_data_config("pi0_ee_30_state_no_left")
@dataclasses.dataclass(frozen=True)
class Pi0EE30StateNoLeftDataConfig(DataConfigFactory):
    clip_state: int = 30
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "ee_state",  
                        "actions": "ee_actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.UntNoLeftInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(model_config.action_dim-2, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
            action_sequence_keys=("ee_actions",),   # explicitly use ee_actions to build action sequences
        )


@register_data_config("pi0_ee_30_state")
@dataclasses.dataclass(frozen=True)
class Pi0EE30StateDataConfig(DataConfigFactory):
    clip_state: int = 30
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "ee_state",  
                        "actions": "ee_actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(model_config.action_dim-2, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
            action_sequence_keys=("ee_actions",),   # explicitly use ee_actions to build action sequences
        )


@register_data_config("pi0_ee_30_state_hand_abs")
@dataclasses.dataclass(frozen=True)
class Pi0EE30StateHandAbsDataConfig(DataConfigFactory):
    clip_state: int = 30
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "ee_state",  
                        "actions": "ee_actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(14, -12, 4, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
            action_sequence_keys=("ee_actions",),   # explicitly use ee_actions to build action sequences
        )


@register_data_config("pi0_ee_42_state")
@dataclasses.dataclass(frozen=True)
class Pi0EE42StateDataConfig(DataConfigFactory):
    clip_state: int = 42
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        assert self.clip_state == model_config.state_dim, f"state_dim {self.clip_state} != model_config.state_dim {model_config.state_dim}"
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "ee_state",  
                        "actions": "ee_actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(model_config.action_dim-2, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
            action_sequence_keys=("ee_actions",),   # explicitly use ee_actions to build action sequences
        )


@register_data_config("pi0_ee_30_state_head")
@dataclasses.dataclass(frozen=True)
class Pi0EE30StateHeadDataConfig(DataConfigFactory):
    clip_state: int = 30
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "top_head",
                        "observation/wrist_image": "hand_left",
                        "observation/right_wrist_image": "hand_right",
                        "observation/state": "ee_state",  
                        "actions": "ee_actions",
                        "prompt": "prompt",
                        "status_bar": "ee_actions_status_bar",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[unt_policy.UntHeadInputs(model_type=model_config.model_type)],
            outputs=[unt_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(model_config.action_dim-2, -2) # last two dimensions are base motion
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],  # convert absolute values to relative values (action-state)
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],  # inverse operation of above
            )
        model_transforms = ModelTransformFactory()(model_config, self)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # norm
            repack_transforms=repack_transform,    # key remapping
            data_transforms=data_transforms,        # input/output transforms
            model_transforms=model_transforms,      # model-related transforms including resize and tokenizer
            action_sequence_keys=("ee_actions",),   # explicitly use ee_actions to build action sequences
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config, self)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config, self)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = "tmp"

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=lambda: get_model_config("pi0")())

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 10000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    gradient_accumulation_steps: int = 1

    deepspeed_config: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / "checkpoints").resolve()  # self.name / self.exp_name

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=get_model_config("pi0")(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=get_model_config("pi0")(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=get_model_config("pi0")(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=get_model_config("pi0")(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=get_model_config("pi0")(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=get_model_config("pi0_fast")(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=get_model_config("pi0")(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=get_model_config("pi0")(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),  # ("s3://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_liberoself",
        model=get_model_config("pi0")(),
        data=LeRobotLiberoDataConfig(
            repo_id="RoboMM/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),  # ("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero90",
        model=get_model_config("pi0")(),
        data=LeRobotLiberoDataConfig(
            repo_id="RoboMM/libero90",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),
        num_train_steps=60_000,
    ),
	TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_unt_30FPS_30StateFrom42_PadResize_NoLeft",
        model=get_model_config("pi0")(state_dim=30),  # state_dim=32
        data=Pi0Joint30StateNoLeftDataConfig(
            repo_id="unt/task_227_pick",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),
        num_train_steps=30_000,
    ),
	TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_ee_30state_randome_magazine",
        model=get_model_config("pi0")(state_dim=30),  # ensure action_dim matches data
        data=Pi0EE30StateNoLeftDataConfig(
            repo_id="unt/task_318",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),  # s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=get_model_config("pi0")(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=get_model_config("pi0")(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=get_model_config("pi0_fast")(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_fast_base/params")),  # "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero90",
        model=get_model_config("pi0_fast")(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="RoboMM/libero90",
            base_config=DataConfig(prompt_from_task=True),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_fast_base/params")),  # "s3://openpi-assets/checkpoints/pi0_fast_base/params"
        num_train_steps=60_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=get_model_config("pi0_fast")(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=get_model_config("pi0_fast")(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=get_model_config("pi0")(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=120,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi05_base/params")),
        pytorch_weight_path=os.path.join(MODEL_ZOO, "physical-intelligence/pytorch_pi05_base/"),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=get_model_config("pi0")(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms={
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            },
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=get_model_config("pi0")(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms={
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            },
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=get_model_config("pi0_fast")(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=get_model_config("pi0")(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=get_model_config("pi0")(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=get_model_config("pi0")(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=get_model_config("pi0")(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=get_model_config("pi0")(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="default",
        model=get_model_config("pi0")(state_dim=30),  # ensure action_dim matches data
        data=Pi0EE30StateNoLeftDataConfig(
            repo_id="unt/task_318",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
			extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(os.path.join(MODEL_ZOO, "physical-intelligence/pi0_base/params")),  # s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def load_config_overrides(config_file: str) -> dict[str, Any]:
    """Load config overrides from YAML file with variable interpolation support.
    
    Supports OmegaConf variable interpolation syntax like ${variable_name}.
    Example:
        num_train_steps: 30_000
        lr_schedule:
            decay_steps: ${num_train_steps}  # Will be resolved to 30_000
    """
    config_path = pathlib.Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_file} not found.")
    
    try:
        # Use OmegaConf to support variable interpolation
        cfg = OmegaConf.load(config_path)
        # Resolve all interpolations and convert to native Python dict
        overrides = OmegaConf.to_container(cfg, resolve=True)
        return overrides
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_file}: {e}")

def apply_config_overrides(base_config: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    """Apply overrides to a base config using only dataclasses.replace."""
    
    def apply_nested_overrides(obj, overrides_dict):
        """Recursively apply overrides to nested dataclass objects."""
        if not isinstance(overrides_dict, dict):
            return overrides_dict
        
        # depth-first: recurse into nested objects before checking current type
        result = obj
        for key, value in overrides_dict.items():
            if hasattr(result, key):
                attr = getattr(result, key)
                if isinstance(value, dict) and hasattr(attr, '__dataclass_fields__'):
                    obj_type = value.pop('type')
                    if isinstance(attr, _model.BaseModelConfig):
                        # Model type selection using registry
                        try:
                            config_class = get_model_config(obj_type)
                            new_obj = config_class(**value)
                            result = dataclasses.replace(result, **{key: new_obj})
                            if 'freeze_filter' in value and value['freeze_filter'] is not None:
                                freeze_filter=new_obj.get_freeze_filter()
                                result = dataclasses.replace(result, freeze_filter=freeze_filter)
                        except (ImportError, ValueError) as e:
                            raise ValueError(f"Unknown model type: '{obj_type}'")
                    elif isinstance(attr, DataConfigFactory):
                        config_class = get_data_config(obj_type)
                        if 'base_config' in value:
                            base_config_dict = value.pop('base_config')
                            value['base_config'] = get_data_config(base_config_dict.pop('type'))(**base_config_dict)
                        new_obj = config_class(**value)
                        result = dataclasses.replace(result, **{key: new_obj})
                    elif isinstance(attr, weight_loaders.WeightLoader):
                        config_class = get_weight_loader(obj_type)
                        new_obj = config_class(**value)
                        result = dataclasses.replace(result, **{key: new_obj})
                    elif _optimizer.is_lr_schedule_config(attr):
                        config_class = get_scheduler(obj_type)
                        new_obj = config_class(**value)
                        result = dataclasses.replace(result, **{key: new_obj})
                    elif _optimizer.is_optimizer_config(attr):
                        config_class = get_optimizer_config(obj_type)
                        new_obj = config_class(**value)
                        result = dataclasses.replace(result, **{key: new_obj})
                    else:
                        raise ValueError(f"Unknown object type: {type(attr)}")
                else:
                    # Simple value
                    result = dataclasses.replace(result, **{key: value})
        
        return result
    
    # Apply overrides recursively
    return apply_nested_overrides(base_config, overrides)

def cli() -> TrainConfig:
    """Enhanced CLI that supports both original mode and YAML config file overrides.
    
    Parameter priority (from highest to lowest):
    1. Command line arguments
    2. YAML file overrides
    3. Default config values
    
    Workflow:
    1. If YAML file is provided:
       - Load YAML overrides
       - Apply YAML overrides to base config
       - Process command line args (which override YAML values)
    2. If no YAML file:
       - Process command line args directly
    """
    # Check if a YAML config file is specified as the first argument
    config_file = None
    config_name_from_file = sys.argv[1]
    
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')):
        config_file = sys.argv[1]
        # Extract config name from filename (without extension)
        config_name_from_file = pathlib.Path(config_file).stem
        # Remove the config file from argv so tyro can process the rest
        sys.argv.pop(1)

        # Check if the config name exists in _CONFIGS_DICT
        if config_name_from_file in _CONFIGS_DICT:
            # Insert the config name derived from filename
            sys.argv.insert(1, config_name_from_file)
        else:
            # If config name doesn't exist, use a default config and log a warning
            logging.warning(f"Config '{config_name_from_file}' not found in available configs. Using 'default' as base config, and use {config_file} as overrides.")
            sys.argv.insert(1, "default")
        # If config name is provided after the file, keep it as is
        # This allows: python train.py config.yaml default
    
        # Apply YAML overrides first (if provided)
    if config_file:
        try:
            # Load YAML overrides
            overrides = load_config_overrides(config_file)
            
            # Temporarily modify sys.argv to only include script name and config name
            # This prevents tyro from processing command line args during YAML application
            original_argv = copy.deepcopy(sys.argv)
            sys.argv = sys.argv[:2]  # Keep only script name and config name
            
            # Get base config and apply YAML overrides
            base_config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
            base_config = apply_config_overrides(base_config, overrides)
            _CONFIGS_DICT[base_config.name] = base_config
            logging.info(f"Applied YAML overrides from {config_file}")
            
            # Restore original argv and let tyro process command line args
            # This ensures command line args have higher priority than YAML
            sys.argv = original_argv

        except Exception as e:
            logging.error(f"Failed to apply config overrides: {e}")
            raise
    
    base_config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
    # Set the config name from filename
    base_config = dataclasses.replace(base_config, name=config_name_from_file)
    logging.info(f"base_config: {base_config}")
    return base_config


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    # if config_name not in _CONFIGS_DICT:
    #     closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
    #     closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
    #     raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    # return _CONFIGS_DICT[config_name]

    sys.argv = ["", config_name]
    return cli()
