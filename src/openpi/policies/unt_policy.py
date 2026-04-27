import dataclasses

import einops, torch
import numpy as np
import logging

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data.pop("observation/image"))
        wrist_image = _parse_image(data.pop("observation/wrist_image"))
        right_wrist_image = _parse_image(data.pop("observation/right_wrist_image"))
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data.pop("observation/state"),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "status_bar" in data:
            inputs["extra_data"] = {"status_bar": data.pop("status_bar")}

        inputs.update(**data)
        
        # # Pad actions to the model action dimension. Keep this for your own dataset.
        # # Actions are only available during training.
        # if "actions" in data:
        #     inputs["actions"] = data.pop("actions")

        # # Pass the prompt (aka language instruction) to the model.
        # # Keep this for your own dataset (but modify the key if the instruction is not
        # # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data.pop("prompt")

        return inputs


@dataclasses.dataclass(frozen=True)
class UntHeadInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data.pop("observation/image"))
        wrist_image = _parse_image(data.pop("observation/wrist_image"))
        right_wrist_image = _parse_image(data.pop("observation/right_wrist_image"))

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data.pop("observation/state"),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "status_bar" in data:
            inputs["extra_data"] = {"status_bar": data.pop("status_bar")}
        inputs.update(**data)

        # # Pad actions to the model action dimension. Keep this for your own dataset.
        # # Actions are only available during training.
        # if "actions" in data:
        #     inputs["actions"] = data["actions"]

        # # Pass the prompt (aka language instruction) to the model.
        # # Keep this for your own dataset (but modify the key if the instruction is not
        # # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UntNoLeftInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data.pop("observation/image"))
        wrist_image = _parse_image(data.pop("observation/wrist_image"))
        right_wrist_image = _parse_image(data.pop("observation/right_wrist_image"))
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data.pop("observation/state"),
            "image": {
                "base_0_rgb": base_image,
                # "left_wrist_0_rgb": wrist_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # "left_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "status_bar" in data:
            inputs["extra_data"] = {"status_bar": data.pop("status_bar")}
        inputs.update(**data)
        # # Pad actions to the model action dimension. Keep this for your own dataset.
        # # Actions are only available during training.
        # if "actions" in data:
        #     inputs["actions"] = data["actions"]

        # # Pass the prompt (aka language instruction) to the model.
        # # Keep this for your own dataset (but modify the key if the instruction is not
        # # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class Unt2DofNoLeftInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = data.pop("observation/state")
        if isinstance(state, torch.Tensor):  # dataloader uses torch tensor
            state_open = torch.ones([*state.shape[:-1], 2], dtype=state.dtype)
            state_open[..., 0] = state[...,14] < 3.63  # whether left gripper is closed; 1 means closed
            state_open[..., 1] = state[...,20] < 3.63  # whether right gripper is closed; 1 means closed
            state = torch.concat([state[:14], state_open, state[26:]], dim=-1)
        elif isinstance(state, np.ndarray):  # inference uses numpy arrays
            state_open = np.ones([*state.shape[:-1], 2], dtype=state.dtype)
            state_open[..., 0] = state[...,14] < 3.63  # whether left gripper is closed; 1 means closed
            state_open[..., 1] = state[...,20] < 3.63  # whether right gripper is closed; 1 means closed
            state = np.concatenate([state[:14], state_open, state[26:]], axis=-1)
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data.pop("observation/image"))
        wrist_image = _parse_image(data.pop("observation/wrist_image"))
        right_wrist_image = _parse_image(data.pop("observation/right_wrist_image"))
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # "left_wrist_0_rgb": wrist_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # "left_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            actions = data.pop("actions")
            if isinstance(state, torch.Tensor):
                actions_open = torch.ones([*actions.shape[:-1], 2], dtype=actions.dtype)
                actions_open[..., 0] = actions[...,14] < 3.63  # whether left gripper is closed; 1 means closed
                actions_open[..., 1] = actions[...,20] < 3.63  # whether right gripper is closed; 1 means closed
                actions = torch.concat([actions[..., :14], actions_open, actions[..., 26:]], dim=-1)
            elif isinstance(state, np.ndarray):
                actions_open = np.ones([*actions.shape[:-1], 2], dtype=actions.dtype)
                actions_open[..., 0] = actions[...,14] < 3.63  # whether left gripper is closed; 1 means closed
                actions_open[..., 1] = actions[...,20] < 3.63  # whether right gripper is closed; 1 means closed
                actions = np.concatenate([actions[..., :14], actions_open, actions[..., 26:]], axis=-1)
            inputs["actions"] = actions
        if "status_bar" in data:
            inputs["extra_data"] = {"status_bar": data.pop("status_bar")}
        inputs.update(**data)

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data.pop("actions")[..., :32])
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class Unt2DofOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, :22])
        hand_open = np.array([[3.6652, 0.1221, 0.1221, 0.1221, 0.1221, 1.2217],                        # open state
                              [1.6929, 3.4208, 0.4208, 0.4208, 0.4208, 1.2217]], dtype=actions.dtype)  # closed state
        mask = (actions[..., 14] < 0.5)  # value 1 indicates closed
        left = np.where(mask[..., None], hand_open[0], hand_open[1])
        mask = (actions[..., 15] < 0.5)
        right = np.where(mask[..., None], hand_open[0], hand_open[1])
        # keep only rows where mask is True
        actions = np.concatenate([actions[..., :14], left, right, actions[..., 16:]], axis=-1)
        return {"actions": actions}
    