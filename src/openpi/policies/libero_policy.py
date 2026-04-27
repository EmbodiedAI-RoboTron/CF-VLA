import dataclasses

import einops
import numpy as np
import logging

from torch.nn import intrinsic

from openpi import transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

import openpi.shared.robosuite_transform_utils as trans_utils
import openpi.shared.robosuite_control_utils as contr_utils


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
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif len(image.shape) == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def draw_ee_projection_on_image(image, pixel_coords, color=(0, 255, 0), radius=5, thickness=2):
    """
    Draw end-effector projection on image.
    
    Args:
        image: (H, W, 3) image array
        pixel_coords: (2,) pixel coordinates [u, v]
        color: RGB color tuple (default: green)
        radius: circle radius (default: 5)
        thickness: circle thickness (default: 2)
    
    Returns:
        image: image with projection drawn
    """
    if pixel_coords is None:
        return image
    
    u, v = int(pixel_coords[0]), int(pixel_coords[1])
    h, w = image.shape[:2]
    
    # Check if coordinates are within image bounds
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(image, (u, v), radius, color, thickness)
        # Draw a cross for better visibility
        cv2.line(image, (u - radius, v), (u + radius, v), color, thickness)
        cv2.line(image, (u, v - radius), (u, v + radius), color, thickness)
    
    return image


def project_ee_to_image(ee_pos, intrinsics, extrinsics):
    """
    Project end-effector 3D position to 2D image coordinates.
    
    Args:
        ee_pos: (3,) end-effector position in world coordinates
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (3, 4) camera extrinsic matrix [R|t]
    
    Returns:
        pixel_coords: (2,) pixel coordinates [u, v], or None if behind camera
    """
    # Convert to homogeneous coordinates
    ee_pos_h = np.append(ee_pos, 1.0)
    
    # Combine intrinsics and extrinsics into a single projection matrix (3x4)
    projection_matrix = intrinsics @ extrinsics  # (3, 3) @ (3, 4) = (3, 4)
    
    # Project directly to image plane
    pixel_coords_h = projection_matrix @ ee_pos_h  # (3, 4) @ (4, 1) = (3,)
    
    # Check if point is behind camera (w < 0, which corresponds to z < 0)
    if pixel_coords_h[2] <= 0:
        return None
    
    # Normalize by depth (homogeneous coordinate)
    pixel_coords = pixel_coords_h[:2] / pixel_coords_h[2]  # (2,)
    
    return pixel_coords


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    # model_type: _model.ModelType
    model_config: _model.BaseModelConfig

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
        state = data.pop("observation/state")
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_config.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        # if "actions" in data:
        #     inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]

        inputs.setdefault("extra_data", {})
        # if 'intrinsics' in data:
        #     ee_pos = np.array(state[:3])
        #     agentview_intrinsics, agentview_extrinsics = np.array(data.pop("intrinsics")), np.array(data.pop("extrinsics"))
        #     pixel_coords = project_ee_to_image(ee_pos, agentview_intrinsics, agentview_extrinsics)
        #     pixel_coords = np.clip(pixel_coords/np.array([base_image.shape[1]-1, base_image.shape[0]-1]), 0, 1).astype(np.float32)
        #     inputs["extra_data"]["ee_pixel_coords"] = pixel_coords
        #     # agentview_image = np.ascontiguousarray(base_image.copy())
        #     # agentview_image = draw_ee_projection_on_image(agentview_image, pixel_coords)
        if "terminal" in data:
            inputs["extra_data"]["terminal"] = data.pop("terminal")
        if self.model_config.model_kwargs.get("use_ref", False) and 'ref' in data:
        #     if self.model_config.model_kwargs.get("ref_type", "actions") == "goals":
        #         ee_pos = state[:3]
        #         ee_ori_mat = trans_utils.quat2mat(trans_utils.axisangle2quat(state[3:6]))
        #         # scaled_delta = data['actions'][0, :6] * np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        #         # goal_ori = trans.set_goal_orientation(scaled_delta[3:], ee_ori_mat)
        #         # goal_pos = trans.set_goal_position(scaled_delta[:3], ee_pos)
        #         goals = data.pop("goals")
        #         refs = []
        #         for t in range(len(goals)):  # compute delta actions
        #             goal_pos, goal_ori = goals[t, ..., :3], goals[t, ..., 3:12].reshape(3, 3)
        #             delta_pos = goal_pos - ee_pos
        #             delta_ori = contr_utils.orientation_error(goal_ori, ee_ori_mat)
        #             refs.append(np.concatenate([delta_pos, delta_ori, goals[t,..., -1:]]))
        #         inputs["extra_data"]["ref"] = np.stack(refs, axis=0)
        #     else:
            inputs["extra_data"]["ref"] = data.pop("ref")  # ref is mainly used for testing
        if self.model_config.model_kwargs.get("use_goals_as_actions", False) and 'goals' in data:
            goals = data.pop("goals")
            # goal_poses, goal_ories = goals[..., :3], goals[..., 3:12].reshape(-1, 3, 3)
            # if self.model_config.model_kwargs.get("goals_type", '') == 'mat2euler':
            #     goal_ori = np.stack([trans_utils.quat2axisangle(trans_utils.mat2quat(goal_ories[t])) for t in range(len(goal_ories))])
            #     actions = np.concatenate([goal_poses, goal_ori, goals[..., -1:]], axis=-1)
            # elif self.model_config.model_kwargs.get("goals_type", '') == 'mat2quat':
            #     goal_ori = np.stack([trans_utils.mat2quat(goal_ories[t]) for t in range(len(goal_ories))])
            #     actions = np.concatenate([goal_poses, goal_ori, goals[..., -1:]], axis=-1)
            # else:
            #     actions = np.concatenate([goal_poses, goal_ories.reshape(-1, 9), goals[..., -1:]], axis=-1)
            data["actions"] = goals
        if self.model_config.model_kwargs.get("use_goals_as_delta_actions", False) and 'goals' in data:
            goals = data.pop("goals")
            actions = []
            delta_pos = goals[..., :3] - state[..., :3]
            delta_ori = np.stack([contr_utils.orientation_error(goals[t, ..., 3:12].reshape(3, 3), trans_utils.quat2mat(trans_utils.axisangle2quat(state[t, 3:6]))) for t in range(len(goals))], axis=0)
            data['actions'] = np.concatenate([delta_pos, delta_ori, goals[..., -1:]], axis=-1)
            data['state'] = state[0]  # take only the first state
        if self.model_config.model_kwargs.get("q_function", False) and 'indices' in data:
            idx, ep_start, ep_end, terminal = data.pop("indices")  # ep_end is exclusive for the current episode
            reward = -1.0 * (ep_end-1 - idx) if terminal > 0 else -1.0 * (ep_end-1 - idx) - self.model_config.model_kwargs.get("c_fail", -1)
            inputs["extra_data"]["reward"] = reward / (ep_end - ep_start)
        if self.model_config.model_kwargs.get("use_depth", False) and 'depth' in data:
            base_image_depth = data.pop("depth")
            wrist_image_depth = data.pop("wrist_depth")
            inputs["extra_data"]["base_0_rgb_depth"] = base_image_depth
            inputs["extra_data"]["left_wrist_0_rgb_depth"] = wrist_image_depth
            inputs["extra_data"]["right_wrist_0_rgb_depth"] = np.zeros_like(base_image_depth)
        if self.model_config.model_kwargs.get("use_cam", False) and 'intrinsics' in data:
            intrinsics, extrinsics = data.pop("intrinsics"), data.pop("extrinsics")
            inputs['extra_data']['base_0_rgb_intrinsics'] = intrinsics
            inputs['extra_data']['base_0_rgb_extrinsics'] = extrinsics
            inputs['extra_data']['left_wrist_0_rgb_intrinsics'] = data.pop("wrist_intrinsics")
            inputs['extra_data']['left_wrist_0_rgb_extrinsics'] = data.pop("wrist_extrinsics")
            inputs['extra_data']['right_wrist_0_rgb_intrinsics'] = np.zeros_like(intrinsics)
            inputs['extra_data']['right_wrist_0_rgb_extrinsics'] = np.zeros_like(extrinsics)

        inputs.update(**data)

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoVaeInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    latent_index: int = 0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        inputs["extra_data"] = {}
        if "latent" in data:
            latent_index = self.latent_index
            if isinstance(self.latent_index, int):
                latent_index = [self.latent_index]
            if len(data["latent"].shape) == 3:
                inputs["extra_data"]["latent"] = einops.rearrange(data["latent"], "(n c) h w -> (n h w) c", c=4)
            else:
                inputs["extra_data"]["latent"] = einops.rearrange(data["latent"][latent_index], "b (n c) h w -> (b n h w) c", c=4)
            inputs["extra_data"]['latent_index'] = np.array(latent_index)
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    model_config: _model.BaseModelConfig

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        if self.model_config.model_kwargs.get("use_goals_as_actions", False):
            # ee_pos, ee_ori = data['state'][:3], data['state'][3:6]
            # ee_ori_mat = trans_utils.quat2mat(trans_utils.axisangle2quat(ee_ori))
            # actions = []
            # if self.model_config.model_kwargs.get("goals_type", '') == 'mat2euler':
            #     goal_poses, goal_ories = data["actions"][..., :3], data["actions"][..., 3:6]
            #     for t in range(len(goal_ories)):
            #         goal_pos = goal_poses[t]
            #         goal_ori = goal_ories[t]
            #         goal_ori_mat = trans_utils.quat2mat(trans_utils.axisangle2quat(goal_ori))
            #         actions.append(np.concatenate([goal_pos, goal_ori_mat.reshape(-1), data["actions"][t, 6:]], axis=-1))
            # elif self.model_config.model_kwargs.get("goals_type", '') == 'mat2quat':
            #     goal_poses, goal_ories = data["actions"][..., :3], data["actions"][..., 3:7]
            #     for t in range(len(goal_ories)):
            #         goal_pos = goal_poses[t]
            #         goal_ori = goal_ories[t]
            #         goal_ori_mat = trans_utils.quat2mat(goal_ori)
            #         actions.append(np.concatenate([goal_pos, goal_ori_mat.reshape(-1), data["actions"][t, 7:]], axis=-1))
            # else:
            #     goal_poses, goal_ories = data["actions"][..., :3], data["actions"][..., 3:12]
            #     for t in range(len(goal_ories)):
            #         goal_pos = goal_poses[t]
            #         goal_ori_mat = goal_ories[t].reshape(3, 3)
            #         actions.append(np.concatenate([goal_pos, goal_ori_mat.reshape(-1), data["actions"][t, 12:]], axis=-1))
            # # actions = np.stack(actions, axis=0) / np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1])
            # data["actions"] = np.stack(actions, axis=0)
            return {"actions": np.asarray(data["actions"][:, :13])}
        elif self.model_config.model_kwargs.get("use_goals_as_delta_actions", False):
            actions = data["actions"][:, :7] / np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1])
            return {"actions": np.asarray(actions)}
        else:
            return {"actions": np.asarray(data["actions"][..., :7])}

@dataclasses.dataclass(frozen=True)
class LiberoVaeOutputs(transforms.DataTransformFn):
    decoder = _tokenizer.DiffuserDecodeLatent()
    def __call__(self, data: dict) -> dict:
        outputs = {"actions": np.asarray(data["actions"][:, :7])}

        if "latent" in data:
            z = einops.rearrange(data["latent"], "(n h w) c -> n c h w", c=4, h=14, w=14)
            for i in range(z.shape[0]):
                images = self.decoder.decode(z[i], save_path=f"tmp_{i}.png")
            outputs["images"] = images
        return outputs
        