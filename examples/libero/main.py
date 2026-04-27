import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import cv2

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = IMAGE_RESOLUTION= 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    # Utils
    video_out_path: str = "tmp/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # validate intrinsics and extrinsics
                    agentview_intrinsics, agentview_extrinsics = get_camera_params_from_env(
                        env, camera_name="agentview", image_size=IMAGE_RESOLUTION
                    )  # intrinsics/extrinsics are flipped because pixel coordinates are flipped
                    # ee_pos = obs["robot0_eef_pos"]
                    # agentview_image = np.ascontiguousarray(np.flip(obs["agentview_image"], axis=(0, 1)))
                    # pixel_coords = project_ee_to_image(ee_pos, agentview_intrinsics, agentview_extrinsics)
                    # # agentview_image = draw_ee_projection_on_image(agentview_image, pixel_coords)
                    # pixel_coords = np.clip(pixel_coords/np.array([agentview_image.shape[1]-1, agentview_image.shape[0]-1]), 0, 1).astype(np.float32)
                    
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        element.update({"intrinsics": agentview_intrinsics,
                            "extrinsics": agentview_extrinsics,
                        })

                        # Query model to get action
                        model_output = client.infer(element)
                        action_chunk = model_output["actions"]
                        # print(model_output)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_camera_params_from_env(env, camera_name="agentview", image_size=256):
    """
    Get camera intrinsic and extrinsic parameters from robosuite environment.
    
    Args:
        env: robosuite environment
        camera_name: name of the camera (default: "agentview")
        image_size: size of the image (default: 256)
    
    Returns:
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (3, 4) camera extrinsic matrix [R|t]
    """
    # Get camera from environment
    camera = env.sim.model.camera(camera_name)
    
    # Get camera position and orientation from MuJoCo
    cam_id = camera.id
    cam_pos = env.sim.data.cam_xpos[cam_id]  # Camera position in world coordinates
    cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3)  # Camera-to-world rotation matrix
    
    # Build extrinsic matrix [R|t] for world-to-camera transformation
    # cam_xmat is camera-to-world rotation, so we need its transpose for world-to-camera
    # MuJoCo camera frame: x-right, y-up, z-backward
    # Standard camera frame: x-right, y-down, z-forward
    # Note: Since images are flipped with np.flip(axis=(0,1)), we need to account for this
    # in the coordinate conversion. The conversion flips all three axes to match the flipped image.
    # Conversion: [x, y, z]_mujoco -> [-x, -y, -z]_standard (to match flipped image)
    R_cam_to_world = cam_mat  # Camera-to-world rotation
    R_world_to_cam_mujoco = R_cam_to_world.T  # World-to-camera rotation (MuJoCo frame)
    
    # Convert from MuJoCo camera frame to standard camera frame
    # Standard frame: x-right, y-down, z-forward
    # All three axes are flipped to account for image flipping in both u and v directions
    R_convert = np.array([
        [-1, 0, 0],  # Flip x (u direction) to match image flip in width
        [0, -1, 0],  # Flip y (v direction) to match image flip in height
        [0, 0, -1]   # Flip z (backward -> forward)
    ], dtype=np.float32)
    R = R_convert @ R_world_to_cam_mujoco  # Apply coordinate conversion
    
    # Translation: t = -R @ cam_pos (in standard camera frame)
    t = -R @ cam_pos
    
    extrinsics = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
    
    # Get intrinsic parameters from MuJoCo camera model
    fovy_deg = env.sim.model.cam_fovy[cam_id]  # Vertical FOV in degrees
    fovy_rad = np.radians(fovy_deg)
    
    # Calculate focal length from vertical FOV
    # focal_length = (image_height / 2) / tan(fovy / 2)
    fy = image_size / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy  # Assume square pixels (aspect ratio = 1)

    # Principal point at image center
    cx = cy = image_size / 2.0
    
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return intrinsics, extrinsics


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
