import logging
import os
import random
import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor, AutoTokenizer
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable
import diffusers
from diffusers import AutoencoderKL
import torch
import torchvision
import einops

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download(os.getenv("MODEL_ZOO") + "/physical-intelligence/paligemma_tokenizer.model", gs={"token": "anon"})  #"gs://big_vision/paligemma_tokenizer.model"
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None, data: dict = None) -> tuple[np.ndarray, np.ndarray, dict]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask), data

    def tokenize_actions(self, actions: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        discretized_actions = np.digitize(actions, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
  
        tokens_all = []
        mask_all = []
        for action in discretized_actions:
            actions_str = " ".join(map(str, action))
            tokens = self._tokenizer.encode(actions_str, add_bos=True)
            
            tokens_len = len(tokens)
            if tokens_len < self._max_len:
                padding = [False] * (self._max_len - tokens_len)
                mask = [True] * tokens_len + padding
                tokens = tokens + padding
            else:
                if len(tokens) > self._max_len:
                    logging.warning(
                        f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                        "Consider increasing the `max_token_len` in your model config if this happens frequently."
                    )
                tokens = tokens[: self._max_len]
                mask = [True] * self._max_len
            tokens_all.append(tokens)
            mask_all.append(mask)
            
        return np.stack(tokens_all, axis=0), np.stack(mask_all, axis=0)

class VlaGemmaTokenizer:
    """
    Tokenizer for VLA Gemma model using PaliGemma2-3B-PT-448.
    This tokenizer handles text prompts and optionally state information for VLA tasks.
    """
    
    def __init__(self, max_len: int = 48, model_id: str = "google/paligemma2-3b-pt-448", tokenizer_kwargs: dict = {}):
        self._max_len = max_len
        self._model_id = model_id
        self._tokenizer_kwargs = tokenizer_kwargs
        # Load the PaliGemma2 processor which includes both tokenizer and image processor
        try:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._tokenizer = self._processor.tokenizer
        except Exception as e:
            logging.warning(f"loading processor failed, using AutoTokenizer instead: {e}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Special tokens for VLA tasks
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"
        self._bos_token_id = self._tokenizer.bos_token_id
        self._eos_token_id = self._tokenizer.eos_token_id
        self._pad_token_id = self._tokenizer.pad_token_id
        self._2dn_token = "\n\n"  # two newline characters
        self._2dn_token_id = self._tokenizer.convert_tokens_to_ids(self._2dn_token)
        self._dn_token = "\n"  # one newline character
        self._dn_token_id = self._tokenizer.convert_tokens_to_ids(self._dn_token)
        self._end_token = "<end_of_turn>"
        self._end_token_id = self._tokenizer.convert_tokens_to_ids(self._end_token)
        
        logging.info(f"Initialized VlaGemmaTokenizer with model_id: {model_id}")
        logging.info(f"Vocab size: {self._tokenizer.vocab_size}")
        logging.info(f"Max length: {max_len}")

    def tokenize(self, prompt: str, state: np.ndarray | None = None, data: dict = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Tokenize a text prompt and optionally state information.
        
        Args:
            prompt: The text prompt to tokenize
            state: Optional state array for Pi05-style models
            
        Returns:
            Tuple of (tokens, attention_mask) as numpy arrays
        """
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        max_length = self._max_len
        if state is not None:
            # Pi05 format: state is part of the discrete language input
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        elif self._tokenizer_kwargs.get("class_free_language", False) and 'extra_data' in data and 'terminal' in data['extra_data']:
            terminal = data['extra_data']['terminal']
            if random.random() < self._tokenizer_kwargs.get("free_drop_probability", 0.0):
                full_prompt = f"{self._bos_token}{cleaned_text}. \n"
            else:
                flag = "Negative" if terminal == 0 else "Positive"
                full_prompt = f"{self._bos_token}{flag}, {cleaned_text}. \n"
        elif self._tokenizer_kwargs.get("pixel_coords_vqa", False) and 'extra_data' in data and 'ee_pixel_coords' in data['extra_data']:
            ee_pixel_coords = data['extra_data'].pop('ee_pixel_coords')
            if random.random() < self._tokenizer_kwargs.get("free_drop_probability", 1.0):
                discretized_state = np.digitize(ee_pixel_coords, bins=np.linspace(0, 1, 1001)[:-1]) - 1
                pixel_str = " ".join(f"{x:03d}" for x in discretized_state)
                full_prompt = f"{self._bos_token}{cleaned_text}. \n" + f"Gripper coordinate:{pixel_str}.{self._end_token}"  # .->236761; \n->107; \n\n->108; :->236787; space->236743
                data['extra_data']['gripper_coordinate_split_tokens'] = np.array([self._dn_token_id, self._end_token_id])
            else:
                full_prompt = f"{self._bos_token}{cleaned_text}. \n"
                data['extra_data']['gripper_coordinate_split_tokens'] = np.array([self._dn_token_id, self._dn_token_id])
            max_length = max_length + 10
        else:
            full_prompt = f"{self._bos_token}{cleaned_text}. \n"
        
        # Tokenize using the PaliGemma2 tokenizer
        encoded = self._tokenizer(
            full_prompt,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
            # padding_side = "right"
        )
        
        tokens = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0].astype(bool)  # Paligemma2 padding appears on the left
        # Log warning if truncation occurred
        if len(self._tokenizer.encode(full_prompt, add_special_tokens=True)) > max_length:
            logging.warning(
                f"Token length exceeds max length ({max_length}), truncating. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently."
            )
        
        if self._tokenizer_kwargs.get("pixel_coords_split_vqa", False) and 'extra_data' in data and 'ee_pixel_coords' in data['extra_data']:
            ee_pixel_coords = data['extra_data'].pop('ee_pixel_coords')
            discretized_state = np.digitize(ee_pixel_coords, bins=np.linspace(0, 1, 1001)[:-1]) - 1
            pixel_str = " ".join(f"{x:03d}" for x in discretized_state)
            vqa_prompt = f"Gripper coordinate:{pixel_str}.{self._end_token}"  # .->236761; \n->107; \n\n->108; :->236787; space->236743
            # Tokenize using the PaliGemma2 tokenizer
            vqa_encoded = self._tokenizer(
                vqa_prompt,
                add_special_tokens=True,
                max_length=20,
                padding="max_length",
                truncation=True,
                return_tensors="np",
                padding_side = "right"
            )
            
            tokens = vqa_encoded['input_ids'][0]
            attention_mask = vqa_encoded['attention_mask'][0].astype(bool)  # Paligemma2 padding appears on the left
            # Log warning if truncation occurred
            if len(self._tokenizer.encode(vqa_prompt, add_special_tokens=True)) > 20:
                logging.warning(
                    f"Token length exceeds max length (20), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            data['extra_data']['vqa_tokens'] = tokens
            data['extra_data']['vqa_attention_mask'] = attention_mask
        
        return tokens, attention_mask, data

    def tokenize_batch(self, prompts: list[str], states: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Tokenize a batch of prompts and states.
        
        Args:
            prompts: List of text prompts
            states: Optional list of state arrays
            
        Returns:
            Tuple of (batch_tokens, batch_attention_masks) as numpy arrays
        """
        if states is None:
            states = [None] * len(prompts)
        
        batch_tokens = []
        batch_masks = []
        
        for prompt, state in zip(prompts, states):
            tokens, mask = self.tokenize(prompt, state)
            batch_tokens.append(tokens)
            batch_masks.append(mask)
        
        return np.stack(batch_tokens), np.stack(batch_masks)

    def decode(self, tokens: np.ndarray, skip_special_tokens: bool = True) -> str:
        """
        Decode tokens back to text.
        
        Args:
            tokens: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        # Convert numpy array to list and filter out padding tokens
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        # Remove padding tokens
        if self._pad_token_id is not None:
            tokens = [t for t in tokens if t != self._pad_token_id]
        
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self._tokenizer.vocab_size

    def get_special_tokens(self) -> dict[str, int]:
        """Get special token IDs."""
        return {
            "bos_token_id": self._bos_token_id,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
        }

    def process_images_and_text(self, images: list, text: str) -> dict:
        """
        Process images and text together using the PaliGemma2 processor.
        This is useful for multimodal inputs.
        
        Args:
            images: List of PIL Images or numpy arrays
            text: Text prompt
            
        Returns:
            Dictionary with processed inputs ready for the model
        """
        return self._processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_len
        )


class VlaQwenVLTokenizer:
    """
    Unified tokenizer for VLA Qwen-VL models (Qwen2.5-VL and Qwen3-VL).
    Supports text + multi-image input using apply_chat_template.
    Mirrors VlaGemmaTokenizer interface for seamless swapping.
    """

    def __init__(self, max_len: int = 48, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", lowercase: bool = False, train: bool = False):
        import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

        self._max_len = max_len
        self._model_id = model_id
        self._lowercase = lowercase
        self._image_keys  = _preprocessing.IMAGE_KEYS
        self._train = train

        # Use HF processor for Qwen-VL (handles both tokenizer and image processor)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._tokenizer = self._processor.tokenizer
        self._image_token_id = self._processor.image_token_id
        # self._vision_start_token_id = self._processor.vision_start_token_id
        # self._vision_end_token_id = self._processor.vision_end_token_id
        self._pad_token_id = self._tokenizer.pad_token_id

        logging.info(f"Initialized VlaQwenVLTokenizer with model_id: {model_id}")
        logging.info(f"Vocab size: {self._tokenizer.vocab_size}")
        logging.info(f"Max length: {max_len}")
        logging.info(f"Lowercase: {lowercase}")

    def tokenize(
        self, 
        prompt: str, 
        state: np.ndarray | None = None,
        data: dict = None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Tokenize text prompt with optional images using apply_chat_template.
        
        Args:
            prompt: Text prompt
            state: Optional robot state (for backward compatibility)
            images: Optional dict of images with keys like "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
                   Each image should be np.ndarray of shape [H, W, 3] with values in [0, 255] uint8
        
        Returns:
            tokens: Token IDs of shape [max_len]
            attention_mask: Attention mask of shape [max_len]
        """
        # Apply text preprocessing based on configuration
        if self._lowercase:
            cleaned_text = prompt.lower().strip().replace("_", " ")
        else:
            cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        else:
            full_prompt = cleaned_text + "\n"

        # Build conversation format for apply_chat_template
        content = []
        
        # Add images if provided (in order: base, left_wrist, right_wrist)
        if False and 'image' in data:
            for key in self._image_keys:
                image = data['image'][key]
                image = torch.from_numpy(image).float()[None]/255.0
                if self._train:
                    # print(f"Applying augmentations to image {key}")
                    # Apply PyTorch-based augmentations
                    if "wrist" not in key:
                        # Geometric augmentations for non-wrist cameras
                        height, width = image.shape[1:3]

                        # Random crop and resize
                        crop_height = int(height * 0.95)
                        crop_width = int(width * 0.95)

                        # Random crop
                        max_h = height - crop_height
                        max_w = width - crop_width
                        if max_h > 0 and max_w > 0:
                            # Use tensor operations instead of .item() for torch.compile compatibility
                            start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                            start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                            image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                        # Resize back to original size
                        image = torch.nn.functional.interpolate(
                            image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                        # Random rotation (small angles)
                        # Use tensor operations instead of .item() for torch.compile compatibility
                        angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                        if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                            # Convert to radians
                            angle_rad = angle * torch.pi / 180.0

                            # Create rotation matrix
                            cos_a = torch.cos(angle_rad)
                            sin_a = torch.sin(angle_rad)

                            # Apply rotation using grid_sample
                            grid_x = torch.linspace(-1, 1, width, device=image.device)
                            grid_y = torch.linspace(-1, 1, height, device=image.device)

                            # Create meshgrid
                            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                            # Expand to batch dimension
                            grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                            grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                            # Apply rotation transformation
                            grid_x_rot = grid_x * cos_a - grid_y * sin_a
                            grid_y_rot = grid_x * sin_a + grid_y * cos_a

                            # Stack and reshape for grid_sample
                            grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                            image = torch.nn.functional.grid_sample(
                                image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                                grid,
                                mode="bilinear",
                                padding_mode="zeros",
                                align_corners=False,
                            ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                    # Color augmentations for all cameras
                    # Random brightness
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
                    image = image * brightness_factor

                    # Random contrast
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
                    mean = image.mean(dim=[1, 2, 3], keepdim=True)
                    image = (image - mean) * contrast_factor + mean

                    # Random saturation (convert to HSV, modify S, convert back)
                    # For simplicity, we'll just apply a random scaling to the color channels
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
                    gray = image.mean(dim=-1, keepdim=True)
                    image = gray + (image - gray) * saturation_factor

                    # Clamp values to [0, 1]
                    image = torch.clamp(image, 0, 1)
                
                image = image[0]
                content.append({"type": "image", "image": image})
        
        # Add text prompt
        content.append({"type": "text", "text": full_prompt})
        
        # Build conversation in the format expected by apply_chat_template
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]    
        encoded = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
        )
        
        tokens = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        image_tokens_num = (tokens == self._image_token_id).sum()
        max_len = self._max_len + image_tokens_num
        if len(tokens) < max_len:
            tokens = torch.cat([tokens, torch.full((max_len - len(tokens),), self._pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.full((max_len - len(attention_mask),), False, dtype=torch.bool)])
        elif len(tokens) > max_len:
            assert False, "Token length exceeds max length"
        
        # start_positions = (tokens == self._vision_start_token_id).nonzero(as_tuple=True)[0]
        # end_positions = (tokens == self._vision_end_token_id).nonzero(as_tuple=True)[0]
        # data['extra_data'] = {
        #     'image_grid_thw': encoded['image_grid_thw'],
        #     'pixel_values': encoded['pixel_values'],
        # }
        # for i, key in enumerate(self._image_keys):
        #     attention_mask[start_positions[i]+1:end_positions[i]] = torch.tensor(data['image_mask'][key], dtype=torch.bool)

        if 'extra_data' not in data:
            data['extra_data'] = {}
        data['extra_data']['prompt'] = prompt

        return tokens, attention_mask, data

    def get_vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def get_special_tokens(self) -> dict[str, int]:
        return {
            "image_token_id": self._image_token_id,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
        }


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = os.getenv("MODEL_ZOO") + "/physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download(os.getenv("MODEL_ZOO") + "/physical-intelligence/paligemma_tokenizer.model", gs={"token": "anon"})  # text token
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download(os.getenv("MODEL_ZOO") + "/physical-intelligence/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download(os.getenv("MODEL_ZOO") + "/physical-intelligence/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

class RoboFlamingoTokenizer:
    def __init__(self, max_len: int = 48):
        import os, sys
        OPENFLAMINGO_PATH = os.getenv("OPENFLAMINGO_PATH", "third_party/open_flamingo")
        MODEL_ZOO = os.getenv("MODEL_ZOO", "checkpoints")
        sys.path.append(OPENFLAMINGO_PATH)
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        self._max_len = max_len
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly"
        use_local_files=False
        cache_dir=MODEL_ZOO  # Defaults to ~/.cache

        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=use_local_files, cache_dir=cache_dir,
        )
        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self._tokenizer = text_tokenizer
        print("YF: RoboFlamingoTokenizer need keep the same with RoboFlamingo model(at init)")


    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        self._tokenizer.padding_side = "right"
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{cleaned_text.strip()}<|endofchunk|>{self._tokenizer.eos_token}")
        ]
        tokens = self._tokenizer(
            sample,
            max_length=self._max_len,
            padding="max_length",
            # truncation="only_first",
            return_tensors="pt",
        )
        return np.asarray(tokens['input_ids'][0]), np.asarray(tokens['attention_mask'][0], dtype=bool)


class DiffuserDecodeLatent:
    def __init__(self):
        device = "cpu"
        checkpoint_dir = os.getenv("AUTOENCODERKL_CHECKPOINT_DIR", "./autoencoderkl-model/checkpoint-7500")
        resolution = 112

        self._vae = AutoencoderKL.from_pretrained(os.path.join(checkpoint_dir, "autoencoderkl"))
        self._vae = self._vae.to(device)
        self._vae.eval()
        self._inference_ctx = torch.autocast(device)

    def decode(self, tokens: np.ndarray, save_path: str = "tmp.png") -> np.ndarray:

        with self._inference_ctx:
            z = tokens.astype(np.float32)
            z = torch.from_numpy(z).to(self._vae.device)[None]
            reconstructions = self._vae.decode(z).sample
            grid = torchvision.utils.make_grid(reconstructions, nrow=1, padding=10, normalize=True)
            torchvision.utils.save_image(grid, save_path)

        return grid.to(torch.float32).cpu().numpy()
