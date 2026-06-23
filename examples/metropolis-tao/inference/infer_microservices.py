#!/usr/bin/env python3
"""
TAO Cosmos Model Server - Cosmos-Reason1-specific implementation of TAO Model Server
Inherits from BaseInferenceMicroserviceServer and implements Cosmos-specific functionality
"""

import sys
from typing import Dict, List, Any, Tuple
from pathlib import Path
import cv2
import argparse
import ast
import logging
import shutil

# Import the base class from tao-core
from cosmos_rl.utils.lora_utils import merge_lora_model
from nvidia_tao_core.microservices.handlers.base_inference_microservice_server import BaseInferenceMicroserviceServer

# Cosmos-specific imports
import qwen_vl_utils
import transformers
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmosInferenceMicroservice(BaseInferenceMicroserviceServer):
    """Cosmos-Reason1-specific implementation of TAO Model Server"""

    def __init__(self, job_id: str, port: int = 8080, cloud_storage=None, **model_params):
        """Initialize Cosmos model server"""
        super().__init__(job_id, port, cloud_storage, **model_params)
        self.model_state_dir = "/tmp/cosmos_models"  # Keep compatibility with existing code
        self.processor = None

    def get_supported_file_extensions(self) -> Tuple[List[str], List[str]]:
        """Get supported file extensions for Cosmos model

        Returns:
            Tuple of (image_extensions, video_extensions)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov']
        return image_extensions, video_extensions

    def load_model_into_memory(self, **kwargs) -> bool:
        """Load Cosmos-Reason1 model implementation

        Args:
            **kwargs: Model-specific configuration parameters (model_name, torch_dtype, device_map, etc.)

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Extract Cosmos-specific parameters
            model_name = kwargs.get('model_path')
            torch_dtype = kwargs.get('torch_dtype', 'auto')
            device_map = kwargs.get('device_map', 'auto')
            enable_lora = kwargs.get('enable_lora', False)
            base_model_path = kwargs.get('base_model_path')

            print(f"Loading Cosmos model: {model_name}")

            # Handle LoRA merging if enabled
            adapter_backup_path = None  # Initialize for later cleanup
            if enable_lora:
                model_name = merge_lora_model(model_name, base_model_path)
                print(f"LoRA merging enabled. Using merged model: {model_name}")
            else:
                # If LoRA is disabled but adapter files exist, temporarily hide them during loading
                adapter_config_path = Path(model_name) / "adapter_config.json"
                if adapter_config_path.exists():
                    adapter_backup_path = Path(model_name) / "adapter_config.json.bak"
                    shutil.move(str(adapter_config_path), str(adapter_backup_path))
                    print("Temporarily hiding adapter files to load base model only")

            # Load Cosmos-Reason1 model
            try:
                self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map
                )

                # Load processor
                self.processor = transformers.AutoProcessor.from_pretrained(model_name)

                print(f"Successfully loaded Cosmos model: {model_name}")
            finally:
                # Restore adapter config if it was temporarily hidden
                if not enable_lora and adapter_backup_path and adapter_backup_path.exists():
                    adapter_config_path = Path(model_name) / "adapter_config.json"
                    shutil.move(str(adapter_backup_path), str(adapter_config_path))
                    print("Restored adapter configuration files")

            return True

        except Exception as e:
            logger.error(f"Failed to load Cosmos model: {e}")
            return False

    def process_media_files(self, input_files: List[str], fps: int = 4, total_pixels: int = 6422528) -> Tuple[List[Dict], List[float]]:
        """Process media files for Cosmos inference

        Args:
            input_files: List of input file paths
            fps: Frame rate for video processing (default: 4)
            total_pixels: Total pixels for video processing (default: 6422528)

        Returns:
            Tuple of (conversation_content, video_durations)
        """
        conversation_content = []
        video_durations = []

        image_exts, video_exts = self.get_supported_file_extensions()

        for input_file in input_files:
            try:
                # Download file if needed
                actual_file_path = self.download_and_process_file(input_file)
                file_ext = Path(actual_file_path).suffix.lower()

                if file_ext in image_exts:
                    conversation_content.append({
                        "type": "image",
                        "image": actual_file_path
                    })
                    video_durations.append(None)

                elif file_ext in video_exts:
                    # Get video duration
                    cap = cv2.VideoCapture(actual_file_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = frame_count / video_fps if video_fps > 0 else 0
                    cap.release()

                    conversation_content.append({
                        "type": "video",
                        "video": actual_file_path,
                        "fps": fps,
                        "total_pixels": total_pixels
                    })
                    video_durations.append(duration)

                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")

            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                raise

        return conversation_content, video_durations

    def run_model_inference(self, **kwargs) -> Dict[str, Any]:
        """Run Cosmos-specific inference

        Args:
            **kwargs: All inference parameters - Cosmos expects 'media' and 'text'

        Returns:
            Inference results dictionary
        """
        try:
            # Extract Cosmos-specific parameters
            media_data = kwargs.get('media', [])
            prompt = kwargs.get('prompt', 'Describe this video.')
            fps = kwargs.get('fps', 4)
            total_pixels = kwargs.get('total_pixels', 6422528)  # 8192 * 28**2
            max_new_tokens = kwargs.get('max_new_tokens', 4096)

            # Validate required parameters
            if not media_data:
                raise ValueError("Cosmos requires 'media' parameter")

            # Handle both string and list inputs for media
            if isinstance(media_data, str):
                input_files = [media_data]
            else:
                input_files = media_data

            # Validate input files have supported extensions
            if not self.validate_input_files(input_files):
                raise ValueError("One or more input files have unsupported extensions")

            # Process media files
            media_content, video_durations = self.process_media_files(input_files, fps, total_pixels)

            # Create conversation in Cosmos format
            conversation = [
                {
                    "role": "user",
                    "content": media_content + [{"type": "text", "text": prompt}]
                }
            ]

            # Process inputs exactly like inference.py
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Run inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            response = output_text[0] if output_text else ""

            # Check if any video was processed
            has_video = any(d is not None for d in video_durations)

            result = {
                "response": response,
                "has_video": has_video,
                "total_files": len(input_files),
                "fps_used": fps,
                "total_pixels_used": total_pixels
            }

            return result

        except Exception as e:
            logger.error(f"Cosmos inference failed: {e}")
            raise

    def validate_input_files(self, input_files: List[str]) -> bool:
        """Validate that input files have Cosmos-supported extensions

        Args:
            input_files: List of input file paths

        Returns:
            True if all files are supported, False otherwise
        """
        image_exts, video_exts = self.get_supported_file_extensions()
        supported_exts = image_exts + video_exts

        for input_file in input_files:
            file_ext = Path(input_file).suffix.lower()
            if file_ext not in supported_exts:
                logger.error(f"Unsupported file type: {file_ext}")
                return False
        return True


def main():
    """Main function for TAO Cosmos Model Server"""
    parser = argparse.ArgumentParser(description="TAO Cosmos Model Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--job", type=ast.literal_eval)
    parser.add_argument("--docker_env_vars", type=ast.literal_eval)
    parser.add_argument("--idle_timeout_minutes", type=int, default=30,
                        help="Minutes of inactivity before auto-deletion (default: 30)")
    parser.add_argument("--disable_auto_deletion", action="store_true",
                        help="Disable automatic deletion on idle timeout")

    args = parser.parse_args()
    print(args.job, type(args.job))
    print(args.docker_env_vars, type(args.docker_env_vars))

    # Create Cosmos model server using factory method
    server = CosmosInferenceMicroservice.create_from_tao_job(
        job_data=args.job,
        docker_env_vars=args.docker_env_vars,
        port=args.port
    )

    # Configure auto-deletion settings
    server.idle_timeout_minutes = args.idle_timeout_minutes
    server.auto_deletion_enabled = not args.disable_auto_deletion

    if server.auto_deletion_enabled:
        logger.info(f"Auto-deletion enabled with {server.idle_timeout_minutes} minute timeout")

    logger.info("Cosmos model server created")

    # Start server immediately - initialization and model loading will happen in background
    server.start_server_immediate()


if __name__ == "__main__":
    main()

