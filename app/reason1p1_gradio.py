import gradio as gr
import torch
import torchvision.transforms.functional
import os
import tempfile
import zipfile
import shutil
from transformers import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
# from huggingface_hub import login  # Commented out - no auth needed for public model

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import numpy as np

# Try to import vLLM, fall back if not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except (ImportError, ValueError) as e:
    VLLM_AVAILABLE = False
    if "aimv2" in str(e):
        print("⚠️ vLLM/transformers version conflict detected (aimv2 config), will use standard transformers")
    else:
        print(f"⚠️ vLLM not available: {e}, will use standard transformers")


def tensor_to_pil_images(video_tensor):
    """
    Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) to a list of PIL images.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        list[PIL.Image.Image]: List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images


def overlay_text(
    images: List[Image.Image],
    fps: float,
    border_height: int = 28,  # this is due to patch size of 28
    temporal_path_size: int = 2,  # Number of positions to cycle through
    font_size: int = 20,
    font_color: str = "white",
) -> Tuple[List[Image.Image], List[float]]:
    """
    Overlay text on a list of PIL images with black border.
    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        border_height: Height of the black border in pixels (default: 28)
        temporal_path_size: Number of positions to cycle through (default: 2)
        font_size: Font size for the text (default: 20)
        font_color: Color of the text (default: "white")

    Returns:
        List of PIL images with text overlay
        List of timestamps
    """

    # Try to use DejaVu Sans Mono font for better readability
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + border_height
        new_image = Image.new("RGB", (width, new_height), color="black")

        # Paste original image at the top
        new_image.paste(image, (0, 0))

        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)

        # Calculate timestamp for current frame
        total_seconds = i / fps
        text = f"{total_seconds:.2f}s"

        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback estimation
            text_width = len(text) * 8
            text_height = 12

        # Define available positions (cycling through horizontal positions)
        position_idx = i % temporal_path_size
        section_width = width // temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        processed_images.append(new_image)

    return processed_images, [i / fps for i in range(len(images))]


def save_timestamped_frames(images_with_timestamp: List[Image.Image], fps: float, base_filename: str) -> str:
    """
    Save timestamped frames to files and create a zip archive.
    
    Args:
        images_with_timestamp: List of PIL images with timestamps
        fps: Frames per second
        base_filename: Base name for the files (without extension)
    
    Returns:
        Path to the created zip file
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="cosmos_frames_")
    
    try:
        # Save each frame
        frame_paths = []
        for i, image in enumerate(images_with_timestamp):
            timestamp = i / fps
            frame_filename = f"frame_{i:04d}_t{timestamp:.2f}s.png"
            frame_path = os.path.join(temp_dir, frame_filename)
            image.save(frame_path, "PNG")
            frame_paths.append(frame_path)
        
        # Create zip file
        zip_filename = f"{base_filename}_timestamped_frames.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for frame_path in frame_paths:
                arcname = os.path.basename(frame_path)
                zipf.write(frame_path, arcname)
        
        return zip_path
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


# Role configurations
ROLES = {
    "Video Timestamping": """"Please provide captions of all the events in the video with timestamps using the following format: 
     <start time> <end time> caption of event 1.\n<start time> <end time> caption of event 2.\n
    At each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question.""",
    "General Assistant": "You are a helpful assistant. Answer the question in the following format: \n<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
    "Video Analyzer": """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video. Watch carefully and focus on the following details:

* Physical accuracy (gravity, collision, object interaction, fluid dynamics, object permanence, etc.)
* Common sense
* Cause-and-effect
* Temporal consistency
* Spatial consistency
* Human motion
* Material and Texture realism

Here are some examples of commonly found artifacts and anomalies:

* If objects penetrate each other, this indicates a failure in collision detection, object interaction, and physical accuracy.
* If hands penetrate each other, or hands pass through objects, this indicates a failure in collision detection, object interaction, and physical accuracy.
* If an object moves in an unexpected way or move without any apparent reason, this suggests a failure in causality, object interaction, and physical accuracy.
* If an object suddenly flips or changes direction, this suggests a failure in temporal consistency.
* If an object suddenly appears or disappears, or the count of objects in the video suddenly changes, this suggests a failure in temporal consistency.
* If an object transforms or deforms half way through the video, this suggests a failure in temporal consistency.
* If an object is used in a way that defies its intended purpose or normal function, this indicates a violation of common sense.
* If the liquid flows through a solid object, such as water flowing through a pan, this suggests a failure in physical accuracy and fluid dynamics.
* If a person's legs or arms suddenly switch positions in an impossible way—such as the left leg appearing where the right leg was just a moment ago, this suggests a failure in human motion and temporal consistency.
* If a person's body suddenly morphs or changes shape, this suggests a failure in human motion and temporal consistency.
* If an object's texture, material or surface is unnaturally smooth, this suggests a failure in object surface reconstruction.

Here are some examples of non-artifacts you should not include in your analysis:

* Being an animated video, such as a cartoon, does not automatically make it artifacts.
* Avoid ungrounded and over-general explanations such as overall impression, artistic style, or background elements.
* The video has no sound. Avoid explanations based on sound.
* Do not mention lighting, shadows, blurring, or camera effects in your analysis.

Answer the question in English with provided options in the following format:
<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.""",
    "Custom Role": "You are a helpful assistant. Answer the question in the following format: \n<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
}

# Global variables
MODEL_PATH = "nvidia/Cosmos-Reason1-7B"
llm = None  # vLLM model
model = None  # transformers model
processor = None
current_method = None


# def authenticate_and_initialize(hf_token):
#     """Authenticate with HF and initialize the model (tries vLLM first, falls back to transformers)"""
#     global llm, model, processor, current_method
#     
#     if not hf_token.strip():
#         return "❌ Please provide a valid HF token!", gr.Button(interactive=True)
#     
#     # Return processing status immediately and disable button
#     def processing_update():
#         return "🔄 Initializing model... Please wait (this may take a few minutes)", gr.Button(interactive=False)
#     
#     try:
#         # Step 1: Authenticate with Hugging Face
#         login(token=hf_token.strip())
#         os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()
#         
#         # Step 2: Initialize processor first
#         processor = AutoProcessor.from_pretrained(
#             MODEL_PATH,
#             token=hf_token.strip()
#         )
#         
#         # Step 3: Try vLLM first
#         if VLLM_AVAILABLE:
#             try:
#                 llm = LLM(
#                     model=MODEL_PATH,
#                     limit_mm_per_prompt={"image": 10, "video": 10},
#                 )
#                 current_method = "vLLM"
#                 return f"✅ Successfully initialized with vLLM!\n🔐 Authenticated with token: {hf_token[:8]}...\n🤖 Model: {MODEL_PATH}\n🚀 Using: vLLM (faster inference)", gr.Button(interactive=True)
#             except Exception as vllm_error:
#                 print(f"vLLM failed: {vllm_error}")
#                 # Fall back to transformers
#                 pass
#         
#         # Step 4: Fall back to standard transformers
#         model = AutoModelForVision2Seq.from_pretrained(
#             MODEL_PATH,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True,
#             token=hf_token.strip()
#         )
#         current_method = "transformers"
#         return f"✅ Successfully initialized with transformers!\n🔐 Authenticated with token: {hf_token[:8]}...\n🤖 Model: {MODEL_PATH}\n🔄 Using: Standard transformers (more compatible)", gr.Button(interactive=True)
#         
#     except Exception as e:
#         error_msg = str(e)
#         if "401" in error_msg or "Repository not found" in error_msg or "Unauthorized" in error_msg:
#             return f"❌ Authentication failed!\n\n🔑 Invalid HF token or insufficient permissions.\n\nPlease check:\n- Token is valid and starts with 'hf_'\n- You have access to the private repository: {MODEL_PATH}\n- Token has 'read' permissions\n\nError: {error_msg}", gr.Button(interactive=True)
#         else:
#             return f"❌ Model initialization failed: {error_msg}", gr.Button(interactive=True)


def initialize_model():
    """Initialize the model without authentication (for public models)"""
    global llm, model, processor, current_method
    
    try:
        # Step 1: Initialize processor first
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # Step 2: Try vLLM first
        if VLLM_AVAILABLE:
            try:
                llm = LLM(
                    model=MODEL_PATH,
                    limit_mm_per_prompt={"image": 10, "video": 10},
                )
                current_method = "vLLM"
                return f"✅ Successfully initialized with vLLM!\n🤖 Model: {MODEL_PATH}\n🚀 Using: vLLM (faster inference)", gr.Button(interactive=True)
            except Exception as vllm_error:
                print(f"vLLM failed: {vllm_error}")
                # Fall back to transformers
                pass
        
        # Step 3: Fall back to standard transformers
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        current_method = "transformers"
        return f"✅ Successfully initialized with transformers!\n🤖 Model: {MODEL_PATH}\n🔄 Using: Standard transformers (more compatible)", gr.Button(interactive=True)
        
    except Exception as e:
        return f"❌ Model initialization failed: {str(e)}", gr.Button(interactive=True)


def process_media_vllm(image_path=None, video_path=None, text_prompt="Describe the notable events in the provided video.", fps=2.0, total_pixels=12688256, role="Video Timestamping", custom_role_text="", temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_tokens=1024):
    """Process image or video using vLLM and save frames if video"""
    # Your exact sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    # Use custom role text if role is "Custom Role"
    role_prompt = custom_role_text if role == "Custom Role" else ROLES[role]

    # Build messages based on input type
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": []}
    ]
    
    # Add text prompt
    if text_prompt:
        messages[1]["content"].append({"type": "text", "text": text_prompt})
    
    # Add image if provided
    if image_path is not None:
        messages[1]["content"].append({"type": "image", "image": image_path})
    
    # Add video if provided
    if video_path is not None:
        messages[1]["content"].append({
            "type": "video", 
            "video": video_path,
            "fps": fps,
            "total_pixels": total_pixels,
        })

    # Your exact processing logic
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    # Store timestamped images for download (only for video)
    all_timestamped_images = []
    zip_path = None
    
    if video_inputs is not None:
        video_inputs_with_timestamp = []
        for video in video_inputs:
            images = tensor_to_pil_images(video)
            images_with_timestamp, _ = overlay_text(images, fps)
            all_timestamped_images.extend(images_with_timestamp)  # Store for zip
            tensors = [torchvision.transforms.functional.pil_to_tensor(img) for img in images_with_timestamp]
            tensors = torch.stack(tensors, dim=0)
            video_inputs_with_timestamp.append(tensors)
        video_inputs = video_inputs_with_timestamp

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    # Create zip file with timestamped frames (only for video)
    if all_timestamped_images:
        base_filename = os.path.splitext(os.path.basename(video_path))[0] if video_path else "video"
        zip_path = save_timestamped_frames(all_timestamped_images, fps, f"{base_filename}_vllm")

    return generated_text, zip_path


def process_media_transformers(image_path=None, video_path=None, text_prompt="Describe the notable events in the provided video.", fps=2.0, total_pixels=6369152, role="Video Timestamping", custom_role_text="", temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_new_tokens=4096):
    """Process image or video using standard transformers and save frames if video"""
    
    # Use custom role text if role is "Custom Role"
    role_prompt = custom_role_text if role == "Custom Role" else ROLES[role]

    # Build messages based on input type
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": []}
    ]
    
    # Add text prompt
    if text_prompt:
        messages[1]["content"].append({"type": "text", "text": text_prompt})
    
    # Add image if provided
    if image_path is not None:
        messages[1]["content"].append({"type": "image", "image": image_path})
    
    # Add video if provided
    if video_path is not None:
        messages[1]["content"].append({
            "type": "video", 
            "video": video_path,
            "fps": fps,
            "total_pixels": total_pixels,
        })

    # Process the prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process vision information
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    # Add timestamps to video frames and store for download (only for video)
    all_timestamped_images = []
    zip_path = None
    
    if video_inputs is not None:
        video_inputs_with_timestamp = []
        for video in video_inputs:
            images = tensor_to_pil_images(video)
            images_with_timestamp, _ = overlay_text(images, fps)
            all_timestamped_images.extend(images_with_timestamp)  # Store for zip
            tensors = [torchvision.transforms.functional.pil_to_tensor(img) for img in images_with_timestamp]
            tensors = torch.stack(tensors, dim=0)
            video_inputs_with_timestamp.append(tensors)
        video_inputs = video_inputs_with_timestamp

    # Prepare inputs for the model
    inputs = processor(
        text=prompt,
        images=image_inputs if image_inputs is not None else None,
        videos=video_inputs if video_inputs is not None else None,
        return_tensors="pt"
    ).to(model.device)

    # Generation config
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }

    # Generate output
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Create zip file with timestamped frames (only for video)
    if all_timestamped_images:
        base_filename = os.path.splitext(os.path.basename(video_path))[0] if video_path else "video"
        zip_path = save_timestamped_frames(all_timestamped_images, fps, f"{base_filename}_transformers")
    
    return generated_text, zip_path


def process_media(image_path=None, video_path=None, text_prompt="Describe the notable events in the provided video.", fps=2.0, total_pixels=12688256, role="Video Timestamping", custom_role_text="", temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_tokens=4096):
    """Process image or video with current method (vLLM or transformers)"""
    
    if current_method is None:
        return "❌ Please initialize the model first!", None
    
    if not image_path and not video_path:
        return "❌ Please upload an image or video first!", None
    
    try:
        if current_method == "vLLM" and llm is not None:
            result, zip_path = process_media_vllm(image_path, video_path, text_prompt, fps, total_pixels, role, custom_role_text, temperature, top_p, repetition_penalty, max_tokens)
            return f"🚀 [vLLM] {result}", zip_path
        elif current_method == "transformers" and model is not None:
            # Use different default for transformers if not explicitly set
            transformer_pixels = total_pixels if total_pixels != 12688256 else 6369152
            result, zip_path = process_media_transformers(image_path, video_path, text_prompt, fps, transformer_pixels, role, custom_role_text, temperature, top_p, repetition_penalty, max_tokens)
            return f"🔄 [Transformers] {result}", zip_path
        else:
            return "❌ Model not properly initialized!", None

    except Exception as e:
        return f"❌ Error processing media: {str(e)}", None


# Simple Gradio interface
with gr.Blocks(title="Cosmos-Reason1.1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Cosmos-Reason1.1 (Latest checkpoint: Aug 1st, 2025)")
    gr.Markdown("🖼️ **Image & Video Analysis** | 🎭 **Multiple Roles** | 🚀 **vLLM + Transformers**")
    
    # Token input and model initialization - COMMENTED OUT (no auth needed for public model)
    # with gr.Row():
    #     hf_token = gr.Textbox(
    #         label="HF Token", 
    #         type="password",
    #         placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    #     )
    #     init_btn = gr.Button("Initialize Model")
    # 
    # init_status = gr.Markdown("Enter HF token and click Initialize Model")
    
    # Simple initialization for public model
    with gr.Row():
        init_btn = gr.Button("Initialize Model")
    
    init_status = gr.Markdown("Click Initialize Model to load the public model")
    
    # Role selection
    with gr.Row():
        role_selector = gr.Dropdown(
            choices=list(ROLES.keys()),
            value="Video Timestamping",
            label="Select Role"
        )
        
    custom_role_panel = gr.Group(visible=False)
    with custom_role_panel:
        custom_role_text = gr.Textbox(
            label="Custom Role Instructions",
            placeholder="Enter custom role instructions here...",
            lines=5,
            value=ROLES["Custom Role"]
        )
        apply_custom_role = gr.Button("Apply Custom Role")
        custom_role_status = gr.Markdown()
        
        def update_custom_role(text):
            ROLES["Custom Role"] = text
            return "Custom role updated successfully!"
        
        apply_custom_role.click(
            fn=update_custom_role,
            inputs=[custom_role_text],
            outputs=[custom_role_status]
        )
    
    def toggle_custom_role(role):
        return gr.update(visible=(role == "Custom Role"))

    # Media processing
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Media Input")
            image_input = gr.Image(label="Image Input (Optional)", type="filepath")
            video_input = gr.Video(label="Video Input (Optional)")
            gr.Markdown("**Note:** Both inputs are optional for API usage. Please provide either an image OR a video (at least one is required for processing).")
            
            text_input = gr.Textbox(
                label="Text Prompt", 
                value="Describe the notable events in the provided video.",
                placeholder="Enter your question or instruction..."
            )
            
            # Set up role selector event handlers now that text_input is defined
            def update_text_prompt_based_on_role(role):
                # Update text prompt based on role
                if role == "Video Timestamping":
                    default_prompt = "Describe the notable events in the provided video."
                elif role == "General Assistant":
                    default_prompt = "Analyze this image/video and provide insights."
                elif role == "Video Analyzer":
                    default_prompt = "Analyze the video for artifacts and anomalies."
                else:
                    default_prompt = "Describe what you see in this image/video."
                
                return gr.update(visible=(role == "Custom Role")), gr.update(value=default_prompt)
            
            role_selector.change(
                fn=update_text_prompt_based_on_role,
                inputs=[role_selector],
                outputs=[custom_role_panel, text_input]
            )
            
            with gr.Row():
                fps_input = gr.Slider(0.5, 10.0, value=2.0, step=0.5, label="FPS (Video only)")
                total_pixels_input = gr.Number(value=12688256, label="Total Pixels", info="Vision tokens: vLLM=12688256, Transformers=6369152")
            
            # Generation configuration
            gr.Markdown("### Generation Parameters")
            with gr.Row():
                temperature_input = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="Temperature")
                top_p_input = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
            with gr.Row():
                repetition_penalty_input = gr.Slider(1.0, 2.0, value=1.05, step=0.05, label="Repetition Penalty")
                max_tokens_input = gr.Slider(512, 8192, value=1024, step=256, label="Max Tokens")
            
            process_btn = gr.Button("Process Media")
        
        with gr.Column():
            output = gr.Textbox(label="Generated Captions", lines=15)
            
            # Download section
            gr.Markdown("### Download Timestamped Frames")
            download_file = gr.File(
                label="Timestamped Frames (ZIP)",
                visible=False
            )
            download_info = gr.Markdown("Process a video to download timestamped frames")
    
    # Event handlers with button state management
    def handle_initialization():
        """Handle initialization with button state management (no auth needed)"""
        # First update: show processing and disable button
        yield "🔄 Initializing model... Please wait (this may take a few minutes)", gr.Button(interactive=False)
        
        # Perform actual initialization
        result, button_state = initialize_model()
        
        # Final update: show result and re-enable button
        yield result, gr.Button(interactive=True)
    
    # def handle_initialization(hf_token):
    #     """Handle initialization with button state management"""
    #     # First update: show processing and disable button
    #     yield "🔄 Initializing model... Please wait (this may take a few minutes)", gr.Button(interactive=False)
    #     
    #     # Perform actual initialization
    #     result, button_state = authenticate_and_initialize(hf_token)
    #     
    #     # Final update: show result and re-enable button
    #     yield result, gr.Button(interactive=True)
    
    init_btn.click(
        fn=handle_initialization,
        inputs=[],  # No hf_token input needed for public model
        outputs=[init_status, init_btn]
    )
    
    def handle_media_processing(image_path=None, video_path=None, text_prompt="Describe the notable events in the provided video.", fps=2.0, total_pixels=12688256, role="Video Timestamping", custom_role_text="", temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_tokens=4096):
        """Handle media processing with button state management"""
        # First update: show processing and disable button
        yield "🔄 Processing media... Please wait (this may take several minutes)", gr.File(visible=False), "⏳ Processing...", gr.Button(interactive=False)
        
        # Perform actual processing
        result, zip_path = process_media(image_path, video_path, text_prompt, fps, total_pixels, role, custom_role_text, temperature, top_p, repetition_penalty, max_tokens)
        
        # Final update: show results and re-enable button
        if zip_path:
            yield result, gr.File(value=zip_path, visible=True), "✅ Timestamped frames ready for download!", gr.Button(interactive=True)
        else:
            if video_path:
                yield result, gr.File(visible=False), "❌ No frames available for download", gr.Button(interactive=True)
            else:
                yield result, gr.File(visible=False), "✅ Image processing completed (no frames to download)", gr.Button(interactive=True)
    
    process_btn.click(
        fn=handle_media_processing,
        inputs=[image_input, video_input, text_input, fps_input, total_pixels_input, role_selector, custom_role_text, temperature_input, top_p_input, repetition_penalty_input, max_tokens_input],
        outputs=[output, download_file, download_info, process_btn],
        api_name="process_media"
    )
    
    # Info section - Collapsible
    with gr.Accordion("How it works", open=False):
        gr.Markdown("""
        ### Process Flow:
        1. **Tries vLLM first** - Your exact script with fast inference
        2. **Falls back to transformers** - If vLLM has compatibility issues
        3. **Supports both images and videos** - Process either media type
        4. **Role-based processing** - Different system prompts for different tasks
        5. **Saves timestamped frames** - Downloads ZIP file with all video frames
        
        ### Features:
        - 🚀 **vLLM**: Faster inference, may have compatibility issues
        - 🔄 **Transformers**: Slower but more compatible  
        - 🖼️ **Image Support**: Process single images with any role
        - 🎥 **Video Support**: Process videos with timestamp overlays
        - 🌐 **API Ready**: Both image and video inputs are optional parameters in API calls
        - 🔧 **Flexible API**: Call with just image, just video, or custom parameters - defaults handle the rest
        - 🎭 **Role System**: Pre-defined roles (Video Timestamping, General Assistant, Video Analyzer, Custom)
        - 📁 **Frame Download**: ZIP file with timestamped frames for videos (format: `frame_XXXX_tY.YYs.png`)
        - 🕐 **Pixel Timestamps**: Each video frame shows timestamp in bottom border
        - ⚙️ **Configurable Generation**: Adjust temperature, top_p, repetition penalty, and max tokens
        - 🎯 **Configurable Pixels**: Control vision token count for processing
        
        ### Roles:
        - **Video Timestamping**: Default role for video analysis with timestamps
        - **General Assistant**: General purpose assistant with thinking format
        - **Video Analyzer**: Specialized for detecting artifacts and anomalies in videos
        - **Custom Role**: Define your own system prompt
        
        ### Parameter Details:
        - **Total Pixels**: Controls vision processing quality (vLLM default: 12,688,256, Transformers: 6,369,152)
        - **Temperature**: Controls randomness (0.1 = deterministic, 2.0 = very creative)
        - **Top P**: Nucleus sampling threshold (0.95 = consider top 95% probable tokens)
        - **Repetition Penalty**: Reduces repetitive text (1.05 = slight penalty)
        - **Max Tokens**: Maximum length of generated response (4096 = ~3000 words)
        - **FPS**: Frames per second for video processing (affects timestamp accuracy)
        
        ### Frame Naming (Video only):
        `frame_0001_t0.50s.png` = Frame 1 at 0.50 seconds
        
        ### API Usage:
        **Option 1: Python Client (simplified - only pass what you need)**
        ```python
        from gradio_client import Client
        client = Client("http://localhost:8080")
        
        # Image only (easiest way)
        result = client.predict(
            "/path/to/image.jpg",  # image_input
            api_name="/process_media"
        )
        
        # Video only (easiest way) 
        result = client.predict(
            None,                  # image_input
            "/path/to/video.mp4",  # video_input
            api_name="/process_media"
        )
        
        # Custom prompt with image
        result = client.predict(
            "/path/to/image.jpg",     # image_input
            None,                     # video_input
            "Analyze this image",     # text_prompt
            api_name="/process_media"
        )
        ```
        
        **Option 2: HTTP API (can omit trailing parameters)**
        ```bash
        # Image only processing
        curl -X POST "http://localhost:8080/api/process_media" \\
          -H "Content-Type: application/json" \\
          -d '{"data": ["/path/to/image.jpg"]}'
          
        # Video only processing  
        curl -X POST "http://localhost:8080/api/process_media" \\
          -H "Content-Type: application/json" \\
          -d '{"data": [null, "/path/to/video.mp4"]}'
        ```
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        show_error=True,
        allowed_paths=["/mnt/pvc/gradio", "/mnt/pvc/code/cosmos-reason1/app"],
    ) 