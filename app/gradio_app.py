import gradio as gr
import torch
import os
import json
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# Model configuration
MODEL_PATH = "/mnt/pvc/checkpoints/nvidia/Cosmos-Reason1-7B"    # Older checkpoint
# MODEL_PATH = "nvidia/Cosmos-Reason1-7B"    # Latest checkpoint from the Hugging Face

# Role configurations
ROLES = {
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

# Default configuration
default_config = {
    "attention_mode": "sdpa",
    "torch_dtype": "float16",
    "device_map": "auto",
    "trust_remote_code": True
}

# Load or create config file
config_file = "cosmos_reason1_config.json"
try:
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=4)
        config = default_config
    else:
        with open(config_file, "r") as f:
            config = json.load(f)
except Exception as e:
    print(f"Warning: Could not load config file: {e}")
    print("Using default configuration")
    config = default_config

# Initialize the model with configuration
try:
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=getattr(torch, config["torch_dtype"]),
        device_map=config["device_map"],
        trust_remote_code=config["trust_remote_code"]
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Initialize sampling parameters
generation_config = {
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_new_tokens": 1024,
}

# Initialize the processor
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading processor: {e}")
    raise

def process_input(image=None, video=None, text_prompt="", temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_tokens=1024, total_pixels=12688256, fps=2.0, role="General Assistant", custom_role_text=""):
    """Process the input and generate a response."""
    try:
        # Validate that at least one media input is provided
        if image is None and video is None:
            return "Please provide either an image or a video input.", "❌ No media input provided. Please upload an image or video."
        
        # Use custom role text if role is "Custom Role"
        role_prompt = custom_role_text if role == "Custom Role" else ROLES[role]
        
        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": []}
        ]
        
        # Add text prompt
        if text_prompt:
            messages[1]["content"].append({"type": "text", "text": text_prompt})
        
        # Add image if provided
        if image is not None:
            messages[1]["content"].append({"type": "image", "image": image})
        
        # Add video if provided
        if video is not None:
            messages[1]["content"].append({
                "type": "video",
                "video": video,
                "fps": fps,  # Use the user-provided FPS parameter
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
        
        # Prepare inputs
        inputs = processor(
            text=prompt,
            images=image_inputs if image_inputs is not None else None,
            videos=video_inputs if video_inputs is not None else None,
            return_tensors="pt"
        ).to(model.device)
        
        # Update generation config with user parameters
        current_generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_tokens,
        }
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **current_generation_config
            )
        
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, "✅ Generation completed successfully!"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"Error processing input: {str(e)}", f"❌ Error occurred:\n{error_trace}"

def apply_config_changes(attention_mode, torch_dtype, device_map):
    """Apply configuration changes and save to file."""
    try:
        config = {
            "attention_mode": attention_mode,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        return "Configuration updated. Please restart the application for changes to take effect."
    except Exception as e:
        return f"Error updating configuration: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Cosmos-Reason1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Cosmos-Reason1")
    gr.Markdown("Upload an image or video and ask a question about it.")
    gr.Markdown(
        """
        [[Model]](https://huggingface.co/nvidia/Cosmos-Reason1-7B) | [[Code]](https://github.com/nvidia-cosmos/cosmos-reason1)
        """
    )
    
    # with gr.Accordion("Model Configuration", open=False):
    #     attention_mode = gr.Dropdown(
    #         choices=["sdpa", "xformers", "flash_attention_2"],
    #         value=config["attention_mode"],
    #         label="Attention Mode"
    #     )
    #     torch_dtype = gr.Dropdown(
    #         choices=["float16", "bfloat16", "float32"],
    #         value=config["torch_dtype"],
    #         label="Torch Data Type"
    #     )
    #     device_map = gr.Dropdown(
    #         choices=["auto", "cuda", "cpu"],
    #         value=config["device_map"],
    #         label="Device Map"
    #     )
    #     config_btn = gr.Button("Apply Configuration")
    #     config_msg = gr.Markdown()
        
    #     config_btn.click(
    #         fn=apply_config_changes,
    #         inputs=[attention_mode, torch_dtype, device_map],
    #         outputs=config_msg
    #     )
    
    with gr.Row():
        with gr.Column():
            role_selector = gr.Dropdown(
                choices=list(ROLES.keys()),
                value="General Assistant",
                label="Select Role"
            )
            
            custom_role_panel = gr.Group(visible=False)
            with custom_role_panel:
                custom_role_text = gr.Textbox(
                    label="Custom Role Instructions",
                    placeholder="Enter custom role instructions here...",
                    lines=10,
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
            
            role_selector.change(
                fn=toggle_custom_role,
                inputs=[role_selector],
                outputs=[custom_role_panel]
            )
            
            image_input = gr.Image(label="Image Input (Optional)", type="filepath")
            video_input = gr.Video(label="Video Input (Optional)")
            text_input = gr.Textbox(label="Question", placeholder="Ask a question about the image or video...")
            
            gr.Markdown("**Note:** Please provide either an image OR a video (both inputs are optional, but at least one is required).")
            
            with gr.Accordion("Generation Parameters", open=False):
                temperature = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.05, label="Repetition Penalty")
                max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
                fps = gr.Slider(1.0, 8.0, value=2.0, step=1.0, label="Video FPS (Frames Per Second)")
                total_pixels = gr.Number(value=12688256, label="Total Pixels", info="Vision tokens: vLLM=12688256, Transformers=6369152")
                
                gr.Markdown("**Note:** Lower FPS values reduce memory usage and processing time for videos.", elem_classes=["info-box"])
            
            submit_btn = gr.Button("Submit")
        
        with gr.Column():
            output = gr.Textbox(label="Model Response", lines=10)
            status = gr.Markdown(label="Status")
    
    submit_btn.click(
        fn=process_input,
        inputs=[
            image_input,
            video_input,
            text_input,
            temperature,
            top_p,
            repetition_penalty,
            max_tokens,
            total_pixels,
            fps,
            role_selector,
            custom_role_text
        ],
        outputs=[output, status]
    )
    
    # Example for image
    image_examples = [
        [
            "/mnt/pvc/code/cosmos-reason1/app/group_in_park.jpg",
            "What is happening in this image?"
        ]
    ]
    
    # Example for video
    video_examples = [
        [
            "/mnt/pvc/code/cosmos-reason1/app/car_curb_video.mp4",
            "Analyze the video, what is wrong with it?"
        ]
    ]
    
    # Image example block
    gr.Examples(
        examples=image_examples,
        inputs=[image_input, text_input],
        label="Image Example: click to load then hit Submit"
    )
    
    # Video example block
    gr.Examples(
        examples=video_examples,
        inputs=[video_input, text_input],
        label="Video Example: click to load then hit Submit"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        # Configure file upload limits
        # max_file_size="500MB",  # Adjust as needed
        allowed_paths=["/mnt/pvc/gradio", "/mnt/pvc/code/cosmos-reason1/app"],  # Allow access to output directory
    ) 