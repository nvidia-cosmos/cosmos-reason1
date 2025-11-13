import gradio as gr
import os
import shutil
from pathlib import Path
import subprocess
import time
import re
import threading
from flask import Flask, send_file, request, jsonify
from werkzeug.serving import make_server
from openai import OpenAI

# from urllib.parse import urlencode
from urllib.parse import quote, unquote
# OpenAI client for TensorRT-LLM backend
openai_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="tensorrt_llm",
)

# Video static service
VIDEO_SERVER_PORT = 8002
# VIDEO_SERVER_HOST = "127.0.0.1"
VIDEO_SERVER_HOST = "0.0.0.0"
VIDEO_BASE_URL = f"http://{VIDEO_SERVER_HOST}:{VIDEO_SERVER_PORT}"

# flask application for video service
video_server_app = Flask(__name__)
video_server = None
video_server_thread = None

@video_server_app.route('/upload', methods=['POST'])
def upload_file():
    """file upload api"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # temp uploaded file save path
        temp_dir = Path("tmp_assets")
        temp_dir.mkdir(exist_ok=True)
        
        filename = os.path.basename(file.filename)
        file_path = temp_dir / filename
        
        if file_path.exists():
            name, ext = os.path.splitext(filename)
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}{ext}"
            file_path = temp_dir / filename
        
        file.save(str(file_path))
        
        # return file information and download url
        file_url = f"{VIDEO_BASE_URL}/download/{quote(filename)}"
        return jsonify({
            'success': True,
            'filename': filename,
            'url': file_url,
            'size': file_path.stat().st_size
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@video_server_app.route('/download/<path:filename>')
def download_file(filename):
    """file download"""
    try:
        temp_dir = Path("tmp_assets")
        filename = unquote(filename)
        filename = os.path.basename(filename)
        file_path = temp_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # 根据文件扩展名确定 MIME 类型
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.json': 'application/json',
        }
        mimetype = mime_types.get(ext, 'application/octet-stream')
        
        return send_file(
            str(file_path),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@video_server_app.route('/video/<path:filename>')
def serve_video(filename):
    """video file service"""
    temp_dir = Path("tmp_assets")
    filename = unquote(filename)
    video_path = temp_dir / filename
    if video_path.exists():
        return send_file(
            str(video_path),
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    return "Video not found", 404

def start_video_server():
    """backedn thread for video service"""
    global video_server
    temp_dir = Path("tmp_assets")
    temp_dir.mkdir(exist_ok=True)
    
    video_server = make_server(
        VIDEO_SERVER_HOST,
        VIDEO_SERVER_PORT,
        video_server_app,
        threaded=True
    )
    video_server.serve_forever()


def get_gpu_info():
    """nvidia-smi command for GPU info"""
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", shell=True
        ).decode()
        used, total = result.strip().split(',')
        used = int(used.strip())
        total = int(total.strip())
        percent = 100 * used / total if total > 0 else 0
        info = f"GPU Memory: {used}MB / {total}MB ({percent:.1f}%)"
    except Exception:
        info = "GPU Info: N/A"
    return info

def parse_tags(output_text):
    think = ""
    answer = ""
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    return think, answer

def cosmos_reason1_infer_openai(video, prompt):
    start = time.time()

    temp_dir = Path("tmp_assets")
    temp_dir.mkdir(exist_ok=True)

    shutil.copy(video, temp_dir)
    video_path = temp_dir / Path(video).name
    video_filename = video_path.name
    video_filename = quote(video_filename)

    video_url = f"http://192.168.199.139:8002/video/{video_filename}"
    # video_url = urlencode(video_url)

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }, {
            "type": "video_url",
            "video_url": {
                "url": video_url
            }
        }]
    }]
    print(messages)

    response = openai_client.chat.completions.create(
        # model=model_name,
        model="/mnt/cosmos-reason1/model_data/Cosmos-Reason1-7B",
        messages=messages,
        max_tokens=4096,
    )

    output_text = response.choices[0].message.content
    reasoning, answer = parse_tags(output_text)

    elapsed = time.time() - start
    gpu_info = get_gpu_info()
    status = f"{gpu_info} | Infer time: {elapsed:.2f}s | Video URL: {video_url}"

    return reasoning, answer, status

if __name__ == "__main__":
    # start video static service
    if video_server_thread is None or not video_server_thread.is_alive():
        video_server_thread = threading.Thread(target=start_video_server, daemon=True)
        video_server_thread.start()
        print(f"Video server started at {VIDEO_BASE_URL}")
        time.sleep(1)
    
    # ====== Gradio UI ==========
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("## Warehouse Reasoning Demo (Cosmos Reason1)")
        with gr.Row():
            with gr.Column(scale=3):
                video_input = gr.Video(label="Video Input (.mp4)", format="mp4")
                user_prompt = gr.Textbox(label="User Prompt", lines=2, max_lines=5, placeholder="Ask a question about the video ...")
                run_btn = gr.Button("Run")
            with gr.Column(scale=2):
                gr.Markdown("### Output")
                reasoning_box = gr.Textbox(label="Reasoning Process", interactive=False, lines=8, show_copy_button=True)
                answer_box = gr.Textbox(label="Model Response", interactive=False, lines=2, show_copy_button=True)
                gpu_label = gr.Markdown(value="GPU Memory", elem_id="gpu-status")
        run_btn.click(
            # cosmos_reason1_infer,
            cosmos_reason1_infer_openai,
            inputs=[video_input, user_prompt],
            outputs=[reasoning_box, answer_box, gpu_label]
        )


    demo.launch(
        server_name="0.0.0.0",
        server_port=8001
    )
