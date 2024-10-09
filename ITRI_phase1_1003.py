import os
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL
from PIL import Image, ImageEnhance

""" Guidance Scale (指導比例)：決定模型在生成圖像時應該多大程度上遵循你輸入的提示詞

較低的Guidance Scale（例如1到4）：生成的圖像會比較自由，圖像可能與提示詞不完全對應，模型會有更多的創造性發揮，並生成可能比較模糊或者不符合指示的內容。
較高的Guidance Scale（例如7到20）：生成的圖像將會更加緊密地遵循提示詞，但如果設置過高，可能會導致圖像看起來不自然，或者帶來過度過濾、過強的約束。

Inference Steps (推理步驟)：擴散過程中的迭代次數，步驟越多，圖像的質量通常越好，但也需要更多的計算資源和時間。 """

# 載入Realistic Vision模型
base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# 替換為你的VAE模型目錄路徑
vae_model_dir = "VAE_weight"
vae = AutoencoderKL.from_pretrained(vae_model_dir, torch_dtype=torch.float16)
vae.to("cuda")
pipe.vae = vae

# 設定為Euler Ancestral (Euler a) scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 加載 LoRA 權重(後面再補上 LoRA 的 scale 參數)
lora_weights_path = "LoRA_weight\\last003.safetensors"
pipe.load_lora_weights(lora_weights_path, lora_scale=0.75)

# 調整圖像亮度和對比度的函數
def adjust_image(image, brightness=1.0, contrast=1.0):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)
    
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)
    
    return image

# 生成圖像的函數
def generate_image(prompt, width, height, guidance_scale, num_inference_steps):
    seed = int.from_bytes(os.urandom(2), "big")
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return image

# 即時調整圖像亮度和對比度的函數
def preview_adjustments(image, brightness, contrast):
    return adjust_image(image, brightness, contrast)

# 使用Gradio創建介面
with gr.Blocks() as demo:
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="輸入提示詞")
    
    with gr.Row():
        width_input = gr.Slider(256, 2048, value=1024, step=64, label="Width")
        height_input = gr.Slider(256, 2048, value=1024, step=64, label="Height")
    
    with gr.Row():
        guidance_input = gr.Slider(1, 20, value=7, step=1, label="Guidance Scale")
        steps_input = gr.Slider(1, 100, value=30, step=1, label="Inference Steps")
    
    generate_btn = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image", tool="editor", allow_download=True)
    
    with gr.Row():
        brightness_input = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Brightness")
        contrast_input = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Contrast")
    
    preview_btn = gr.Button("Preview Adjustments")
    adjusted_image = gr.Image(label="Adjusted Image", tool="editor", allow_download=True)

    # 生成圖像
    def generate_and_display(prompt, width, height, guidance_scale, num_inference_steps):
        image = generate_image(prompt, width, height, guidance_scale, num_inference_steps)
        return image

    generate_btn.click(
        fn=generate_and_display, 
        inputs=[prompt_input, width_input, height_input, guidance_input, steps_input], 
        outputs=output_image
    )
    
    preview_btn.click(
        fn=preview_adjustments, 
        inputs=[output_image, brightness_input, contrast_input], 
        outputs=adjusted_image
    )

demo.launch()
