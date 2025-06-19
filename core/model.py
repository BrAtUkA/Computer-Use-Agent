# model loading and inference

import sys
import torch
import time
from threading import Thread
from typing import Any, Generator, List
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


MODEL_PATH = r"C:\AI\Holo2-4B"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

SPEED_PRESETS = {
    "Quality": 1280 * 28 * 28,
    "Balanced": 896 * 28 * 28,
    "Fast": 768 * 28 * 28,
    "Fastest": 512 * 28 * 28,
}

current_max_pixels = MAX_PIXELS

def set_speed_preset(preset_name: str):
    global current_max_pixels
    if preset_name in SPEED_PRESETS:
        current_max_pixels = SPEED_PRESETS[preset_name]
        print(f"[CONFIG] Speed preset set to '{preset_name}': max_pixels={current_max_pixels}")
    else:
        print(f"[CONFIG] Unknown preset '{preset_name}', keeping current: {current_max_pixels}")

def get_speed_presets():
    return list(SPEED_PRESETS.keys())


model = None
processor = None
model_loaded = False
load_error_message = ""


def load_model():
    global model, processor, model_loaded, load_error_message
    
    print(f"Loading model and processor from {MODEL_PATH}...")
    
    try:
        attn_impl = "sdpa"
        print("Using SDPA (PyTorch native) for efficient attention")
        
        print("About to load model...")
        sys.stdout.flush()
        
        # 8bit quant needed for this models custom code
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        
        model.eval()

        if model and hasattr(model, 'device'):
            print(f"Model loaded on: {model.device}")

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
        model_loaded = True
        print("Model and processor loaded successfully.")
        print(f"Image config: min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}")
        
    except Exception as e:
        import traceback
        load_error_message = (
            f"Error loading model/processor: {e}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            "Make sure the model files are present at the specified path."
        )
        print(load_error_message)
        sys.stdout.flush()


def resize_image_for_model(pil_image: Image.Image) -> Image.Image:
    image_proc_config = processor.image_processor
    
    patch_size = getattr(image_proc_config, 'patch_size', 14)
    merge_size = getattr(image_proc_config, 'merge_size', 2)
    
    min_pixels = MIN_PIXELS
    max_pixels = current_max_pixels
    
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    print(f"[DEBUG] Image resize: {pil_image.width}x{pil_image.height} -> {resized_width}x{resized_height}")
    return pil_image.resize(size=(resized_width, resized_height), resample=Image.Resampling.LANCZOS)


def run_inference(messages_for_template: List[dict[str, Any]], pil_image: Image.Image, 
                  max_new_tokens: int = 1024, thinking: bool = True) -> str:
    start_time = time.time()
    
    device = model.device
    text_prompt = processor.apply_chat_template(
        messages_for_template, 
        tokenize=False, 
        add_generation_prompt=True,
        thinking=thinking
    )
    
    prep_start = time.time()
    inputs = processor(text=[text_prompt], images=[pil_image], padding=True, return_tensors="pt").to(device)
    prep_time = time.time() - prep_start
    
    input_ids = inputs.input_ids
    num_input_tokens = input_ids.shape[1]
    print(f"[PERF] Input tokens: {num_input_tokens}, Prep time: {prep_time:.2f}s")
    
    gen_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            use_cache=True,
        )
    gen_time = time.time() - gen_start
    
    num_output_tokens = generated_ids.shape[1] - num_input_tokens
    tokens_per_sec = num_output_tokens / gen_time if gen_time > 0 else 0
    print(f"[PERF] Generated {num_output_tokens} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    decoded_output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    total_time = time.time() - start_time
    print(f"[PERF] Total inference time: {total_time:.2f}s")
    
    del inputs, input_ids, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    
    return decoded_output[0] if decoded_output else ""


def run_inference_streaming(
    messages_for_template: List[dict[str, Any]], 
    pil_image: Image.Image, 
    max_new_tokens: int = 1024, 
    thinking: bool = True
) -> Generator[str, None, None]:
    start_time = time.time()
    
    device = model.device
    text_prompt = processor.apply_chat_template(
        messages_for_template, 
        tokenize=False, 
        add_generation_prompt=True,
        thinking=thinking
    )
    
    inputs = processor(
        text=[text_prompt], 
        images=[pil_image], 
        padding=True, 
        return_tensors="pt"
    ).to(device)
    
    num_input_tokens = inputs.input_ids.shape[1]
    print(f"[PERF] Input tokens: {num_input_tokens}")
    
    streamer = TextIteratorStreamer(
        processor.tokenizer, 
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    token_count = 0
    for new_text in streamer:
        generated_text += new_text
        token_count += 1
        yield generated_text
    
    thread.join()
    
    total_time = time.time() - start_time
    tokens_per_sec = token_count / total_time if total_time > 0 else 0
    print(f"[PERF] Streamed {token_count} tokens in {total_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    del inputs
    torch.cuda.empty_cache()
