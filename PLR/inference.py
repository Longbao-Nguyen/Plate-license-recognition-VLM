import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
import os
import time

def run_inference():
    # ======================================================================
    # CẤU HÌNH ĐƯỜNG DẪN
    # ======================================================================
    base_model_path = "./model/qwen2-vl-2b-instruct"
    adapter_path = "./results_finetune/checkpoint-40"
    test_folder = "./dataset/test_image"

    # ======================================================================
    # LOAD MODEL + ADAPTER
    # ======================================================================
    print(f"--- Loading Base Model từ: {base_model_path} ---")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"--- Gộp LoRA Adapter từ: {adapter_path} ---")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.merge_and_unload()
    except Exception as e:
        print(f"❌ LỖI KHI LOAD ADAPTER: {e}")
        return

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    # ======================================================================
    # LẤY LIST ẢNH
    # ======================================================================
    image_files = [
        f for f in os.listdir(test_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("❌ Không tìm thấy ảnh nào trong thư mục test_image/")
        return

    print(f"\n--- Phát hiện {len(image_files)} ảnh cần inference ---\n")

    # ======================================================================
    # CHẠY INFERENCE
    # ======================================================================
    total_time = 0.0

    for idx, file_name in enumerate(image_files, 1):
        image_path = os.path.join(test_folder, file_name)

        print(f"({idx}/{len(image_files)}) Đang xử lý: {file_name}")

        image = Image.open(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Hãy đọc chính xác các ký tự có trên biển số xe trong ảnh này."},
                ],
            }
        ]

        # Build text prompt
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # --- Time measurement ---
        start_time = time.time()

        generated_ids = model.generate(**inputs, max_new_tokens=128)


        # Decode output
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        infer_time = time.time() - start_time
        total_time += infer_time
        
        print(f"   ➤ Kết quả: {output_text}")
        print(f"   ⏱️ Thời gian: {infer_time:.3f} giây\n")

    # ======================================================================
    # THỐNG KÊ THỜI GIAN
    # ======================================================================
    avg_time = total_time / len(image_files)

    print("=" * 50)
    print(f"⏱️ Tổng thời gian inference: {total_time:.3f} giây")
    print(f"⏱️ Thời gian trung bình mỗi ảnh: {avg_time:.3f} giây")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_inference()
