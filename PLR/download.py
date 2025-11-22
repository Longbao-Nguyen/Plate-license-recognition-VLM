"""
Script để tải Qwen2-VL model về máy local
Chạy script này trên máy có internet trước khi chuyển sang HPC
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os

# Định nghĩa model ID và folder lưu
model_id = 'Qwen/Qwen2-VL-2B-Instruct'
save_folder = './model/qwen2-vl-2b-instruct'

# Tạo folder nếu chưa tồn tại
os.makedirs(save_folder, exist_ok=True)

print(f"Đang tải model: {model_id}")
print(f"Sẽ lưu vào: {save_folder}")
print("=" * 60)

# Tải processor
print("\n[1/2] Đang tải Processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)
processor.save_pretrained(save_folder)
print("✓ Processor đã được tải và lưu")

# Tải model
print("\n[2/2] Đang tải Model (có thể mất vài phút)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    trust_remote_code=True
)
model.save_pretrained(save_folder)
print("✓ Model đã được tải và lưu")

print("\n" + "=" * 60)
print("✅ Hoàn tất! Model đã được tải xuống thành công!")
print(f"📁 Vị trí: {save_folder}")
print("\nBạn có thể copy folder này sang HPC và load bằng:")
print(f"  processor = AutoProcessor.from_pretrained('{save_folder}')")
print(f"  model = Qwen2VLForConditionalGeneration.from_pretrained('{save_folder}')")
print("=" * 60)