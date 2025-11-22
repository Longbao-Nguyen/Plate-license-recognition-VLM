import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data(data_folder):
    print(f"--- Đang load dataset từ {data_folder} ---")
    with open(os.path.join(data_folder, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_folder, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_folder, 'X_val.pkl'), 'rb') as f:
        X_val = pickle.load(f)
    with open(os.path.join(data_folder, 'y_val.pkl'), 'rb') as f:
        y_val = pickle.load(f)
    return X_train, y_train, X_val, y_val

# ============================================================================
# 2. DATASET & COLLATOR (ĐÃ FIX MASKING & LOGIC)
# ============================================================================

class LicensePlateQwenDataset(Dataset):
    def __init__(self, images, labels, processor):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.system_prompt = "Bạn là một AI hỗ trợ nhận diện biển số xe chính xác."
        self.user_instruction = "Hãy đọc chính xác các ký tự có trên biển số xe trong ảnh này."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        image = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
        label = self.labels[idx]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
        return messages

class DataCollatorForQwen2VL:
    """
    Collator FIX FINAL: Tìm kiếm chuỗi token ID của '<|im_start|>assistant\n'
    để mask chính xác phần prompt.
    """
    def __init__(self, processor):
        self.processor = processor
        self.ignore_index = -100
        # Dựa trên log của bạn: 151644=<|im_start|>, 77091=assistant, 198=\n
        self.assistant_start_ids = [151644, 77091, 198] 

    def __call__(self, batch):
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch
        ]
        image_inputs, video_inputs = process_vision_info(batch)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # --- LOGIC MASKING DỰA TRÊN TOKEN ID ---
        for i in range(len(labels)):
            ids_list = labels[i].tolist()
            
            # Tìm vị trí bắt đầu câu trả lời
            start_index = 0
            found = False
            
            # Quét qua chuỗi để tìm marker [151644, 77091, 198]
            # Chúng ta tìm lần xuất hiện cuối cùng (trong trường hợp few-shot, nhưng ở đây là zero-shot)
            for idx in range(len(ids_list) - 3):
                if (ids_list[idx] == 151644 and 
                    ids_list[idx+1] == 77091 and 
                    ids_list[idx+2] == 198):
                    
                    # Mask đến hết token 198 (\n). Câu trả lời bắt đầu từ idx + 3
                    start_index = idx + 3 
                    found = True
                    break # Tìm thấy rồi thì dừng (với zero-shot chat template)
            
            if found:
                # Mask toàn bộ phần trước câu trả lời
                labels[i, :start_index] = self.ignore_index
            else:
                print(f"Warning: Không tìm thấy assistant header trong mẫu {i}")
            
            # Mask padding tokens (quan trọng)
            labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = self.ignore_index
            
            # Mask image tokens (để giảm nhiễu loss)
            labels[i][labels[i] == 151655] = self.ignore_index

        inputs["labels"] = labels
        
        # # --- DEBUG CHECK (Chỉ in lần đầu) ---
        # if not hasattr(self, 'debug_done'):
        #     print("\n--- DEBUG LABELS SAU KHI FIX ---")
        #     # Lấy mẫu đầu tiên để check
        #     sample_label = labels[0]
        #     valid_tokens = sample_label[sample_label != -100]
        #     print(f"Original Length: {len(labels[0])}")
        #     print(f"Valid Tokens Length: {len(valid_tokens)}")
        #     print(f"Valid Tokens IDs: {valid_tokens}")
        #     print(f"Decoded Valid String: '{self.processor.decode(valid_tokens)}'")
        #     self.debug_done = True
            
        return inputs

# ============================================================================
# 3. TRAINING SETUP
# ============================================================================
def main():
    dataset_folder = "./dataset/processed_dataset"
    model_path = "./model/qwen2-vl-2b-instruct"
    output_dir = "./results_finetune"

    X_train, y_train, X_val, y_val = load_data(dataset_folder)

    # Load Processor
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    
    # Load Model Quantization
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("--- Loading Model... ---")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config, 
        device_map="auto",
    )
    
    # === FIX QUAN TRỌNG: KÍCH HOẠT GRADIENT CHO INPUT ===
    model = prepare_model_for_kbit_training(model)
    
    # Dòng này sửa lỗi "None of the inputs have requires_grad=True"
    model.enable_input_require_grads() 
    # ====================================================

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset & Collator
    train_dataset = LicensePlateQwenDataset(X_train, y_train, processor)
    val_dataset = LicensePlateQwenDataset(X_val, y_val, processor)
    data_collator = DataCollatorForQwen2VL(processor)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8, # Nếu GPU mạnh thì tăng lên 8
        gradient_accumulation_steps=8,
        learning_rate=2e-4,            # Tăng nhẹ LR để thoát khỏi vùng loss bão hòa
        num_train_epochs=3,
        logging_steps=5,               # Log thường xuyên hơn để theo dõi
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        bf16=True,                     # Bắt buộc dùng bf16 hoặc fp16
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False, # Tránh lỗi trong môi trường đa GPU (hoặc macbook)
        gradient_checkpointing=False,      # Phải khớp với model.gradient_checkpointing_enable()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("--- Bắt đầu Training (Đã fix Gradient) ---")
    trainer.train()
    
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    print("Training hoàn tất!")

if __name__ == "__main__":
    main()