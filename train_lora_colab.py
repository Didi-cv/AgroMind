# 这是专门在 Google Colab 上运行的 LoRA 微调脚本
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import os

# --- 1. 检查环境 ---
print("🚀 开始LoRA微调验证实验...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 当前使用设备: {device}")

model_id = "openbmb/MiniCPM-V-4.6"
save_path = "/content/drive/MyDrive/agromind-lora"

# --- 2. 加载模型和处理器 ---
print(f"📦 正在加载模型: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# --- 3. 配置 LoRA ---
print("⚙️ 正在配置 LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ LoRA微调环境已就绪！模型可以开始训练。")
