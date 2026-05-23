from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

model_id = "openbmb/MiniCPM-V-4.6"

print("正在加载模型...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "假设你是一位农业专家。请用中文回答：如果一片番茄叶子上出现褐色斑点，边缘发黄，可能是什么病害？请给出3种可能性并按可能性排序。"}
        ]
    }
]

print("正在推理...")
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"模型回答：\n{output_text[0]}")
print("\n✅ 模型推理验证成功！")
