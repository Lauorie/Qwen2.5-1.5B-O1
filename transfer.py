# transfer .pth to .bin
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 1. 加载配置
config = AutoConfig.from_pretrained("/root/app/models/Qwen2.5-1.5B")

# 2. 创建模型实例
model = AutoModelForCausalLM.from_config(config)

# 3. 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/root/app/models/Qwen2.5-1.5B")

# 4. 添加 special token
new_special_token = "<|reasoning|>"
tokenizer.add_special_tokens({"additional_special_tokens": [new_special_token]})

# 5. 调整模型的 embedding 层大小以适应新增的 token
model.resize_token_embeddings(len(tokenizer))

# 6. 加载预训练权重 (.pth 文件)
checkpoint_path = "/root/app/checkpoints/Qwen2.5-1.5B_SFT_final.pth"
state_dict = torch.load(checkpoint_path)

# 7. 将权重加载到模型中
model.load_state_dict(state_dict)

# 8. 将模型转换为半精度以节省显存
model = model.half()

# 9. 保存转换后的模型和 tokenizer 到指定目录
save_directory = "./checkpoints"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer have been saved to {save_directory}")
