from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login

import os

hf_token = os.get_env("HUGGINGFACE_API_KEY")
login(token=hf_token)

# wandb.login(key=wb_token)
# run = wandb.init(
#     project='Fine-tune Llama 3.2 on Customer Support Dataset', 
#     job_type="training", 
#     anonymous="allow"
# )

# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Load model
base_model = "unsloth/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)