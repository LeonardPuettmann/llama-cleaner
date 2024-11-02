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
from datasets import *

import os
import glob 
import json

hf_token = os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)

# wandb.login(key=wb_token)
# run = wandb.init(
#     project='Fine-tune Llama 3.2 on Customer Support Dataset', 
#     job_type="training", 
#     anonymous="allow"
# )

# # Set torch dtype and attention implementation
# if torch.cuda.get_device_capability()[0] >= 8:
#     torch_dtype = torch.bfloat16
#     attn_implementation = "flash_attention_2"
# else:
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


### 
# add data stuff here 
### 

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config) 

new_model = "llama-3.2-1b-it-finetuned"

# prep the dataset
def load_json_data(folder_path, key):
    data = []
    for file_path in glob.glob(folder_path + '/*.json'):
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data.extend(json_data[key])
    return data

raw_pages = load_json_data("../data_prep/data", key="ocr_results")
cleaned_pages = load_json_data("../data_prep/data", key="cleaned_pages")

test_size = 0.2
train_size = int(len(cleaned_pages) * (1 - test_size))
train_raw_pages, test_raw_pages = raw_pages[:train_size], raw_pages[train_size:]
train_cleaned_pages, test_cleaned_pages = cleaned_pages[:train_size], cleaned_pages[train_size:]

dataset = DatasetDict({
    'train': Dataset.from_dict({"raw_pages": train_raw_pages[:len(train_cleaned_pages)], "cleaned_pages": train_cleaned_pages}),
    'test': Dataset.from_dict({"raw_pages": test_raw_pages[:len(test_cleaned_pages)], "cleaned_pages": test_cleaned_pages})
})

base_model = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

instruction = """
Du bist ein Experte für Textbereinigung. Deine Aufgabe ist es, einen Eingabetext zu bereinigen, der aus einem PDF-Dokument extrahiert wurde. Der Inhalt ist immer nur von einer einzelnen Seite, es sollte also nicht zu viel Text auf einmal sein. Es ist sehr wichtig, dass keine Daten und Informationen verloren gehen und dass die Originaltexte in keiner Weise verändert werden!
Antworte ausschließlich in Deutsch und keiner anderen Sprache.

Du hast die folgenden Aufgaben:
- Entferne alle seltsamen Textteile und Sonderzeichen.
- Entferne alle unnötigen Leerzeichen und Zeilenumbrüche.
- Organisiere die Formatierung.
- Korrektur von Rechtschreibfehlern.
- Handling von Formatierungsfehlern.

Gib nur den bereinigten und formatierten Text zurück und nichts anderes! Füge keinen eigenen Text hinzu! Achte auf Vollständigkeit, es darf kein Inhalt verloren gehen und es muss alles 100 % vollständig sein!
"""

def format_chat_template(row):
    
    row_json = [{"role": "system", "content": instruction},
               {"role": "user", "content": row["raw_pages"]},
               {"role": "assistant", "content": row["cleaned_pages"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template
)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to=None,
    remove_unused_columns=False
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()
#wandb.finish()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
#trainer.model.push_to_hub(new_model, use_temp_dir=False)