import fitz 
import time
from transformers import pipeline
import torch
import requests
import json

import os
import re
from groq import Groq, RateLimitError

import easyocr
import pytesseract

from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer
from huggingface_hub import InferenceClient

from PIL import Image

import pytesseract
from PIL import Image
# import easyocr  # Uncomment if you plan to use easyocr
# from transformers import AutoTokenizer, AutoModel  # Uncomment if you plan to use transformers

_tesseract_ocr_reader = None
_easyocr_reader = None
_ocr_tokenizer = None
_ocr_model = None
_llama_pipe = None

def get_tesseract_ocr_reader():
    global _tesseract_ocr_reader
    if _tesseract_ocr_reader is None:
        # Initialize pytesseract with the desired language
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        _tesseract_ocr_reader = pytesseract.pytesseract
    return _tesseract_ocr_reader

def call_tesseract_ocr_model(tmp_path):
    ocr_reader = get_tesseract_ocr_reader()

    # Open the image file
    image = Image.open(tmp_path)

    # Perform OCR on the image
    tessdata_dir_config = "--tessdata-dir " + r"C:\Program Files\Tesseract-OCR\tessdata"
    extracted_text = ocr_reader.image_to_string(image)  # You can add more languages if needed

    return extracted_text

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['de'])  # You can add more languages if needed
    return _easyocr_reader

def call_easyocr_model(tmp_path):
    ocr_reader = get_easyocr_reader()

    # Perform OCR on the image
    result = ocr_reader.readtext(tmp_path)

    # Extract text from the OCR result
    extracted_text = [text for (bbox, text, prob) in result]
    extracted_text = " ".join(extracted_text)

    return extracted_text

def get_ocr_tokenizer():
    global _ocr_tokenizer
    if _ocr_tokenizer is None:
        _ocr_tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    return _ocr_tokenizer

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        tokenizer = get_ocr_tokenizer()
        _ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        _ocr_model = _ocr_model.eval().cuda()
    return _ocr_model

def call_transformer_ocr_model(tmp_path):
    ocr_tokenizer = get_ocr_tokenizer()
    ocr_model = get_ocr_model()

    # plain texts OCR
    result = ocr_model.chat(ocr_tokenizer, tmp_path, ocr_type='ocr')
    return result

def get_llama_pipeline():
    global _llama_pipe
    if _llama_pipe is None:
        _llama_pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
            device_map="auto",
        )
    return _llama_pipe


def call_llama_locally(user_prompt, system_prompt) -> str:
    pipe = get_llama_pipeline()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hier der Inhalt aus dem Dokument: {user_prompt}"},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=4096,
    )

    return outputs[0]["generated_text"][-1]


def call_llama_hf(system_prompt, user_prompt):
    client = InferenceClient(
        "meta-llama/Llama-3.1-70B-Instruct",
        token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Hier der Inhalt aus dem Dokument: {user_prompt}"},
                ],
                max_tokens=4096,
                stream=False,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Error occurred: {e}. Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"Max retries reached. Last error: {e}")
                raise

def call_llama_groq(system_prompt, user_prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    model_list = [
        "llama-3.1-70b-versatile",
        "llama-3.2-11b-text-preview",
        "llama-3.2-90b-text-preview",
        #"llama-3.1-8b-instant",
    ]

    for model in model_list:
        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                    temperature=0.0, 
                    max_tokens=8000
                )
                return chat_completion.choices[0].message.content
            except RateLimitError as e:
                error_message = e.args[0]
                match = re.search(r'Please try again in (\d+m\d+\.\d+s)', error_message)
                if match:
                    wait_time_str = match.group(1)
                    wait_time = parse_wait_time(wait_time_str)
                    if wait_time > 30:  # 30 seconds
                        print(f"Rate limit reached with {model}, wait time longer than 30 seconds. Trying next model...")
                        break#
                    else:
                        print(f"Rate limit reached. Waiting for {wait_time_str}...")
                        time.sleep(wait_time)
                else:
                    print("Rate limit reached but could not parse wait time. Retrying in 1 minute...")
                    time.sleep(60)  # Wait for 1 minute if parsing fails

    # If all models have been tried and rate limited, call the other API
    print("All models rate limited. Calling model from HuggingFace remotely...")
    return call_llama_hf(system_prompt, user_prompt)

def parse_wait_time(wait_time_str):
    minutes, seconds = 0, 0
    if 'm' in wait_time_str:
        minutes = int(wait_time_str.split('m')[0])
        wait_time_str = wait_time_str.split('m')[1]
    if 's' in wait_time_str:
        seconds = float(wait_time_str.split('s')[0])
    return minutes * 60 + seconds

def call_mistral_remote(system_prompt, user_prompt):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {os.getenv("MISTRAL_API_KEY")}'
    }

    payload = {
        "model": "mistral-large-latest",
        "messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Hier der Inhalt aus dem Dokument: {user_prompt}"},
        ],
        "max_tokens": 16000,
        "temperature": 0.0
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, data=json.dumps(payload))
    response.raise_for_status()  # Raise an exception for HTTP errors
    response_object = response.json()
    answer = response_object["choices"][0]["message"]["content"]
    return answer    


def save_progress(progress_file, page_num, ocr_results, cleaned_pages):
    with open(progress_file, 'w') as f:
        json.dump({
            'last_page': page_num,
            'ocr_results': ocr_results,
            'cleaned_pages': cleaned_pages
        }, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def processing_pipeline(pdf_path, progress_file, system_prompt, mode='ocr', page_range=None, verbose=False, start_page=0):
    if '-PARTIAL' in progress_file:
        final_progress_file = progress_file.replace('-PARTIAL', '-DONE')
    else:
        final_progress_file = progress_file

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_document = fitz.open(pdf_path)

    progress = load_progress(progress_file)
    if progress:
        last_page = progress['last_page']
        ocr_results = progress['ocr_results']
        cleaned_pages = progress['cleaned_pages']
        start_page = last_page + 1 if start_page == 0 else start_page
    else:
        last_page = -1
        ocr_results = []
        cleaned_pages = []
        start_page = 0

    if page_range:
        start_page, end_page = page_range
        pages_to_process = range(max(start_page, last_page + 1), min(end_page, len(pdf_document)) + 1)
    else:
        pages_to_process = range(max(start_page, last_page + 1), len(pdf_document))

    for page_num in tqdm(pages_to_process):
        page = pdf_document.load_page(page_num)

        if mode == 'ocr':
            #print("using ocr")
            pix = page.get_pixmap(dpi=300)

            # Convert to PIL image and resize
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.resize((img.width // 2, img.height // 2))  # Resize to half the original size

            tmp_path = f"./tmp/tmp_page_{page_num + 1}.jpg"
            img.save(tmp_path, compression_level=2)

            ocr_result = call_tesseract_ocr_model(tmp_path)
            ocr_results.append(ocr_result)

            if verbose:
                print(f"Page {page_num} result: {ocr_result} \n\n --- \n\n")

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        elif mode == 'raw':
            #print("getting text")
            ocr_result = page.get_text("text")  # Extract raw text from the page
            ocr_results.append(ocr_result)

            if verbose:
                print(f"Page {page_num} raw text: {ocr_result} \n\n --- \n\n")

        #print("calling model")
        cleaned_result = call_llama_groq(system_prompt=system_prompt, user_prompt=ocr_result)
        cleaned_pages.append(cleaned_result)

        if verbose:
            print(f"Cleaned {page_num} result: {cleaned_result} \n\n --- \n\n")

        # Save progress after each page
        save_progress(progress_file, page_num, ocr_results, cleaned_pages)

    # Save final progress
    save_progress(final_progress_file, page_num, ocr_results, cleaned_pages)
    os.remove(progress_file)

    return ocr_results, cleaned_pages

def process_pdfs(data_folder, pdfs_folder, system_prompt, mode='ocr', page_range=None, verbose=False):
    # Get a list of PDF files in the pdfs_folder
    pdf_files = [f for f in os.listdir(pdfs_folder) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdfs_folder, pdf_file)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        progress_file = os.path.join(data_folder, f"{base_name}-DONE.json")
        partial_progress_file = os.path.join(data_folder, f"{base_name}-PARTIAL.json")

        if os.path.exists(progress_file):
            print(f"Skipping {pdf_file} as it has already been processed.")
            continue

        processing_pipeline(pdf_path, partial_progress_file, system_prompt, mode, page_range, verbose)