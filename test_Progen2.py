import re
import numpy as np
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prompts = ["1"]

mode_name = "./checkpoints/Progen2/KPO_prune"
model = AutoModelForCausalLM.from_pretrained(mode_name,trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(mode_name,trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    print("Padding token added:", tokenizer.pad_token)

def clean_generated_sequence(generated_text, stop_token="2"):

    stop_index = generated_text.find(stop_token)
    if stop_index != -1:
        generated_text = generated_text[:stop_index]

    return generated_text

def generate_sentences(prompts, num_sentences=1010, max_length=500, batch_size=8):
    generated_sentences = []
    clean_generated_sentences = []

    with tqdm(total=num_sentences, desc="Generating sequences") as pbar:
        while len(generated_sentences) < num_sentences:
            batch_prompts = [random.choice(prompts) for _ in range(batch_size)]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

            outputs = model.generate(inputs['input_ids'], max_length=max_length, do_sample=True, temperature=0.9, pad_token_id=tokenizer.pad_token_id)
            generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for idx, generated_text in enumerate(generated_batch):
                generated_part = generated_text[len(batch_prompts[idx]):].strip()
                if generated_part:
                    generated_sentences.append(generated_part)
                    cleaned_sequence = clean_generated_sequence(generated_part)
                    clean_generated_sentences.append(cleaned_sequence)
                    pbar.update(1)
                if len(generated_sentences) >= num_sentences:
                    break
    return generated_sentences[:num_sentences], clean_generated_sentences[:num_sentences]

all_generated_sentences, clean_generated_sequences = generate_sentences(prompts, num_sentences=1000, batch_size=2)

data_cleaned = {'sequence': clean_generated_sequences, 'label': [0] * len(clean_generated_sequences)}
df_cleaned = pd.DataFrame(data_cleaned)
df_cleaned.to_excel('./res/KPO/progen2.xlsx', index=False)
