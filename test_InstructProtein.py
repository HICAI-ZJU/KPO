import re
import numpy as np
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prompts = ["Instruction: I would like a protein that is non-toxic and harmless."]
model = AutoModelForCausalLM.from_pretrained("./checkpoints/InstructProtein/KPO_prune/checkpoint-1700").to(device)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/InstructProtein/KPO_prune/checkpoint-1700")
tokenizer.pad_token = tokenizer.eos_token

def clean_generated_sequence(generated_text, stop_token="</protein>"):
    generated_text = generated_text.replace("Ƥ", "")

    stop_index = generated_text.find(stop_token)
    if stop_index != -1:
        generated_text = generated_text[:stop_index]

    cleaned_sequence = re.findall(r'[ACDEFGHIKLMNPQRSTVWY]+', generated_text)

    return ''.join(cleaned_sequence)

def generate_sentences(prompts, num_sentences=1010, max_length=1000, batch_size=8):
    generated_sentences = []
    clean_generated_sentences = []
    with tqdm(total=num_sentences, desc="Generating sequences") as pbar:
        while len(generated_sentences) < num_sentences:

            batch_prompts = [prompts[0] for _ in range(batch_size)]
            inputs = tokenizer(batch_prompts,return_tensors="pt", padding=True).to(device)

            outputs = model.generate(inputs['input_ids'], do_sample=True, max_length=max_length, temperature=1.6, pad_token_id=tokenizer.eos_token_id)
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


all_generated_sentences, clean_generated_sequences = generate_sentences(prompts, 1010, batch_size=32)
# clean_texts = [item for item in all_generated_sentences if item != ""]
# clean_texts = clean_texts[:1000]
clean_texts_1 = [item for item in clean_generated_sequences if item != ""]
clean_texts_1 = clean_texts_1[:1000]

data = {'sequence': clean_texts_1, 'label': [0] * len(clean_texts_1)}
df_1 = pd.DataFrame(data)
df_1.to_excel('./res/KPO/instruct_KPO_prune.xlsx', index=False)




