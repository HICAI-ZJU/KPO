import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from Bio import pairwise2
from Bio.Align import substitution_matrices

def is_amino_acid_sequence(sequence):
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(residue in valid_amino_acids for residue in sequence)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prompts = [
    '<|endoftext|>'
]

model = AutoModelForCausalLM.from_pretrained("./checkpoints/prot_gpt2/KPO_prune")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("./model/ProtGPT2")
tokenizer .pad_token = tokenizer .eos_token
gen_kwargs = {
    "top_k": 950,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

def generate_sentences(prompts, num_sentences=100, max_length=1000):
    generated_sentences = []

    for _ in range(num_sentences):
        selected_prompt = random.choice(prompts)

        inputs = tokenizer.encode(selected_prompt, return_tensors="pt").to(device)

        outputs = model.generate(inputs, max_length=max_length, **gen_kwargs)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_sentences.append(generated_text)

    return generated_sentences


all_generated_sentences = generate_sentences(prompts,1100)
clean_texts = all_generated_sentences
clean_texts = [item for item in clean_texts if item != ""]
clean_texts = clean_texts[:1000]

data = {
    'sequence': clean_texts,
    'label': [0] * len(clean_texts)
}

df = pd.DataFrame(data)

df.to_excel('./res/KPO/protgpt2_KPO.xlsx', index=False)


