import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from transformers import GPT2Tokenizer,TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer
from datasets import load_dataset, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class CustomDPOTrainer(DPOTrainer):

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )[0]

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "./model/ProtGPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,
                                            low_cpu_mem_usage=True).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name,
                                            low_cpu_mem_usage=True).to(device)
file_path = "./KPO_data/KPO_prune_protein_output_ProtGPT2.json"
dataset = load_dataset("json", data_files=file_path)


# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./checkpoints/ProtGPT2/KPO_prune",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=100,
    save_safetensors=False,
    learning_rate=5e-5,
    logging_dir="./logs",
    remove_unused_columns=False,
    save_total_limit=2,
)

dpo_trainer = CustomDPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    args=training_args
)

dpo_trainer.train()