import transformers    

model = transformers.AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    use_auth_token=True,
    device_map="auto",
    offload_folder="./offload",
    attn_implementation="eager"
)

tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b")
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

from models.gemma_2 import Gemma

hooked_model = Gemma(model, tokenizer)

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch
import tasks.truthfulqa
from collections import defaultdict


task = tasks.truthfulqa.TruthfulQA("testing")
tokenized_datasets = []
for seed in [42, 0, 5, 10, 15]:
    tokenized_dataset = task.get_tokenized_dataset(tokenizer=tokenizer, batch_size=16, subset=True, testing_split=False,
                                                random_seed=seed, subset_len=81, max_length=1000)
    tokenized_datasets.append(tokenized_dataset)

accs = []
logit_diffs = []
for dataset_i in range(len(tokenized_datasets)):
    dataloader = DataLoader(tokenized_datasets[dataset_i], batch_size=16, shuffle=False)
    correct = 0
    sum_logit_diff = 0
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            batch_tokens = batch["tokens"].cuda()
            
            logits = hooked_model.model(batch_tokens).logits
            correct_logits = [tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in batch["correct_label"]]
            incorrect_logits = [
                    [
                        tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in range(num_labels) if label != correct_label
                    ]
                    for correct_label, num_labels in zip(batch["correct_label"], batch["num_labels"])
                ]
            incorrect_logits_probs = [torch.tensor([logits[batch_i, -1, incorrect_id] for incorrect_id in incorrect_logits[batch_i]]) for batch_i in range(len(incorrect_logits))]
            max_incorrect_logit = [incorrect_logits[batch_i][torch.argmax(incorrect_logits_probs[batch_i])] for batch_i in range(len(incorrect_logits))]
            logit_diff = [logits[batch_i, -1, correct_logits[batch_i]] - logits[batch_i, -1, max_incorrect_logit[batch_i]] for batch_i in range(len(incorrect_logits))]
        
            correct += (torch.tensor(logit_diff) > 0).sum()
            sum_logit_diff += (torch.tensor(logit_diff)).sum()
    accs.append(correct / len(tokenized_dataset))
    logit_diffs.append(sum_logit_diff / len(tokenized_dataset))
    print(dataset_i, accs)
import json
with open("cache/accs_by_k_alpha_gemma_full_layer_on_probing_dataset_no_intervention.json", "w") as file:
    json.dump(str(accs), file)