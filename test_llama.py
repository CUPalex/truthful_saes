import transformers
import torch  

k = 80
alpha = 30

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_auth_token=True,
    device_map="auto",
    offload_folder="./offload"
)

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

from models.llama_7b import Llama

hooked_model = Llama(model, tokenizer)

import numpy as np
import json
with open("cache/accuracies_sorted.json", "r") as file:
    accuracies_sorted = json.load(file)
with open("cache/steering_directions.json", "r") as file:
    steering_directions_json = json.load(file)
steering_directions = {}
for item in steering_directions_json:
    steering_directions[eval(item)] = np.array(steering_directions_json[item])
with open("cache/stds.json", "r") as file:
    stds_json = json.load(file)
stds = {}
for item in stds_json:
    stds[eval(item)] = stds_json[item]

import tasks.truthfulqa
task = tasks.truthfulqa.TruthfulQA("testing")
tokenized_dataset = task.get_tokenized_dataset(tokenizer=tokenizer, batch_size=2, subset=False, random_seed=42, subset_len=100, max_length=1000)
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch
dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=False)
correct = 0
sum_logit_diff = 0
positions_to_steer = [(t[1], t[2], -1) for t in accuracies_sorted[:k]]
vectors = [alpha * steering_directions[(pos[0], pos[1])] * stds[(pos[0], pos[1])] for pos in positions_to_steer]
hooked_model.set_steering_vectors(vectors, positions_to_steer)
with torch.no_grad():
    for iter, batch in tqdm(enumerate(dataloader)):
        batch_tokens = batch["tokens"].cuda()
        
        logits = hooked_model.model(batch_tokens).logits.cpu()
        correct_logits = [tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in batch["correct_label"]]
        incorrect_logits = [
                [
                    tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in range(num_labels) if label != correct_label
                ]
                for correct_label, num_labels in zip(batch["correct_label"], batch["num_labels"])
            ]
        incorrect_logits_probs = [np.array([logits[batch_i, -1, incorrect_id] for incorrect_id in incorrect_logits[batch_i]]) for batch_i in range(len(incorrect_logits))]
        max_incorrect_logit = [incorrect_logits[batch_i][np.argmax(incorrect_logits_probs[batch_i])] for batch_i in range(len(incorrect_logits))]
        logit_diff = [logits[batch_i, -1, correct_logits[batch_i]] - logits[batch_i, -1, max_incorrect_logit[batch_i]] for batch_i in range(len(incorrect_logits))]
        
        correct += (np.array(logit_diff) > 0).sum()
        sum_logit_diff += (np.array(logit_diff)).sum()
print(k, alpha, correct / len(tokenized_dataset), sum_logit_diff / len(tokenized_dataset))
