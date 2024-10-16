import torch
import transformers   

k = 1
alpha = 10

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

import json
import numpy as np
with open("cache/accuracies_sorted_gemma.json", "r") as file:
    accuracies_sorted = json.load(file)
with open("cache/steering_directions_gemma_full_layer.json", "r") as file:
    steering_directions_json = json.load(file)
steering_directions = {}
for item in steering_directions_json:
    steering_directions[eval(item)] = np.array(steering_directions_json[item])
with open("cache/stds_gemma_full_layer.json", "r") as file:
    stds_json = json.load(file)
stds = {}
for item in stds_json:
    stds[eval(item)] = stds_json[item]

accuracies_by_layer = {layer: np.mean([p[0] for p in accuracies_sorted if p[1] == layer]) for layer in range(26)}
accuracies_by_layer_sorted = sorted([[acc, l] for l, acc in accuracies_by_layer.items()])

import tasks.truthfulqa
task = tasks.truthfulqa.TruthfulQA("testing")
tokenized_dataset = task.get_tokenized_dataset(tokenizer=tokenizer, batch_size=16, subset=False, random_seed=42, subset_len=100, max_length=1000)
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch
dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
correct = 0
sum_logit_diff = 0
positions_to_steer = [(t[1], h, -1) for t in accuracies_by_layer_sorted[:k] for h in range(8)]
vectors = [alpha * steering_directions[pos[0]][pos[1] * 256:(pos[1] + 1) * 256] * stds[pos[0]] for pos in positions_to_steer]
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