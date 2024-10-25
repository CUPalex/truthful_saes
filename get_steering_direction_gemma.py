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
task = tasks.truthfulqa.TruthfulQA("probing")
tokenized_dataset = task.get_tokenized_dataset(tokenizer=tokenizer, batch_size=2, subset=False, random_seed=42, subset_len=100, max_length=1000)


steering_directions = {(layer, head): None for layer in range(26) for head in range(8)}
stds = {(layer, head): None for layer in range(26) for head in range(8)}

all_activations = {pos : [] for pos in steering_directions.keys()}

dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=False)

with torch.no_grad():
    for batch in tqdm(dataloader):
        batch_tokens = batch["tokens"].cuda()
        hooked_model.model(batch_tokens)

        for layer, head in steering_directions.keys():
            all_activations[(layer, head)].append(hooked_model.cache_activations[(layer, head)][
                np.arange(batch_tokens.shape[0]), batch["len_of_input"] - 1, :
                ].detach().cpu())

for layer, head in steering_directions.keys():
    all_activations_dataset = torch.vstack(all_activations[(layer, head)])
    X = all_activations_dataset.numpy()
    y = np.array([1 if item["label"] == "True" else -1 for item in tokenized_dataset])
    pos_activations = X[y == 1]
    neg_activations = X[y == -1]
    steering_directions[(layer, head)] = np.mean(pos_activations, axis=0) - np.mean(neg_activations, axis=0)
    direction_norm = np.linalg.norm(steering_directions[(layer, head)])
    stds[(layer, head)] = np.std(np.dot(X, steering_directions[(layer, head)]) / direction_norm)

import json
with open("cache/steering_directions_gemma.json", "w") as file:
    json.dump({str(item): steering_directions[item].tolist() for item in steering_directions}, file)
with open("cache/stds_gemma.json", "w") as file:
    json.dump({str(item): stds[item].item() for item in stds}, file)