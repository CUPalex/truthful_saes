from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device = device)

from tqdm import tqdm
import tasks.truthfulqa
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from sae_lens import SAE

steering_directions = []
stds = []
hook_names = []

task = tasks.truthfulqa.TruthfulQA("probing")
tokenized_dataset = task.get_tokenized_dataset(tokenizer=model.tokenizer, batch_size=2, subset=False,
                                                random_seed=42, subset_len=81, max_length=1000)

layers = [l for l in range(26)]

for layer in tqdm(layers):
    sae, cfg_dict, _ = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-att-canonical",
        sae_id = f"layer_{layer}/width_16k/canonical",
        device = device
    )

    hook_point = sae.cfg.hook_name
    hook_names.append(hook_point)
    sae.eval()
    correct_activations = []

    dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_tokens = batch["tokens"]
            _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            correct_activations.append(feature_acts[np.arange(batch_tokens.shape[0]), batch["len_of_input"] - 1, :].detach().cpu())
            del cache

    correct_activations_dataset = torch.vstack(correct_activations)
    X = correct_activations_dataset.numpy()
    y = np.array([1 if item["label"] == "True" else -1 for item in tokenized_dataset])
    X_correct = X[y == 1]
    X_incorrect = X[y == -1]
    steering_direction = (X_correct.mean(axis=0) - X_incorrect.mean(axis=0))
    direction_norm = np.linalg.norm(steering_direction)
    std = np.std(np.dot(X, steering_direction) / direction_norm)
    steering_directions.append(steering_direction)
    stds.append(std)

import json
with open("cache/steering_in_sae_space/steering_directions.json", "w") as file:
    json.dump({str(layer): direction.tolist() for layer, direction in zip(layers, steering_directions)}, file)
with open("cache/steering_in_sae_space/stds.json", "w") as file:
    json.dump({str(layer): std.item() for layer, std in zip(layers, stds)}, file)
