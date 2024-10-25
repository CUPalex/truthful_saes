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


import tasks.truthfulqa
task = tasks.truthfulqa.TruthfulQA("probing")
tokenized_dataset = task.get_tokenized_dataset(tokenizer=tokenizer, batch_size=2, subset=False, random_seed=42, subset_len=100, max_length=1000)

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch

accuracies = []
correct_activations = {(layer, head) : []
                       for layer in range(len(hooked_model.model.model.layers))
                       for head in range(hooked_model.model.model.layers[0].self_attn.num_heads)}

dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=False)

with torch.no_grad():
    for batch in tqdm(dataloader):
        batch_tokens = batch["tokens"].cuda()
        hooked_model.model(batch_tokens)

        for layer in range(len(hooked_model.model.model.layers)):
            for head in range(hooked_model.model.model.layers[0].self_attn.num_heads):
                correct_activations[(layer, head)].append(hooked_model.cache_activations[(layer, head)][
                    np.arange(batch_tokens.shape[0]), batch["len_of_input"] - 1, :
                    ].detach().cpu())

for layer in range(len(hooked_model.model.model.layers)):
    for head in range(hooked_model.model.model.layers[0].self_attn.num_heads):
        correct_activations_dataset = torch.vstack(correct_activations[(layer, head)])
        X = correct_activations_dataset.numpy()
        y = np.array([1 if item["label"] == "True" else -1 for item in tokenized_dataset])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        lr = LogisticRegression(penalty="l1", solver="liblinear").fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

accuracies_sorted = []
i = 0
for layer in range(len(hooked_model.model.model.layers)):
    for head in range(hooked_model.model.model.layers[0].self_attn.num_heads):
        accuracies_sorted.append((-accuracies[i], layer, head))
        i += 1
accuracies_sorted = sorted(accuracies_sorted)

import json
with open("cache/accuracies_sorted_gemma.json", "w") as file:
    json.dump(accuracies_sorted, file)