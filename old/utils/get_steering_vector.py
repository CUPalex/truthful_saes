from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from sae_lens import SAE

def get_steering_vector_lr_and_mean_diff(model, sae, tokenized_dataset, batch_size):
    correct_activations = []

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

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
    X_difference = (X_correct.mean(axis=0) - X_incorrect.mean(axis=0))

    lr = LogisticRegression(penalty="l1", solver="liblinear").fit(X, y)

    return X_difference, lr


def get_steering_vectors_lr_and_mean_diff(model, layers_to_invade, device, tokenized_dataset, batch_size=8, type_of_activation="att"):
    Xs_difference = []
    lrs = []
    hook_names = []

    for layer in tqdm(layers_to_invade):
        sae, cfg_dict, _ = SAE.from_pretrained(
            release = f"gemma-scope-2b-pt-{type_of_activation}-canonical",
            sae_id = f"layer_{layer}/width_16k/canonical",
            device = device
        )

        hook_point = sae.cfg.hook_name
        hook_names.append(hook_point)

        sae.eval()
        difference_layer, lr_layer = get_steering_vector_lr_and_mean_diff(model, sae, tokenized_dataset, batch_size)
        
        Xs_difference.append(difference_layer)
        lrs.append(lr_layer)
    return Xs_difference, lrs, hook_names
