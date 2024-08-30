from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from sae_lens import SAE

def get_accuracies_per_layer(model, tokenized_dataset, type_of_activation="res", batch_size=8, device="cuda")
    accuracies = []
    num_of_features = []
    for layer in range(0, 25):
        sae, cfg_dict, _ = SAE.from_pretrained(
            release = f"gemma-scope-2b-pt-{type_of_activation}-canonical",
            sae_id = f"layer_{layer}/width_16k/canonical",
            device = device
        )

        # get hook point
        hook_point = sae.cfg.hook_name

        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        lr = LogisticRegression(penalty="l1", solver="liblinear").fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        num_of_features.append((lr.coef_ != 0).sum())
    return accuracies, num_of_features