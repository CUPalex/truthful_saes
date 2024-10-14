import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.cuda.device_count()

from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

device = "cuda" if torch.cuda.is_available() else "cpu"

# get model
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device = device)

layer = 15

# get the SAE for this layer
sae, cfg_dict, _ = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-att-canonical",
    sae_id = f"layer_{layer}/width_16k/canonical",
    device = device
)

# get hook point
hook_point = sae.cfg.hook_name
print(hook_point)

from utils.get_steering_vector import get_steering_vectors_lr_and_mean_diff
from utils.dataset import get_tokenized_arc_easy_for_probing

res_layers_to_invade = [9, 10, 11, 13, 15]

tokenized_dataset = get_tokenized_arc_easy_for_probing(model=model, sae_context_size=sae.cfg.context_size,
                                                       dataset_split="train", batch_size=8, select_items=500)
Xs_difference_res, lrs_res, hook_names_res = get_steering_vectors_lr_and_mean_diff(model=model, layers_to_invade=res_layers_to_invade,
                                                                       device="cuda",
                                                                       tokenized_dataset=tokenized_dataset, batch_size=8, type_of_activation="res")

from utils.steering import get_arc_easy_accuracy_with_steering_multiple_hooks
import numpy as np

evaluated = {}
for activation_vector, layer, hook_point in zip(Xs_difference_res, res_layers_to_invade, hook_names_res):
    if layer not in evaluated:
        evaluated[layer] = []
    for num_vectors in np.logspace(0, np.log10((Xs_difference_res[0] != 0).sum()), 10, base=10., dtype=int)[:7]:
        sae, cfg_dict, _ = SAE.from_pretrained(
            release = "gemma-scope-2b-pt-res-canonical",
            sae_id = f"layer_{layer}/width_16k/canonical",
            device = device
        )
        features = np.argsort(-activation_vector)[:num_vectors]
        coefs = -np.sort(-activation_vector)[:num_vectors]
        steering_vector = sum([sae.W_dec[feature] * coef for feature, coef in zip(features, coefs)])

        steering_strengths = [1.]
        res = get_arc_easy_accuracy_with_steering_multiple_hooks(model=model, hook_points=[hook_point], sae_context_size=1024,
                    batch_size=8, dataset_split="validation", steering_vectors=[steering_vector],
                    steering_strengths=steering_strengths)
        print(layer, num_vectors, res)
        evaluated[layer].append(res)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 1)
fig.set_size_inches((8, 4))
for layer in res_layers_to_invade:
    axs.plot([e[0] for e in evaluated[layer]], label=f"residual stream on layer {layer}")
axs.legend()
axs.set_xlabel("# features summed to create a steering vector")
axs.set_ylabel("accuracy")
labels = np.logspace(0, np.log10((Xs_difference_res[0] != 0).sum()), 10, base=10., dtype=int)[:7].tolist()

axs.set_xticks(np.arange(len(labels)), labels=labels)
fig.tight_layout()
plt.savefig("acc_by_num_of_features_on_different_layers_2.png")

import json

with open("acc_by_num_of_features_on_different_layers_2.json", "w") as file:
    json.dump(evaluated, file)