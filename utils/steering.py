from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sae_lens import SAE
import torch
from .testing import get_arc_easy_accuracy
from .dataset import get_tokenized_arc_easy_for_testing
from functools import partial

def get_lr_coefs(batch_size, tokenized_dataset, sae, model):
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
    y = np.array([1 if item["label"] == "True" else 0 for item in tokenized_dataset])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    lr = LogisticRegression(penalty="l1", solver="liblinear").fit(X_train, y_train)
    return lr

def find_max_activation(model, sae, tokenized_dataset, feature_idxs, batch_size):
    '''
    Find the maximum activation for a given feature index. This is useful for 
    calibrating the right amount of the feature to add.
    '''

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    max_activations = {feature_idx: 0. for feature_idx in feature_idxs}

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_tokens = batch["tokens"]
            _, cache = model.run_with_cache(batch_tokens, 
                                            stop_at_layer=sae.cfg.hook_layer + 1,
                                            names_filter=[sae.cfg.hook_name],
                                            prepend_bos=True)

            feature_acts = sae.encode(cache[sae.cfg.hook_name]).squeeze().flatten(0, 1)
            for feature_idx in feature_idxs:
                batch_max_activation = feature_acts[:, feature_idx].max().item()
                max_activations[feature_idx] = max(max_activations[feature_idx], batch_max_activation)

            del cache
    return max_activations

def steering(activations, hook, steering_strengths=[1.0], steering_vectors=[None], max_acts=[1.], pos_to_intervene=[None]):
    # Note if the feature fires anyway, we'd be adding to that here.
    for max_act, steering_strength, steering_vector in zip(max_acts, steering_strengths, steering_vectors):

        # bad conding style to take attention into account
        if len(activations.shape) == 4:
            for head in range(8):
                activations[np.arange(activations.shape[0]), pos_to_intervene, head, :] += max_act * steering_strength * steering_vector[head * 256: (head + 1) * 256]
        else:
            activations[np.arange(activations.shape[0]), pos_to_intervene, :] += max_act * steering_strength * steering_vector
    return activations

def generate_with_steering(model, sae, prompt, steering_features, max_acts, pos_to_intervene, steering_strengths, max_new_tokens=95):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    
    steering_vectors = [sae.W_dec[steering_feature].to(model.cfg.device) for steering_feature in steering_features]
    
    steering_hook = partial(
        steering,
        steering_vectors=steering_vectors,
        steering_strengths=steering_strengths,
        max_acts=max_acts,
        pos_to_intervene=pos_to_intervene
    )
    
    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos = True,
            prepend_bos = sae.cfg.prepend_bos,
        )
    
    return model.tokenizer.decode(output[0])

def get_arc_easy_accuracy_with_steering(model, sae, batch_size, dataset_split, steering_vectors, max_acts, steering_strengths):

    steering_vectors = [steering_vector.to(model.cfg.device) for steering_vector in steering_vectors]
    
    tokenized_dataset = get_tokenized_arc_easy_for_testing(model, sae.cfg.context_size, dataset_split, batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    sum_logit_diff = 0
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            batch_tokens = batch["tokens"]

            steering_hook = partial(
                steering,
                steering_vectors=steering_vectors,
                steering_strengths=steering_strengths,
                max_acts=max_acts,
                pos_to_intervene=[-1 for i in range(batch_tokens.shape[0])] # padding on the left
            )

            with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
                logits = model(batch_tokens, return_type="logits").cpu()
            correct_logits = [model.tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in batch["correct_label"]]
            incorrect_logits = [
                    [
                        model.tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in range(num_labels) if label != correct_label
                    ]
                    for correct_label, num_labels in zip(batch["correct_label"], batch["num_labels"])
                ]
            incorrect_logits_probs = [np.array([logits[batch_i, -1, incorrect_id] for incorrect_id in incorrect_logits[batch_i]]) for batch_i in range(len(incorrect_logits))]
            max_incorrect_logit = [incorrect_logits[batch_i][np.argmax(incorrect_logits_probs[batch_i])] for batch_i in range(len(incorrect_logits))]
            logit_diff = [logits[batch_i, -1, correct_logits[batch_i]] - logits[batch_i, -1, max_incorrect_logit[batch_i]] for batch_i in range(len(incorrect_logits))]
            
            correct += (np.array(logit_diff) > 0).sum()
            sum_logit_diff += (np.array(logit_diff)).sum()
    return correct / len(tokenized_dataset), sum_logit_diff / len(tokenized_dataset)

def get_arc_easy_accuracy_with_steering_multiple_hooks(model, hook_points, sae_context_size, batch_size, dataset_split, steering_vectors, steering_strengths):

    steering_vectors = [steering_vector.to(model.cfg.device) for steering_vector in steering_vectors]
    
    tokenized_dataset = get_tokenized_arc_easy_for_testing(model, sae_context_size, dataset_split, batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    sum_logit_diff = 0
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            batch_tokens = batch["tokens"]

            steering_hooks = []
            for hook_point, steering_vector, steering_strength in zip(hook_points, steering_vectors, steering_strengths):
                steering_hook = partial(
                    steering,
                    steering_vectors=[steering_vector],
                    steering_strengths=[steering_strength],
                    max_acts=[1.],
                    pos_to_intervene=[-1 for i in range(batch_tokens.shape[0])] # padding on the left
                )
                steering_hooks.append((hook_point, steering_hook))

            with model.hooks(fwd_hooks=steering_hooks):
                logits = model(batch_tokens, return_type="logits").cpu()
            correct_logits = [model.tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in batch["correct_label"]]
            incorrect_logits = [
                    [
                        model.tokenizer.convert_tokens_to_ids(chr(ord("A") + label)) for label in range(num_labels) if label != correct_label
                    ]
                    for correct_label, num_labels in zip(batch["correct_label"], batch["num_labels"])
                ]
            incorrect_logits_probs = [np.array([logits[batch_i, -1, incorrect_id] for incorrect_id in incorrect_logits[batch_i]]) for batch_i in range(len(incorrect_logits))]
            max_incorrect_logit = [incorrect_logits[batch_i][np.argmax(incorrect_logits_probs[batch_i])] for batch_i in range(len(incorrect_logits))]
            logit_diff = [logits[batch_i, -1, correct_logits[batch_i]] - logits[batch_i, -1, max_incorrect_logit[batch_i]] for batch_i in range(len(incorrect_logits))]
            
            correct += (np.array(logit_diff) > 0).sum()
            sum_logit_diff += (np.array(logit_diff)).sum()
    return correct / len(tokenized_dataset), sum_logit_diff / len(tokenized_dataset)