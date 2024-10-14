from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch

from .dataset import get_tokenized_arc_easy_for_testing
from .truthfulqa import get_tokenized_truthful_qa_for_testing

def get_arc_easy_accuracy(model, sae_context_size, batch_size=8, dataset_split="train"):
    tokenized_dataset = get_tokenized_arc_easy_for_testing(model, sae_context_size, dataset_split, batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    sum_logit_diff = 0
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            batch_tokens = batch["tokens"]
            
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

def get_truthful_qa_accuracy(model, sae_context_size, batch_size=8):
    tokenized_dataset = get_tokenized_truthful_qa_for_testing(model, sae_context_size, batch_size)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    sum_logit_diff = 0
    with torch.no_grad():
        for iter, batch in tqdm(enumerate(dataloader)):
            batch_tokens = batch["tokens"]
            
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