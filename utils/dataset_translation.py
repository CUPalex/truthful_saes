from datasets import load_dataset
import numpy as np
from functools import partial

def format_examples_probing(examples) -> dict[str, list]:
    examples_formatted = {"sent": [],
                        "label": []}
    for example in examples["translation"]:
        examples_formatted["sent"].append(example["ru"])
        examples_formatted["label"].append("False")
        examples_formatted["sent"].append(example["en"])
        examples_formatted["label"].append("True")
    return examples_formatted

def get_probing_dataset_translation(batch_size=8, dataset_split="validation", len_of_dataset=1000):
    dataset = load_dataset(
        "wmt/wmt19", "ru-en",
        split=dataset_split,
        streaming=False,
    )
    probing_dataset = dataset.map(format_examples_probing, batched=True,
                    batch_size=batch_size, remove_columns=dataset.column_names)

    probing_dataset = probing_dataset.shuffle(seed=42).select(range(len_of_dataset))
    print(len(probing_dataset), len(probing_dataset.filter(lambda example: example["label"] == "True")))
    return probing_dataset

def tokenize_probing(examples, column_name, tokenizer, max_length):
    tokenizer.padding_side = "right"
    text = examples[column_name]
    tokens = tokenizer(text, return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
    len_of_input = np.argmax(tokens == tokenizer.pad_token_id, axis=1)
    assert (tokens[len_of_input == 0] != tokenizer.pad_token_id).all(), (len_of_input, tokens)
    len_of_input[len_of_input == 0] = tokens.shape[1]
    return {"tokens": tokens, "len_of_input": len_of_input}

def get_tokenized_probing_translation_dataset(model, sae_context_size, batch_size=8, dataset_split="validation", len_of_dataset=1000):
    probing_dataset = get_probing_dataset_translation(batch_size=batch_size,
                                                      dataset_split=dataset_split,
                                                      len_of_dataset=len_of_dataset)
    tokenized_dataset = probing_dataset.map(
        partial(tokenize_probing,
        column_name = "sent",
        tokenizer = model.tokenizer,
        max_length=sae_context_size),
        batched=True,
        batch_size=8,
        num_proc=None
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens", "label", "len_of_input", "sent"])
    return tokenized_dataset

def format_examples_testing(examples) -> dict[str, list]:
    examples_formatted = {"ru": [],
                          "en": []}
    for example in examples["translation"]:
        examples_formatted["ru"].append("Translate into english: "+ example["ru"])
        examples_formatted["en"].append(example["en"])
    return examples_formatted

def get_testing_dataset_translation(batch_size=8, dataset_split="validation", len_of_dataset=1000):
    dataset = load_dataset(
        "wmt/wmt19", "ru-en",
        split=dataset_split,
        streaming=False,
    )
    testing_dataset = dataset.map(format_examples_testing, batched=True,
                    batch_size=batch_size, remove_columns=dataset.column_names)

    testing_dataset = testing_dataset.shuffle(seed=42).select(range(len_of_dataset))
    return testing_dataset

def tokenize_testing(examples, tokenizer, max_length):
    tokenizer.padding_side = "left"
    ru_tokens = tokenizer(examples["ru"], return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
    en_tokens = tokenizer(examples["en"], return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
    return {"ru_tokens": ru_tokens, "en_tokens": en_tokens}

def get_tokenized_testing_translation_dataset(model, sae_context_size, batch_size=8, dataset_split="validation", len_of_dataset=1000):
    testing_dataset = get_testing_dataset_translation(batch_size=batch_size, dataset_split=dataset_split, len_of_dataset=len_of_dataset)
    tokenized_dataset = testing_dataset.map(
        partial(tokenize_testing,
        tokenizer = model.tokenizer,
        max_length=sae_context_size),
        batched=True,
        batch_size=batch_size,
        num_proc=None
    )
    tokenized_dataset.set_format(type="torch", columns=["ru_tokens", "en_tokens", "ru", "en"])
    return tokenized_dataset