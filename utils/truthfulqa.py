from datasets import load_dataset
import numpy as np
from functools import partial

def format_examples(examples) -> dict[str, list]:
    examples_formatted = {"formatted_question": [],
                          "correct_label": [],
                          "num_labels": []}
    for example_question, example_choices in zip(examples["question"], examples["mc1_targets"]):
        labels = [label == 1 for label in example_choices["labels"]]
        permute = np.random.permutation(len(example_choices["choices"]))
        choices_permuted = [example_choices["choices"][i] for i in permute]
        ohe_labels_permuted = [labels[i] for i in permute]
        formatted_question = example_question + " Choose the correct answer.\n" + "\n".join(
            [chr(ord("A") + i) + ". " + choice for i, choice in enumerate(choices_permuted)]
        ) + "\nAnswer: "
        examples_formatted["formatted_question"].append(formatted_question)
        examples_formatted["correct_label"].append(np.argmax(ohe_labels_permuted))
        examples_formatted["num_labels"].append(len(example_choices["choices"]))
    return examples_formatted

def get_formatted_truthful_qa_for_testing():
    dataset = load_dataset(
        "truthfulqa/truthful_qa", "multiple_choice",
        split="validation",
        streaming=False,
    )
    formatted_dataset = dataset.map(format_examples, batched=True, batch_size=1)
    return formatted_dataset

def tokenize(examples, column_name, tokenizer, max_length):
        tokenizer.padding_side = "left"
        text = examples[column_name]
        tokens = tokenizer(text, return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
        return {"tokens": tokens}

def get_tokenized_truthful_qa_for_testing(model, sae_context_size, batch_size=8):
    formatted_dataset = get_formatted_truthful_qa_for_testing()
    tokenized_dataset = formatted_dataset.map(
        partial(tokenize,
        column_name = "formatted_question",
        tokenizer = model.tokenizer,
        max_length=sae_context_size),
        batched=True,
        batch_size=batch_size,
        num_proc=None
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens", "formatted_question", "correct_label", "num_labels"])
    return tokenized_dataset

def format_examples_probing(examples) -> dict[str, list]:
    examples_formatted = {"sent": [],
                          "label": []}
    for example_question, example_choices in zip(examples["question"], examples["mc1_targets"]):
        have_correct_example, have_incorrect_example = False, False
        for choice, label in zip(example_choices["choices"], example_choices["labels"]):
            if label == 1 and not have_correct_example:
                examples_formatted["sent"].append(example_question + " " + choice)
                examples_formatted["label"].append("True")
                have_correct_example = True
            elif not have_incorrect_example:
                examples_formatted["sent"].append(example_question + " " + choice)
                examples_formatted["label"].append("False")
                have_incorrect_example = True
    return examples_formatted

def get_formatted_truthful_qa_for_probing(batch_size=8, select_items=500):
    dataset = load_dataset(
        "truthfulqa/truthful_qa", "multiple_choice",
        split="validation",
        streaming=False,
    )
    probing_dataset = dataset.map(format_examples_probing, batched=True,
                    batch_size=batch_size, remove_columns=dataset.column_names)

    probing_dataset = probing_dataset.shuffle(seed=42).select(range(select_items))
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

def get_tokenized_truthful_qa_for_probing(model, sae_context_size, batch_size=8, select_items=500):
    probing_dataset = get_formatted_truthful_qa_for_probing(batch_size, select_items)
    tokenized_dataset = probing_dataset.map(
        partial(tokenize_probing,
        column_name = "sent",
        tokenizer = model.tokenizer,
        max_length=sae_context_size),
        batched=True,
        batch_size=batch_size,
        num_proc=None
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens", "label", "len_of_input", "sent"])
    return tokenized_dataset