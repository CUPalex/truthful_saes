from datasets import load_dataset
import numpy as np
from functools import partial

class TruthfulQA:
    def __init__(self, type):
        assert type in ["testing", "probing"]
        self.type = type

    @staticmethod
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

    def get_dataset_for_probing(self, tokenizer, batch_size, subset, subset_len, random_seed, max_length, probing_split=True, **kwargs):
        def tokenize(examples, column_name, tokenizer, max_length):
            tokenizer.padding_side = "right"
            text = examples[column_name]
            tokens = tokenizer(text, return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
            len_of_input = np.argmax(tokens == tokenizer.pad_token_id, axis=1)
            assert (tokens[len_of_input == 0] != tokenizer.pad_token_id).all(), (len_of_input, tokens)
            len_of_input[len_of_input == 0] = tokens.shape[1]
            return {"tokens": tokens, "len_of_input": len_of_input}

        dataset = load_dataset(
            "truthfulqa/truthful_qa", "multiple_choice",
            split="validation",
            streaming=False,
        )
        dataset = dataset.train_test_split(test_size=0.5, seed=42)["test" if probing_split else "train"]
        probing_dataset = dataset.map(TruthfulQA.format_examples_probing, batched=True,
                        batch_size=batch_size, remove_columns=dataset.column_names)
        if subset:
            probing_dataset = probing_dataset.shuffle(seed=random_seed).select(range(subset_len))
        else:
            probing_dataset = probing_dataset.shuffle(seed=random_seed)
        tokenized_dataset = probing_dataset.map(
            partial(tokenize,
            column_name = "sent",
            tokenizer = tokenizer,
            max_length=max_length),
            batched=True,
            batch_size=batch_size,
            num_proc=None
        )
        tokenized_dataset.set_format(type="torch", columns=["tokens", "label", "len_of_input", "sent"])
        return tokenized_dataset
    
    def format_examples_testing(self, examples) -> dict[str, list]:
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
        
    def get_dataset_for_testing(self, tokenizer, batch_size, max_length, subset, subset_len, random_seed, testing_split=True, **kwargs):
        def tokenize(examples, column_name, tokenizer, max_length):
            tokenizer.padding_side = "left"
            text = examples[column_name]
            tokens = tokenizer(text, return_tensors="np", padding="longest", max_length=max_length)["input_ids"]
            return {"tokens": tokens}
        dataset = load_dataset(
            "truthfulqa/truthful_qa", "multiple_choice",
            split="validation",
            streaming=False,
        )
        dataset = dataset.train_test_split(test_size=0.5, seed=42)["train" if testing_split else "test"]
        formatted_dataset = dataset.map(self.format_examples_testing, batched=True, batch_size=1)
        if subset:
            formatted_dataset = formatted_dataset.shuffle(seed=random_seed).select(range(subset_len))
        else:
            formatted_dataset = formatted_dataset.shuffle(seed=random_seed)
        tokenized_dataset = formatted_dataset.map(
            partial(tokenize,
            column_name = "formatted_question",
            tokenizer = tokenizer,
            max_length = max_length),
            batched=True,
            batch_size=batch_size,
            num_proc=None
        )
        tokenized_dataset.set_format(type="torch", columns=["tokens", "formatted_question", "correct_label", "num_labels"])
        return tokenized_dataset
    
    def get_tokenized_dataset(self, **kwargs):
        if self.type == "testing":
            assert "tokenizer" in kwargs and "batch_size" in kwargs and "max_length" in kwargs and "subset" in kwargs and "subset_len" in kwargs and "random_seed" in kwargs
            return self.get_dataset_for_testing(**kwargs)
        if self.type == "probing":
            assert "tokenizer" in kwargs and "batch_size" in kwargs and "max_length" in kwargs and "subset" in kwargs and "subset_len" in kwargs and "random_seed" in kwargs
            return self.get_dataset_for_probing(**kwargs)
