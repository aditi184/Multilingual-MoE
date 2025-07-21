import datasets
from olmo.util import load_hf_dataset

dataset = load_hf_dataset("hails/mmlu_no_train", "abstract_algebra", split="dev")


def format_example(doc, key, mc_labels):
    question_prefix = ""
    if not mc_labels:
        question_prefix = "Question: "  # To make context more clear
    question = question_prefix + doc["question"].strip()
    choices = ""
    if mc_labels:
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])])
    prompt = f"{question}\n{choices}Answer:"
    return prompt


def format_example_v2(doc, mc_labels=True):
    question_prefix = ""
    if not mc_labels:
        question_prefix = "Question: "
    question = question_prefix + doc["question"].strip()

    # Extract and format choices
    if mc_labels:
        option_map = {
            "A": doc["option_a"],
            "B": doc["option_b"],
            "C": doc["option_c"],
            "D": doc["option_d"],
        }
        choices = "".join([f"{k}. {v}\n" for k, v in option_map.items()])
    else:
        choices = ""

    prompt = f"{question}\n{choices}Answer:"
    return prompt



keys = ["A", "B", "C", "D"]

for i in range(1):
    mc_labels = True
    output_text = format_example(dataset[i], keys, mc_labels)
    print(output_text)




dataset_global = load_hf_dataset("global_mmlu", "en", split="dev")

for i in range(1):
    mc_labels = True
    output_text = format_example_v2(dataset_global[i], mc_labels)
    print(output_text)
    
 # This is a duplicate, but included for completeness
