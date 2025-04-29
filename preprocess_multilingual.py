# python preprocess_multilingual.py --languages en hi ru zh ar --samples 1966535 --output_dir /home/mila/k/khandela/scratch/CulturaX_text/multilang
# python preprocess_multilingual.py --languages en hi ru zh ar --samples 98326775 --output_dir /home/mila/k/khandela/scratch/CulturaX_text/multilang
import argparse
from datasets import load_dataset
import json
import os
from itertools import cycle, islice

def create_doc(example, language, idx):
    return {
        "id": f"{language}_{idx}",
        "text": example["text"],
        "source": example["source"],
        "timestamp": example["timestamp"],
        "url": example["url"],
        "metadata": {},
        "language": language
    }

def download_culturax_multilang(languages, num_samples, output_dir):
    if num_samples <= 5000:
        raise ValueError("Number of samples must be greater than 5000 to split into test and train.")

    os.makedirs(output_dir, exist_ok=True)

    # Stream and cache dataset iterators per language
    datasets = {}
    for lang in languages:
        datasets[lang] = iter(load_dataset("uonlp/CulturaX", lang, streaming=True)["train"])

    # === STEP 1: Create separate test sets for each language ===
    for lang in languages:
        test_samples = list(islice(datasets[lang], 5000))
        test_data = [create_doc(ex, lang, idx) for idx, ex in enumerate(test_samples)]
        
        test_path = os.path.join(output_dir, f"{lang}_test.json")
        with open(test_path, "w",  encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)
        
        print(f"Saved test set for {lang} to {test_path}")

    # === STEP 2: Interleave training data across languages ===
    total_train_samples = num_samples - 5000 * len(languages)
    samples_per_lang = total_train_samples // len(languages)
    iters = {lang: islice(datasets[lang], samples_per_lang) for lang in languages}
    lang_cycle = cycle(languages)

    train_file = None
    train_count = 0
    file_counter = 1
    train_file_path = os.path.join(output_dir, f"multilang_train_{file_counter}.jsonl")
    train_file = open(train_file_path, "w", encoding="utf-8")


    while train_count < samples_per_lang * len(languages):
        for lang in lang_cycle:
            try:
                example = next(iters[lang])
                doc = create_doc(example, lang, train_count)
                train_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                train_count += 1

                # If limit reached, rotate file
                if train_count % 1000000 == 0:
                    train_file.close()
                    file_counter += 1
                    train_file_path = os.path.join(output_dir, f"multilang_train_{file_counter}.jsonl")
                    train_file = open(train_file_path, "w", encoding="utf-8")


                if train_count >= samples_per_lang * len(languages):
                    break

            except StopIteration:
                continue

    train_file.close()
    print(f"Saved training data across {file_counter} file(s) to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and interleave CulturaX dataset for multiple languages.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of language codes (e.g., en hi es fr tr).")
    parser.add_argument("--samples", type=int, required=True, help="Total number of samples across all languages (must be >5000 * num_languages).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving data.")

    args = parser.parse_args()

    min_required_samples = 5000 * len(args.languages)
    if args.samples <= min_required_samples:
        parser.error(f"--samples must be greater than {min_required_samples} to allow 5000 test samples per language.")

    download_culturax_multilang(args.languages, args.samples, args.output_dir)
