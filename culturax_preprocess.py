# python culturax_preprocess.py --language en --samples 1966535 --output_dir /home/mila/k/khandela/scratch/CulturaX_text/en-train
import argparse
from datasets import load_dataset
import json
import os

def create_doc(example, language, idx):
    return {
        "id": f"{language}_{idx}",
        "text": example["text"],
        "source": example["source"],
        "timestamp": example["timestamp"],
        "url": example["url"],
        "metadata": {}
    }

def download_culturax(language, num_samples, output_dir):
    """Downloads CulturaX dataset for a specific language, splits into test and train JSON files."""
    if num_samples <= 5000:
        raise ValueError("Number of samples must be greater than 5000 to split into test and train.")

    try:
        dataset = load_dataset("uonlp/CulturaX", language, streaming=True)
        os.makedirs(output_dir, exist_ok=True)

        test_data = []
        train_data_count = 0
        current_file = None
        file_counter = 0

        for idx, example in enumerate(dataset["train"]):
            if idx >= num_samples:
                break

            if idx < 5000:
                # Collect test data
                test_data.append(create_doc(example, language, idx))
            else:
                # Process train data
                train_data_count += 1

                # Check if new file needs to be created
                if (train_data_count - 1) % 1000000 == 0:
                    if current_file is not None:
                        current_file.close()
                    file_counter = ((train_data_count - 1) // 1000000) + 1
                    train_filename = os.path.join(output_dir, f"{language}_train_{file_counter}.jsonl")
                    current_file = open(train_filename, "w")

                # Write the doc to the current JSONL file
                doc = create_doc(example, language, idx)
                current_file.write(json.dumps(doc) + "\n")

        # Close the last train file if open
        if current_file is not None:
            current_file.close()

        # Write test data to JSON file
        test_path = os.path.join(output_dir, f"{language}_test.json")
        with open(test_path, "w") as f:
            json.dump(test_data, f)

        print(f"Test data saved to {test_path}")
        print(f"Train data split into {file_counter} JSONL files with up to 1,000,000 samples each.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split CulturaX dataset into test and train.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., hi).")
    parser.add_argument("--samples", type=int, required=True, help="Total number of samples to process (must be >5000).")
    parser.add_argument("--output_dir", type=str, default="/home/mila/k/khandela/scratch/CulturaX_text", help="Output directory.")

    args = parser.parse_args()

    if args.samples <= 5000:
        parser.error("--samples must be greater than 5000")

    download_culturax(args.language, args.samples, args.output_dir)