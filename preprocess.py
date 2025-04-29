# run via python preprocess.py --language hi --samples 10000
import argparse
from datasets import load_dataset
import json
import os

def download_culturax(language, num_samples, output_dir="/home/mila/k/khandela/scratch/CulturaX_text"):
    """Downloads CulturaX dataset for a specific language and saves it to JSONL."""

    try:
        dataset_name = "uonlp/CulturaX"
        # dataset = load_dataset("/network/datasets/culturax.var/culturax_13lang_huggingface/CulturaX", language)
        # data_files = "/network/datasets/culturax.var/culturax_13lang_huggingface/CulturaX/hi/*.parquet"
        dataset = load_dataset(dataset_name, language, streaming=True)
        # dataset = dataset.take(num_samples)
        print(dataset)
        os.makedirs(output_dir, exist_ok=True)

        jsonl_path = os.path.join(output_dir, f"culturax_{language}.jsonl")
        
        with open(jsonl_path, "w") as f:
            for idx, example in enumerate(dataset["train"]):
                doc = {
                    "id": f"{language}_{idx}",
                    "text": example["text"],
                    "source": example["source"],
                    "timestamp": example["timestamp"],
                    "url": example["url"],
                    "metadata": {}
                }
                
                f.write(json.dumps(doc) + "\n")

                if idx == num_samples:
                    break

        print(f"Dataset saved to {jsonl_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CulturaX dataset.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., en, hi).")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples to download.")
    parser.add_argument("--output_dir", type=str, default="/home/mila/k/khandela/scratch/CulturaX_text", help="Output directory.")

    args = parser.parse_args()

    download_culturax(args.language, args.samples, args.output_dir)