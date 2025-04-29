import datasets
import os

# List of language splits to download
languages = [
    'ar', 'en','hi', 'ru', 'zh'
]

# Base directory where individual language folders will be saved
base_save_path = "/home/mila/k/khandela/OLMo/olmo_data/hf_datasets/global_mmlu"

# Ensure the base directory exists
os.makedirs(base_save_path, exist_ok=True)

print(f"Starting download and save process for {len(languages)} language splits.")
print(f"Datasets will be saved under: {base_save_path}")

for lang_code in languages:
    print(f"\nProcessing language: {lang_code}")
    try:
        # 1. Load the specific language split
        print(f"  Loading '{lang_code}' split...")
        # Use cache_dir if you want to control where the raw download goes
        # ds_split = datasets.load_dataset(
        #     "CohereForAI/Global-MMLU",
        #     name=lang_code,
        #     cache_dir="/path/to/your/hf_cache" # Optional: specify cache loc
        # )
        ds_split = datasets.load_dataset("CohereForAI/Global-MMLU", name=lang_code)
        print(f"  Successfully loaded '{lang_code}'.")

        # 2. Define the specific save path for this split
        split_save_path = os.path.join(base_save_path, lang_code)
        print(f"  Saving '{lang_code}' split to: {split_save_path}")

        # 3. Save the loaded split to disk
        ds_split.save_to_disk(split_save_path)
        print(f"  Successfully saved '{lang_code}' to disk.")

    except Exception as e:
        print(f"  ERROR processing language {lang_code}: {e}")
        # Decide if you want to stop or continue with the next language
        # For example, 'continue' will skip to the next language
        continue

print("\nFinished processing all specified language splits.")