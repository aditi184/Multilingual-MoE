import json

def convert_test_format(input_path, output_path):
    """Converts test JSON array to JSONL format"""
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        data = json.load(f_in)
        for item in data:
            f_out.write(json.dumps(item) + "\n")

# Example usage:
convert_test_format("/home/mila/k/khandela/scratch/CulturaX_text/test/zh_test.json", "/home/mila/k/khandela/scratch/CulturaX_text/test/zh_test.jsonl")