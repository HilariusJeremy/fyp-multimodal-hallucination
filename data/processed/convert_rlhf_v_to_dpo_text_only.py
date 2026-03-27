# scripts/convert_rlhf_v_to_text_dpo.py
import re
from datasets import load_dataset

IMAGE_PLACEHOLDER = "<image>"

def clean_text(text: str) -> str:
    """Remove <image> placeholders and tidy up leftover whitespace."""
    text = text.replace(IMAGE_PLACEHOLDER, "")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def clean_example(example):
    # conversations is a list of {"from": ..., "value": ...}
    example["conversations"] = [
        {**turn, "value": clean_text(turn["value"])}
        for turn in example["conversations"]
    ]
    # chosen and rejected are single dicts {"from": ..., "value": ...}
    for field in ("chosen", "rejected"):
        val = example[field]
        if isinstance(val, dict):
            example[field] = {**val, "value": clean_text(val["value"])}
    return example

# Load from HF cache (no re-download if already cached)
ds = load_dataset("llamafactory/RLHF-V", split="train")
print(f"Loaded {len(ds)} examples")

# Clean
ds_clean = ds.map(clean_example)

# Drop the images column
ds_clean = ds_clean.remove_columns(["images"])

# Verify no <image> tokens remain
def verify(example):
    for turn in example["conversations"]:
        assert IMAGE_PLACEHOLDER not in turn["value"], f"Found placeholder in conversations: {turn}"
    for field in ("chosen", "rejected"):
        assert IMAGE_PLACEHOLDER not in example[field]["value"], f"Found placeholder in {field}: {example[field]}"

ds_clean.map(verify)
print("Verification passed — no <image> tokens remain")

# Save to data/processed/
output_path = "data/processed/rlhf_v_text.json"
ds_clean.to_json(output_path)
print(f"Saved {len(ds_clean)} examples to {output_path}")
