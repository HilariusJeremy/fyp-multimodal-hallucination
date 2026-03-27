# scripts/convert_rlhf_v_to_sft.py
import json
import os
from datasets import load_dataset

def convert_rlhf_v_to_sft(
    parquet_file_path: str = "data/raw/RLHF-V-Dataset.parquet",
    output_jsonl: str = "data/processed/rlhf_v_sft.jsonl",
    output_images_dir: str = "data/processed/rlhf_v_images"
):
    """Convert RLHF-V parquet dataset to SFT format and save images."""
    
    # Load the parquet file
    print(f"Loading dataset from {parquet_file_path}...")
    dataset = load_dataset("parquet", data_files=parquet_file_path)["train"]
    
    print(f"Loaded dataset with columns: {dataset.column_names}")
    print(f"Total samples: {len(dataset)}\n")
    
    # Create images directory
    os.makedirs(output_images_dir, exist_ok=True)
    
    sft_samples = []
    image_counter = 0
    error_count = 0
    skipped_no_images = 0
    
    for idx, sample in enumerate(dataset):
        try:
            # Get conversations (user messages)
            conversations = sample.get("conversations", [])
            if not conversations:
                error_count += 1
                continue
            
            # Get chosen response
            chosen = sample.get("chosen")
            if chosen is None:
                error_count += 1
                continue
            
            # Add chosen response to conversations
            new_conversations = list(conversations) + [chosen]
            
            # Create SFT sample
            sft_sample = {
                "conversations": new_conversations,
            }
            
            # Handle images - extract from parquet and save locally
            images_list = sample.get("images", [])
            if images_list and len(images_list) > 0:
                image_paths = []
                for img_idx, img in enumerate(images_list):
                    if img is not None:  # Skip None/empty images
                        try:
                            # Save image with unique name
                            image_filename = f"image_{image_counter}.jpg"
                            image_path = os.path.join(output_images_dir, image_filename)
                            
                            # Save PIL Image
                            if hasattr(img, 'save'):
                                img.save(image_path, quality=95)
                                # Store full relative path from repo root
                                image_paths.append(os.path.join("data/processed/rlhf_v_images", image_filename))
                                image_counter += 1
                            else:
                                print(f"Warning at sample {idx}: Unknown image type {type(img)}")
                                continue
                        except Exception as e:
                            print(f"Error saving image at sample {idx}, image {img_idx}: {e}")
                            error_count += 1
                            continue
                
                if image_paths:
                    sft_sample["images"] = image_paths
                else:
                    skipped_no_images += 1
            else:
                skipped_no_images += 1
            
            sft_samples.append(sft_sample)
            
            if (idx + 1) % 500 == 0:
                print(f"✓ Processed {idx + 1}/{len(dataset)} samples, saved {image_counter} images...")
        
        except Exception as e:
            print(f"✗ Error processing sample {idx}: {e}")
            error_count += 1
            continue
    
    # Save to JSONL
    with open(output_jsonl, "w") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"  Total samples converted: {len(sft_samples)}")
    print(f"  Total images saved: {image_counter}")
    print(f"  Samples skipped (no images): {skipped_no_images}")
    print(f"  Errors encountered: {error_count}")
    print(f"\n  Output JSONL: {output_jsonl}")
    print(f"  Output images: {output_images_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    parquet_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/RLHF-V-Dataset.parquet"
    output_jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/processed/rlhf_v_sft.jsonl"
    output_images = sys.argv[3] if len(sys.argv) > 3 else "data/processed/rlhf_v_images"
    
    convert_rlhf_v_to_sft(parquet_file, output_jsonl, output_images)
