import argparse
import json
import os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def convert_url_to_local_path(image_src, image_dir="data/raw/mmhalbench/images"):
    """Convert URL to local filename by extracting the filename from the URL."""
    if image_src.startswith('http://') or image_src.startswith('https://'):
        # Extract filename from URL (last part after the final '/')
        filename = image_src.split('/')[-1]
        return os.path.join(image_dir, filename)
    else:
        # Already a local path
        return image_src


def load_image(image_file):
    """Load image from local file path."""
    image = Image.open(image_file).convert('RGB')
    return image


def build_qwen_model(model_path, adapter_path=None):
    """Build and return the Qwen3-VL model and processor."""
    print("="*80)
    if adapter_path is not None:
        print("🔧 MODE: LoRA/Adapter Fine-tuned Model")
        print(f"📦 Base Model: {model_path}")
        print(f"🎯 Adapter: {adapter_path}")
    else:
        print("📦 MODE: Base Model (No Adapter)")
        print(f"📦 Model Path: {model_path}")
    print("="*80)

    print("\nLoading base model...")
    # Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )

    # Load adapter if provided
    if adapter_path is not None:
        print(f"\n🔧 Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ Adapter loaded successfully! Using fine-tuned model.")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     model_path,
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained(model_path)

    print("✅ Model and processor loaded successfully!\n")
    return model, processor, (adapter_path is not None)


def get_model_response(model, processor, image_file, question, max_new_tokens=512):
    """Get response from Qwen3-VL model for a given image and question."""
    # Prepare the message format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Qwen3-VL-4B-Instruct inference on MMHalBench')
    parser.add_argument('--input', type=str, default='data/raw/mmhalbench/response_template.json',
                        help='Input JSON file containing images and questions')
    parser.add_argument('--output', type=str, default='data/raw/mmhalbench/responses/response_qwen3vl_4b.json',
                        help='Output JSON file to save model responses')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model or HuggingFace model ID (e.g. Qwen/Qwen3-VL-4B-Instruct)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Path to the adapter/LoRA weights (optional)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # Build the Qwen3-VL model
    model, processor, is_adapter = build_qwen_model(args.model_path, args.adapter_path)

    # Load the benchmark data
    print(f"Loading benchmark data from {args.input}...")
    with open(args.input, 'r') as f:
        json_data = json.load(f)

    model_type = "LoRA Fine-tuned" if is_adapter else "Base Model"
    print(f"Processing {len(json_data)} samples with {model_type}...")
    print("="*80)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process each sample
    for idx, line in enumerate(json_data):
        image_src = line['image_src']
        question = line['question']

        # Convert URL to local filename if needed
        local_image_path = convert_url_to_local_path(image_src)

        print(f"[{idx+1}/{len(json_data)}] Processing: {image_src}")
        print(f"  Loading from local file: {local_image_path}")

        # Get model response
        response = get_model_response(model, processor, local_image_path, question, args.max_new_tokens)

        print(f"  Response: {response[:100]}..." if len(response) > 100 else f"  Response: {response}")

        # Add the response to the data
        line['model_answer'] = response

    # Save the results
    print("="*80)
    print(f"\n✅ Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✅ Done! Processed {len(json_data)} samples with {model_type}.")
    if is_adapter:
        print(f"🔧 Adapter used: {args.adapter_path}")
    print("="*80)
