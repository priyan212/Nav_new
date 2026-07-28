"""Loads Qwen2.5-VL and runs single-call image+text generation.

Uses transformers directly (not Ollama) so prompts/outputs stay structured
JSON-friendly for the caller. This module only does text-in/text-out; JSON
parsing of the response is the caller's responsibility.
"""

import sys

import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_qwen_model(device: str = "cuda") -> tuple:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return model, processor


def run_qwen_inference(
    model,
    processor,
    image: np.ndarray,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    pil_image = Image.fromarray(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


if __name__ == "__main__":
    image_path = sys.argv[1]
    test_image = np.array(Image.open(image_path).convert("RGB"))

    model, processor = load_qwen_model()
    result = run_qwen_inference(
        model, processor, test_image, "Describe this image in one sentence."
    )
    print(result)
