from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "moondream/moondream-2b-2025-04-14-4bit"

print("Torch CUDA available:", torch.cuda.is_available())
print("Loading model:", MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map={"": "cuda"} if torch.cuda.is_available() else {"": "cpu"},
)

model.generation_config.do_sample = False
model.generation_config.max_new_tokens = 40

img = Image.new("RGB", (512, 384), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((120, 120, 390, 260), outline="black", width=5)
draw.text((150, 170), "test image", fill="black")

answer = model.query(img, "Describe the image in one short sentence.")["answer"]

print("Moondream answer:")
print(answer)
print("Moondream test finished.")
