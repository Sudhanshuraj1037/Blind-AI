# modules/scene_describer.py — AI caption of the full scene using BLIP

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import config

class SceneDescriber:
    def __init__(self):
        print("[SceneDescriber] Loading BLIP vision-language model...")
        print("[SceneDescriber] First run will download ~1GB model. Please wait...")
        self.processor = BlipProcessor.from_pretrained(config.SCENE_MODEL)
        self.model     = BlipForConditionalGeneration.from_pretrained(
            config.SCENE_MODEL
        ).to(config.DEVICE)
        self.model.eval()
        print(f"[SceneDescriber] Ready on {config.DEVICE}.")

    def describe(self, frame):
        """
        Generate a natural language description of the full scene.
        Returns a spoken string like:
        'A street scene with cars parked on the side and people walking.'
        """
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image   = Image.fromarray(img_rgb)

            # Prepare input for BLIP
            inputs = self.processor(image, return_tensors="pt").to(config.DEVICE)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=config.SCENE_MAX_TOKENS,
                    num_beams=3,
                )

            caption = self.processor.decode(output[0], skip_special_tokens=True)

            if not caption:
                return None

            # Make it sound natural for the user
            return f"Scene description: {caption}."

        except Exception as e:
            print(f"[SceneDescriber] Error: {e}")
            return None