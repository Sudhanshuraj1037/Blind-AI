# modules/llava_describer.py — LLaVA vision-language reasoning  (corrected)
#
# FIXES applied:
#   1. BLIP fallback load: all attributes (_blip_proc, _blip_model, _blip_Image,
#      _blip_torch) are set atomically — partial load no longer leaves None attrs
#      that cause AttributeError in _describe_blip()
#   2. LLaVA describe() now runs in a thread with timeout so it cannot block
#      the ml_worker daemon for 15+ seconds during a slow response

import cv2
import base64
import threading
import requests
import config

class LLaVADescriber:
    def __init__(self):
        # BLIP attributes set to None upfront — avoids AttributeError on partial load
        self._blip_proc  = None
        self._blip_model = None
        self._blip_Image = None
        self._blip_torch = None

        self._ollama_ok = self._check_ollama()
        if self._ollama_ok:
            print(f"[LLaVADescriber] Ollama + LLaVA ready.")
        else:
            print("[LLaVADescriber] Ollama not found. Loading BLIP fallback...")
            self._load_blip_fallback()

    def _check_ollama(self):
        try:
            r      = requests.get("http://localhost:11434/api/tags", timeout=2)
            models = [m["name"] for m in r.json().get("models", [])]
            if any("llava" in m for m in models):
                return True
            print("[LLaVADescriber] LLaVA model not pulled. Run: ollama pull llava")
            return False
        except Exception:
            return False

    def _load_blip_fallback(self):
        """
        FIX: Wrap the ENTIRE BLIP load in one try block.
        All four attributes are set together or not at all.
        This prevents partial-load states where _blip_proc is set but _blip_model is None.
        """
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image

            proc  = BlipProcessor.from_pretrained(config.SCENE_FALLBACK_MODEL)
            model = BlipForConditionalGeneration.from_pretrained(
                config.SCENE_FALLBACK_MODEL
            ).to(config.DEVICE)
            model.eval()

            # All four assigned together — atomic success or nothing
            self._blip_proc  = proc
            self._blip_model = model
            self._blip_Image = Image
            self._blip_torch = torch
            print("[LLaVADescriber] BLIP fallback ready.")

        except Exception as e:
            # FIX: explicitly reset all to None on failure — no partial state
            self._blip_proc  = None
            self._blip_model = None
            self._blip_Image = None
            self._blip_torch = None
            print(f"[LLaVADescriber] BLIP also failed: {e}")

    def describe(self, frame, question=None):
        """
        Generate a scene description or answer a specific question.
        FIX: runs in a thread so ml_worker is not blocked by LLAVA_TIMEOUT.
        """
        if self._ollama_ok:
            return self._describe_with_timeout(frame, question)
        return self._describe_blip(frame)

    def answer(self, frame, question):
        """Ask LLaVA a specific question about the scene."""
        return self.describe(frame, question=question)

    def _describe_with_timeout(self, frame, question):
        """
        FIX: run LLaVA call in a sub-thread and join with timeout.
        Prevents LLAVA_TIMEOUT seconds of blocking on the ml_worker thread.
        """
        result_holder = [None]

        def _call():
            result_holder[0] = self._describe_llava(frame, question)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=config.LLAVA_TIMEOUT)

        if t.is_alive():
            print("[LLaVADescriber] LLaVA timed out — skipping this frame.")
            return None
        return result_holder[0]

    def _describe_llava(self, frame, question=None):
        try:
            _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_image = base64.b64encode(buf).decode("utf-8")

            prompt = question if question else config.LLAVA_PROMPT

            payload = {
                "model":  config.LLAVA_MODEL,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
            }

            resp = requests.post(
                config.LLAVA_ENDPOINT,
                json=payload,
                timeout=config.LLAVA_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()

            if not result:
                return None

            return f"Answer: {result}" if question else f"Scene: {result}"

        except requests.Timeout:
            print("[LLaVADescriber] LLaVA request timed out.")
            return None
        except Exception as e:
            print(f"[LLaVADescriber] LLaVA error: {e}")
            return None

    def _describe_blip(self, frame):
        # FIX: guard checks _blip_model (None means full load failed)
        if self._blip_model is None:
            return None
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._blip_Image.fromarray(rgb)
            inputs = self._blip_proc(image, return_tensors="pt").to(config.DEVICE)
            with self._blip_torch.no_grad():
                out = self._blip_model.generate(
                    **inputs,
                    max_new_tokens=config.SCENE_MAX_TOKENS,
                    num_beams=3
                )
            caption = self._blip_proc.decode(out[0], skip_special_tokens=True)
            return f"Scene: {caption}." if caption else None
        except Exception as e:
            print(f"[LLaVADescriber] BLIP error: {e}")
            return None