# modules/ocr_reader.py — reads text from the scene using EasyOCR  (corrected)
#
# FIXES applied:
#   1. Removed ssl._create_default_https_context global override — it disabled
#      SSL verification for the ENTIRE process (security risk). If EasyOCR model
#      download fails with SSL errors, run once with EASYOCR_DISABLE_SSL=1 env var.
#   2. gpu=config.OCR_GPU — was hardcoded gpu=False, ignoring config setting

import easyocr
import config

class OCRReader:
    def __init__(self):
        print("[OCRReader] Loading EasyOCR model...")
        print(f"[OCRReader] GPU: {config.OCR_GPU} | Languages: {config.OCR_LANGUAGES}")
        # FIX: pass config.OCR_GPU instead of hardcoded False
        self.reader = easyocr.Reader(
            config.OCR_LANGUAGES,
            gpu=config.OCR_GPU
        )
        print("[OCRReader] Model ready.")

    def read(self, frame):
        """
        Extract text from a frame.
        Returns a cleaned string of all detected text, or None.
        """
        try:
            results = self.reader.readtext(frame)

            lines = []
            for (bbox, text, confidence) in results:
                if confidence >= config.OCR_MIN_CONFIDENCE:
                    cleaned = text.strip()
                    if cleaned:
                        lines.append(cleaned)

            if not lines:
                return None

            full_text = " ".join(lines)
            return f"I can see text that says: {full_text}"

        except Exception as e:
            print(f"[OCRReader] Error during text recognition: {e}")
            return None