# modules/groq_assistant.py — v8 FIXED
#
# BUG FIXED: HTTP 400 Bad Request on OCR/vision commands
#   ROOT CAUSE: "llama-3.2-11b-vision-preview" model name
#   changed or is unavailable on some Groq accounts.
#   Also: image base64 too large caused 400 on some commands.
#
# FIXES:
#   1. Auto-selects working vision model from priority list
#   2. Image resized to 320x240 before encoding (smaller payload)
#   3. JPEG quality lowered to 70 (faster, smaller)
#   4. Fallback to text-only Groq if vision fails
#   5. Better error messages telling user exactly what's wrong

import base64
import threading
import time
import cv2
import requests
import config

PROMPTS = {
    "scene":    (
        "You are helping a BLIND person. Describe this scene in ONE short sentence. "
        "Focus on: people, obstacles, hazards, objects. Be direct and concise."
    ),
    "ocr":      (
        "Read ALL visible text in this image. List each piece of text. "
        "If no text is visible, say exactly: No text found."
    ),
    "currency": (
        "Identify any Indian Rupee banknotes in this image. "
        "State denomination (10, 20, 50, 100, 200, 500 or 2000 rupees). "
        "If none, say exactly: No currency visible."
    ),
    "emotion":  (
        "Describe the facial expression of the person in ONE sentence. "
        "If no person, say: No person visible."
    ),
    "face":     (
        "Describe the person in ONE sentence: age range, gender, clothing color. "
        "If no person, say: No person visible."
    ),
    "objects":  (
        "List the 3 main objects you see, for a blind person. Under 12 words total."
    ),
}

# Vision models tried in order — first working one is used
_VISION_MODELS = [
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    # "llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision",          # <-- Added
    "llama-3.2-90b-vision",          # <-- Added
]

# Text-only model (for weather and fallback)
_TEXT_MODEL = "llama-3.1-8b-instant"

_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


class GroqAssistant:

    def __init__(self, speaker, api_key: str = None):
        self.speaker        = speaker
        self._lock          = threading.Lock()
        self._busy          = False
        self._ready         = False
        self._vision_model  = None
        self._api_key       = None
        self._base_url      = "https://api.groq.com/openai/v1"

        import os
        key = (
            api_key
            or getattr(config, "GROQ_API_KEY", None)
            or os.environ.get("GROQ_API_KEY")
        )

        if not key or "your_key" in str(key):
            print("[Groq] No API key set.")
            print("[Groq] Get free key: https://console.groq.com")
            print("[Groq] Set GROQ_API_KEY in config.py")
            return

        self._api_key = key
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }

        # Test connection and find working vision model
        # Test connection and find working vision model
        if self._test_text_connection():
            self._vision_model = "llama-3.2-11b-vision-preview" 
            self._ready        = True
            print(f"[Groq] Ready. Vision model: {self._vision_model}")
        else:
            print("[Groq] Connection failed. Check API key.")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Public API ────────────────────────────────────────

    def ask_vision(self, frame, query_type: str, confirmation: str = "Looking…"):
        if not self._ready:
            self.speaker.say("Groq AI not configured. Add GROQ_API_KEY to config.py.", priority=8)
            return
        if self._is_busy():
            return
        self.speaker.say(confirmation, priority=8)
        threading.Thread(
            target=self._vision_worker,
            args=(frame.copy(), query_type),
            daemon=True,
            name=f"Groq-{query_type}",
        ).start()

    def ask_weather(self, confirmation: str = "Checking weather…"):
        if self._is_busy():
            return
        self.speaker.say(confirmation, priority=7)
        threading.Thread(
            target=self._weather_worker,
            daemon=True,
            name="Groq-weather",
        ).start()

    # ── Internal workers ──────────────────────────────────

    def _vision_worker(self, frame, query_type: str):
        try:
            prompt = PROMPTS.get(query_type, PROMPTS["scene"])

            if self._vision_model:
                answer = self._call_vision(frame, prompt)
            else:
                # No vision model available — use text-only fallback
                answer = self._call_text_fallback(query_type)

            self.speaker.say(answer or "No result.", priority=8)
            print(f"[Groq] {query_type}: {(answer or '')[:80]}")

        except Exception as e:
            print(f"[Groq] Vision error: {e}")
            self.speaker.say("AI error. Try again.", priority=8)
        finally:
            self._done()

    def _call_vision(self, frame, prompt: str) -> str | None:
        """
        FIX: Resize image to 320x240 before encoding.
        This reduces payload size and fixes HTTP 400 errors.
        """
        # Resize small — reduces payload, faster, fixes 400 errors
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70]   # FIX: was 80
        )
        if not ok:
            return None

        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "model": self._vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens":  100,
            "temperature": 0.1,
        }

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=15,
        )

        # FIX: Handle 400 gracefully — try next model
        if resp.status_code == 400:
            print(f"[Groq] 400 on {self._vision_model} — trying next model")
            next_model = self._find_vision_model(skip=self._vision_model)
            if next_model:
                self._vision_model = next_model
                payload["model"] = next_model
                resp = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=15,
                )
            else:
                return "Vision model unavailable. Try again later."

        resp.raise_for_status()
        return (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def _call_text_fallback(self, query_type: str) -> str:
        """Text-only fallback when vision model unavailable."""
        messages = {
            "ocr":      "Say: I cannot read text without vision access.",
            "scene":    "Say: I cannot describe the scene without vision access.",
            "currency": "Say: I cannot detect currency without vision access.",
            "face":     "Say: I cannot identify faces without vision access.",
            "emotion":  "Say: I cannot detect emotions without vision access.",
            "objects":  "Say: I cannot detect objects without vision access.",
        }
        return messages.get(query_type, "Vision not available.")

    def _weather_worker(self):
        try:
            lat  = getattr(config, "LOCATION_LAT",  31.2554)
            lon  = getattr(config, "LOCATION_LON",  75.7049)
            name = getattr(config, "LOCATION_NAME", "your location")

            r = requests.get(
                _WEATHER_URL,
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "current":          "temperature_2m,weathercode,windspeed_10m",
                    "forecast_days":    1,
                    "timezone":         "auto",
                    "wind_speed_unit":  "kmh",
                    "temperature_unit": "celsius",
                },
                timeout=10,
            )
            r.raise_for_status()
            d   = r.json()["current"]
            msg = (
                f"Weather in {name}: "
                f"{_wcode(d.get('weathercode', 0))}. "
                f"{d.get('temperature_2m', '?'):.0f} degrees Celsius. "
                f"Wind {d.get('windspeed_10m', 0):.0f} kilometres per hour."
            )
            self.speaker.say(msg, priority=7)

        except Exception as e:
            self.speaker.say("Could not fetch weather.", priority=7)
            print(f"[Groq] Weather error: {e}")
        finally:
            self._done()

    # ── Connection helpers ────────────────────────────────

    def _test_text_connection(self) -> bool:
        """Test API key with a cheap text-only call."""
        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model":    _TEXT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=8,
            )
            if resp.status_code == 401:
                print("[Groq] Invalid API key. Check GROQ_API_KEY in config.py.")
                return False
            return resp.status_code == 200
        except Exception as e:
            print(f"[Groq] Connection test failed: {e}")
            return False

    def _find_vision_model(self, skip: str = None) -> str | None:
        """
        Find first working vision model.
        Sends a tiny 1x1 pixel test image to each candidate.
        """
        # 1x1 white pixel JPEG as base64
        tiny_b64 = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
            "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAA"
            "AAAAAAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAA"
            "AAAA/9oADAMBAAIRAxEAPwCwABmX/9k="
        )

        for model in _VISION_MODELS:
            if model == skip:
                continue
            try:
                resp = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type":      "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{tiny_b64}"
                                        },
                                    },
                                    {"type": "text", "text": "ok"},
                                ],
                            }
                        ],
                        "max_tokens": 5,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    print(f"[Groq] Vision model found: {model}")
                    return model
                else:
                    print(f"[Groq] {model}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"[Groq] {model}: {e}")

        print("[Groq] No vision model available. Text-only mode.")
        return None

    def _is_busy(self) -> bool:
        with self._lock:
            if self._busy:
                self.speaker.say("Still working. Please wait.", priority=6)
                return True
            self._busy = True
            return False

    def _done(self):
        with self._lock:
            self._busy = False


def _wcode(code: int) -> str:
    return {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 51: "light drizzle", 61: "light rain",
        63: "moderate rain", 65: "heavy rain",
        71: "light snow", 73: "moderate snow",
        80: "light showers", 81: "moderate showers",
        95: "thunderstorm",
    }.get(code, "mixed conditions")