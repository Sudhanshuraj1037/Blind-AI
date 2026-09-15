<div align="center">

# 👁️ Blind AI

### Real-Time AI-Powered Assistive Vision System

<p>
  <strong>See the world. Understand the surroundings. Hear what matters.</strong>
</p>

<p>
  A modular computer-vision assistant designed to help visually impaired users
  understand their environment through real-time perception, voice interaction,
  and audio feedback.
</p>

<br/>

<img
  src="assets/blind-ai-demo.png"
  alt="Blind AI real-time object detection demo"
  width="900"
/>

<br/>
<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-111111?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Groq](https://img.shields.io/badge/Groq-Vision--Language-F55036?style=for-the-badge)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🧠 Overview

**Blind AI** is a real-time assistive vision system that combines multiple
computer-vision and AI components into a single perception pipeline.

The system processes visual and voice inputs, extracts meaningful information
from the environment, and converts that information into accessible feedback.

The goal is not simply to detect objects, but to build a system that can
**perceive, reason about, and communicate the surrounding environment.**

---

## ✨ What It Can Do

<div align="center">

| Capability | Description |
|:---:|---|
| 🎯 **Object Detection** | Detects objects from the camera feed using YOLO |
| 📏 **Depth Estimation** | Estimates relative scene depth using Depth Anything V2 |
| 🔤 **OCR** | Extracts readable text from the environment |
| 😊 **Face & Emotion** | Detects faces and provides emotion-related perception |
| ✋ **Gesture Recognition** | Processes hand/gesture-based interactions |
| 💰 **Currency Detection** | Uses a custom-trained YOLO model for currency recognition |
| 🚦 **Traffic Light Detection** | Detects traffic-light states through color-based processing |
| 🎙️ **Voice Commands** | Accepts spoken commands for interaction |
| 🧠 **AI Reasoning** | Uses a vision-language model for open-ended scene queries |
| 🔊 **Audio Feedback** | Converts system responses into speech |

</div>

---

# 🏗️ System Architecture

Blind AI follows a modular perception-and-response architecture.

```text
                         ┌──────────────────┐
                         │   Camera Feed    │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       Object Detection       Depth Estimation      OCR
          YOLOv8n           Depth Anything V2
              │                   │                   │
              ├───────────────┬───┴───────────────┬───┤
              │               │                   │
              ▼               ▼                   ▼
        Face/Emotion       Gesture            Currency
              │               │              Detection
              └───────────────┼───────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Fusion Engine  │
                     └────────┬────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Command Parser          Vision-Language
                                      Reasoning
                                          │
                                          ▼
                                  ┌──────────────┐
                                  │ Text-to-Speech│
                                  └──────┬───────┘
                                         │
                                         ▼
                                   Audio Feedback
