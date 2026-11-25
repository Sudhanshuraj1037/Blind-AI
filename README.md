# 👁️ Blind-AI — AI-Powered Smart Assistant for the Visually Impaired

**An intelligent, voice-enabled assistive vision system designed to help visually impaired individuals understand and interact with their surroundings through Artificial Intelligence and Computer Vision.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-purple" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/AI-Assistive%20Technology-orange" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-success" />
</p>

---

## 📌 Overview

**Blind-AI** is an AI-powered assistive technology project that aims to improve environmental awareness and everyday independence for visually impaired individuals.

The system uses a camera, computer vision, deep learning, Optical Character Recognition (OCR), speech processing, and AI-powered question answering to interpret the user's surroundings and communicate relevant information through voice.

Unlike conventional object detection applications, Blind-AI combines multiple specialized AI modules into a unified assistant. It can identify nearby objects, recognize Indian currency notes, read text, recognize familiar faces, estimate scene depth, and respond to voice-based questions.

The primary objective is to create a practical, accessible, and intelligent assistant that provides meaningful environmental information without overwhelming the user with unnecessary announcements.

## ✨ Key Features

### 1. Real-Time Object Detection
- Detects everyday objects using the YOLOv8 deep learning model.
- Identifies objects in the camera's field of view.
- Provides voice-based object descriptions.
- Supports spatial awareness through directional object information.
- Prioritizes relevant environmental information.

### 2. Indian Currency Recognition
- Custom-trained YOLOv8 model for Indian currency detection.
- Recognizes ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 denominations.
- Announces detected currency denominations through voice output.
- Designed to assist visually impaired users during cash transactions.

**Model Performance (Reported Evaluation):**

| Evaluation Metric | Score |
|---|---:|
| Precision | 99.41% |
| Recall | 99.57% |
| mAP@50 | 99.38% |
| mAP@50–95 | 80.10% |
| F1-Score | 99.49% |

*Results are based on the project's reported currency detection model evaluation.*

### 3. Optical Character Recognition (OCR)
- Extracts readable text from camera images.
- Uses EasyOCR for text recognition.
- Converts extracted text into speech.
- Helps users read printed information and surrounding text.

### 4. Voice Interaction System
- Enables users to interact with the assistant through voice.
- Supports speech input and text-to-speech responses.
- Provides spoken feedback for recognized objects and other AI tasks.
- Designed for hands-free interaction.

### 5. Face Recognition and Memory
- Uses DeepFace for face recognition.
- Supports registering and remembering familiar individuals.
- Identifies previously registered faces.
- Provides voice-based identity announcements.

### 6. Depth Estimation
- Integrates MiDaS-based depth estimation.
- Analyzes relative depth information from camera frames.
- Aims to improve environmental and obstacle awareness.

### 7. AI-Powered Question Answering
- Supports natural-language questions about the user's surroundings.
- Integrates AI-based conversational capabilities.
- Explores Gemini and other vision-language model integrations.
- Designed to provide contextual responses rather than only object labels.

### 8. Accessibility-Focused Feedback
- Uses speech instead of relying entirely on visual interfaces.
- Focuses on relevant and actionable announcements.
- Supports directional feedback for detected objects.
- Aims to reduce unnecessary voice interruptions.

---

## 🧠 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Deep Learning | PyTorch |
| Currency Detection | Custom-trained YOLOv8 |
| OCR | EasyOCR |
| Face Recognition | DeepFace |
| Depth Estimation | MiDaS |
| Speech Recognition | Speech input processing |
| Text-to-Speech | pyttsx3 |
| Backend | FastAPI |
| AI Integration | Gemini / Groq / Vision-Language Models |
| Development | VS Code, Jupyter Notebook |
| Model Training | Google Colab |

## 🏗️ System Architecture

Blind-AI follows a modular architecture in which individual AI components handle specialized tasks.

**Processing Pipeline:**

1. **Input Layer:** Captures real-time video frames through the camera and receives voice commands.
2. **Perception Layer:** Processes images using OpenCV and specialized computer vision models.
3. **AI Processing Layer:** Performs object detection, currency recognition, OCR, face recognition, and depth estimation.
4. **Decision Layer:** Determines which information is relevant to the user's request or surroundings.
5. **Communication Layer:** Converts the processed information into understandable speech.
6. **Interaction Layer:** Allows users to ask questions and receive contextual responses.

```text
             CAMERA INPUT
                   |
                   v
          IMAGE PROCESSING
                OpenCV
                   |
                   v
          AI PROCESSING ENGINE
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
    YOLOv8       OCR        DeepFace
   Detection    EasyOCR   Face Memory
       |           |           |
       v           v           v
   Objects      Text       Identity
       |           |           |
       +-----------+-----------+
                   |
                   v
          DEPTH ESTIMATION
                MiDaS
                   |
                   v
           DECISION ENGINE
                   |
          +--------+--------+
          |                 |
          v                 v
      VOICE OUTPUT      AI QUESTION
       pyttsx3          ANSWERING
          |                 |
          +--------+--------+
                   |
                   v
             USER FEEDBACK
```

## 📊 Currency Detection Model

One of the major components of Blind-AI is its custom-trained Indian currency detection model.

**Model Details:**
- Architecture: YOLOv8n
- Input Image Size: 416 × 416
- Training Epochs: 10
- Execution: CPU-compatible inference
- Supported Classes: 10, 20, 50, 100, 200, 500

**Dataset:** [Indian Currency Detection Dataset – Roboflow](https://universe.roboflow.com/project-7nqrn/indian-currency-detection-jkt3c-blkpr/dataset/1)

**Model File:** `models/currency_yolov8n.pt`

The model is optimized for lightweight inference and is intended to support systems with limited computational resources.

## 📂 Project Structure

The project is organized around independent modules for computer vision, AI processing, and voice interaction.

```text
Blind-AI/
│
├── models/
│   └── currency_yolov8n.pt
│
├── currency_detection/
│   └── Currency detection implementation
│
├── face_recognition/
│   └── Face registration and recognition
│
├── ocr/
│   └── Text extraction
│
├── depth_estimation/
│   └── Depth processing
│
├── object_detection/
│   └── YOLO object detection
│
├── backend/
│   └── FastAPI services
│
├── frontend/
│   └── User interface (if applicable)
│
├── requirements.txt
├── README.md
└── main.py
```

*Directory names above illustrate the logical module organization; adapt them to the actual repository structure.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Webcam or compatible camera
- Microphone and speaker
- Git
- Internet connection for optional cloud AI integrations

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sudhanshuraj1037/Blind-AI.git

cd Blind-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Configure API Keys

For optional cloud AI integrations, configure the required API credentials using environment variables.

Never commit API keys or other sensitive credentials to GitHub.

### Step 5: Run the Application

Run the project's configured application entry point after installing dependencies.

For a FastAPI entry point configured as `main.py` with an `app` instance:

```bash
uvicorn main:app --reload
```

Refer to the actual application entry point and configuration when running the project.

---

## 🚀 Future Enhancements

Blind-AI is an actively evolving project. Planned improvements include:

- **Smart Glasses Integration:** Integrating the system with wearable camera glasses for hands-free assistance.
- **GPS Navigation:** Adding location-aware navigation and environmental guidance.
- **Advanced Obstacle Detection:** Improving obstacle awareness and directional distance estimation.
- **Improved Voice Interaction:** Enhancing speech recognition, response speed, and conversational reliability.
- **Offline AI Processing:** Expanding local inference to reduce dependence on internet connectivity.
- **Context-Aware Assistance:** Improving intelligent prioritization of hazards and relevant environmental information.
- **Performance Optimization:** Improving real-time processing on low-resource hardware.

## 🔬 Research and Development

Blind-AI explores the integration of multiple AI disciplines into an assistive technology platform:

- Computer Vision
- Object Detection and Recognition
- Deep Learning
- Optical Character Recognition
- Face Recognition
- Monocular Depth Estimation
- Speech Processing
- Human-Computer Interaction
- Assistive Artificial Intelligence

The project focuses on combining these technologies into a unified, practical solution for accessibility.

## ⚠️ Limitations and Disclaimer

Blind-AI is an experimental assistive technology prototype intended to support environmental awareness.

- AI predictions may occasionally be incorrect.
- Camera performance depends on lighting, positioning, and hardware.
- Depth estimation does not provide guaranteed physical distance measurements.
- The system should not replace a white cane, guide dog, or other established mobility aids.
- Users should not rely exclusively on AI-generated information for safety-critical decisions.

## 👨‍💻 Developer

**Sudhanshu Raj**

B.Tech – Computer Science and Engineering  
AI & Machine Learning Enthusiast | Full Stack AI Developer

- GitHub: [@Sudhanshuraj1037](https://github.com/Sudhanshuraj1037)
- Portfolio: [sudhanshuraj1037.netlify.app](https://sudhanshuraj1037.netlify.app)

## 🤝 Contributions

Contributions, suggestions, and feedback are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement your changes.
4. Submit a pull request.

## 📜 License

This project is open for educational and research purposes. Refer to the repository's LICENSE file for applicable licensing terms.

---

<p align="center">
  <b>Building Intelligent Technology for a More Accessible World.</b>
  <br/>
  <i>Blind-AI — See the World Through Artificial Intelligence.</i>
</p>
