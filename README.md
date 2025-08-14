# Case Study – Data Scientist Assignment

This repository contains my complete solutions for the three-part Data Scientist case study.  
Each section contains modular Python scripts, a Colab-ready consolidated notebook, and a dedicated README.

---

## 📂 Repository Structure

Case-Study-DS/
├── sentiment_analysis/ # Section 1: Text Sentiment Classification (RNN + Transformer)
├── image_classification/ # Section 2: CIFAR-10 Image Classification (ResNet + ViT)
├── speech_to_text/ # Section 3: Audio-to-Text Transcription (ASR)
└── README.md # This file


---

## 📜 Sections Overview

### 1️⃣ Sentiment Analysis – IMDB Reviews
- **Models:** RNN (LSTM/GRU) & Transformer (from scratch)
- **Dataset:** [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Highlights:**  
  - Simple tokenizer & vocab builder (no external NLP libs)
  - Clean training loop & inference pipeline
- 📄 [Section README](sentiment_analysis/README.md)

---

### 2️⃣ Image Classification – CIFAR-10
- **Models:** ResNet & Vision Transformer (ViT)
- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Highlights:**  
  - Torchvision dataset loading with augmentations
  - Modular pipelines for both architectures
- 📄 [Section README](image_classification/README.md)

---

### 3️⃣ Speech-to-Text (ASR)
- **Model:** Simple spectrogram-based encoder-decoder with CTC loss
- **Dataset:** [LibriSpeech Dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) (for testing)
- **Highlights:**  
  - HuggingFace dataset integration
  - Modular preprocessing, training, and inference
- 📄 [Section README](speech_to_text/README.md)

---

## 🛠 How to Run
 REFER to individual README files of each section

 ## IPYNB

 Each section has a consolidated IPYNB Notebook for a breief overview
