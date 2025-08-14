# Audio Transcription (ASR) with LibriSpeech Dummy

## 📂 Project Structure

.
├── download_asr.py          # Downloads LibriSpeech dummy dataset via HuggingFace
├── utils.py                 # Preprocessing, metrics, checkpointing
├── models.py                # Simple ASR model (spectrogram + RNN/Transformer encoder)
├── train_asr.py             # Training loop for ASR
├── infer_asr.py             # Inference on audio files
├── asr_consolidated.ipynb   # Colab-ready consolidated notebook
└── README.md                # This file

## Requirement File

pip install -r requirements.txt 

## 📊 Dataset
- **Name**: `hf-internal-testing/librispeech_asr_dummy` (change to `librispeech_asr` for full set)
- **Splits**: train, validation
- **Format**: `.wav` audio files with `.txt` transcripts

RUN python download_asr.py

## 🧠 Model
- Simple spectrogram-based encoder-decoder
- Configurable hidden sizes, number of layers
- CTC loss for alignment-free training

## 🚀 Training

python train_asr.py --out_dir checkpoints_asr --epochs 10 --batch_size 8 --lr 1e-4


## 🔍 Inference

python infer_asr.py --ckpt checkpoints_asr/asr_best.pt --audio data/SpeechCommands/SpeechCommands/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav



