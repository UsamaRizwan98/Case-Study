# Sentiment Analysis with RNN and Transformer

This project implements two deep learning approaches — an RNN-based model (LSTM/GRU) and a Transformer — to classify movie reviews from the IMDB dataset as **positive** or **negative**.

Both models are trained **from scratch** with minimal preprocessing to respect the “build from scratch” requirement in the case study.

---

## 📂 Project Structure

.
├── utils.py # Tokenization, vocabulary, dataset, training utilities
├── models.py # RNNClassifier & TransformerClassifier
├── train_sentiment.py # RNN (LSTM/GRU) training pipeline
├── train_transformer.py # Transformer training pipeline
├── infer_sentiment.py # Inference script
├── sentiment_analysis_rnn_transformer.ipynb # Colab-ready notebook
└── README.md # This file


---

## Requirement File

pip install -r requirements.txt 

## 📊 Dataset

I used the [Stanford IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) containing **50,000 reviews** labeled as positive or negative.

### Download & Extract
```bash
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz


🧠 Models
1. RNN-based Classifier


Configurable embedding size, hidden size, bidirectionality

Dropout regularization

2. Transformer-based Classifier

Simple Transformer encoder

Configurable embedding size, feedforward dimension, number of layers/heads

Position encoding included

🚀 Training
RNN Model
python train_sentiment.py --out_dir checkpoints_rnn --epochs 6 --batch_size 64 --rnn_type lstm --bidirectional

Transformer Model
python train_transformer.py --out_dir checkpoints_transformer --epochs 6 --batch_size 64


🔍 Inference

For LSTM(RNN):

python infer_sentiment.py --ckpt checkpoints_rnn/best_model.pt --text 'I loved the movie!'

Or for Transformer:

python infer_sentiment.py --ckpt checkpoints_transformer/transformer_best.pt --text 'This was amazing!'

