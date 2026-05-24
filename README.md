# Kanjo Kenshutsu

### Decoding Human Emotion through Adaptive Multimodal Intelligence

---

## System Architecture

![System Architecture](final/assets/architecture_image.png)
---

## Overview

Kanjo Kenshutsu is an adaptive multimodal emotion recognition framework that integrates self-supervised speech representation learning with contextual text modelling for robust emotion classification.

The system combines:

* WavLM-based speech embeddings
* RoBERTa-based contextual text embeddings
* Adaptive gated multimodal fusion
* Transformer-based temporal and contextual modelling

to analyze and classify human emotions from speech and textual information.

The framework was experimentally evaluated on the TESS (Toronto Emotional Speech Set) dataset under:

* Speech-only
* Text-only
* Multimodal fusion

configurations to study modality contribution and adaptive fusion behavior.

---

## Key Features

* Self-supervised speech representation learning using WavLM
* Contextual semantic modelling using RoBERTa
* Adaptive gated multimodal fusion
* Transformer-based temporal modelling
* Emotion-wise classification analysis
* Fusion gate interpretability
* Representation separability analysis using t-SNE / UMAP
* Error analysis and modality informativeness study

---

## Architecture Overview

The framework consists of three major pipelines:

### 1. Speech Processing Pipeline

* Audio preprocessing
* WavLM speech encoder
* Transformer temporal modelling
* Attention pooling
* Speech emotion embeddings

### 2. Text Processing Pipeline

* Whisper ASR transcription
* Tokenization and preprocessing
* RoBERTa contextual encoder
* Contextual semantic embeddings

### 3. Adaptive Fusion Pipeline

* Dynamic modality weighting
* Attention-based multimodal fusion
* Fully connected classifier
* Softmax emotion prediction

---

## Technologies Used

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Deep Learning Framework | PyTorch                   |
| Speech Encoder          | WavLM                     |
| Text Encoder            | RoBERTa                   |
| ASR Model               | Whisper                   |
| NLP Library             | Hugging Face Transformers |
| Audio Processing        | Librosa, Torchaudio       |
| Visualization           | Matplotlib, UMAP, t-SNE   |
| GPU Support             | CUDA                      |

---

## Dataset

### Toronto Emotional Speech Set (TESS)

* 7 emotion classes
* 2800 audio samples
* Studio-quality emotional speech recordings
* English speech dataset

Emotion Classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

---

## Experimental Results

| Model             | Validation Accuracy | Validation Macro F1 |
| ----------------- | ------------------- | ------------------- |
| Speech-Only       | 99.29%              | 99.29%              |
| Text-Only         | 14.29%              | 3.57%               |
| Multimodal Fusion | 99.29%              | 99.29%              |

---

## Key Experimental Findings

* Speech-only modelling achieved near-perfect performance on TESS.
* Text-only modelling collapsed to near-random prediction behavior.
* Adaptive gated fusion consistently prioritized speech representations.
* The model autonomously learned speech-dominant multimodal behavior.
* TESS transcript semantics contributed minimal emotional information.

---

## Scientific Insights

The experiments revealed that the TESS dataset is strongly speech-dominant, where emotional information is primarily encoded through acoustic and prosodic speech characteristics rather than textual semantics.

Fusion gate analysis demonstrated:

* ~81% speech contribution
* ~19% text contribution

showing that the adaptive fusion mechanism dynamically learned modality importance instead of assuming equal multimodal contribution.

---

## Future Scope

* Multilingual and accent-aware emotion recognition
* Real-time streaming inference
* Cross-corpus generalization
* Additional modalities such as facial expressions and video
* Explainable AI integration
* Lightweight deployment optimization

---

## References

1. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
2. RoBERTa: A Robustly Optimized BERT Pretraining Approach
3. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
4. Multimodal Fusion for Speech-Text Emotion Recognition
5. M4SER: Multimodal Multi-representation Multitask and Multistrategy Learning for SER

---

## Author

**Kushal Manikonda**  
Department of Information Technology  
Vasavi College of Engineering  
Hyderabad, India
