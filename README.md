# Conditional GAN for Audio Synthesis 🎶

This repository contains the code for **Task 2: A deep learning-based generative model for audio synthesis**. 
 The project implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) using PyTorch and torchaudio to generate realistic, category-specific audio clips. The model is trained on log-mel spectrograms and can produce novel audio samples for any of the predefined categories it was trained on.

This implementation is designed to be run in a Google Colab environment, leveraging its free GPU resources for efficient training.

## 🚀 Features

* **Conditional Generation**: The WGAN generates audio conditioned on a specific class label, allowing for targeted audio synthesis.
* **WGAN-GP Architecture**: Uses Wasserstein loss with gradient penalty for stable training and improved convergence compared to vanilla GANs.
* **Spectrogram-Based**: The model operates in the frequency domain by generating log-mel spectrograms, a robust representation for audio data.
* **Sample Generation**: At the end of every 20 epochs, the model saves generated audio samples (`.wav`) and plots of their corresponding spectrograms.
* **Two-Stage Workflow**: Separate preprocessing and training notebooks for efficient resource utilization.
* **Optimized Training Pipeline**: Includes mixed precision training (AMP), increased batch sizes, and pre-computed spectrograms for faster iteration as compared to provided sample code.
* **Checkpoint System**: Automatic saving and loading of model checkpoints to resume training seamlessly.
* **Google Colab Ready**: Includes necessary setup for mounting Google Drive to access datasets.

### 🗝️ Key Optimizations
* Precomputed Mel spectograms for faster processing.
* Increased Batch Size for stable training.
* Shorter Spectrograms for faster training.
* Enable the cuDNN auto-tuner to find the most efficient algorithms for our specific hardware and model configuration.
* Mixed Precision for faster training.
* Ignoring user warnings for cleaner output.

---

## 🏗️ Model Architecture

The project implements a WGAN-GP architecture with two main components:

### 1. Generator (The Audio Synthesizer 🎨)
The WGAN_Generator creates audio spectrograms from random noise and a label embedding:
```
Input: Noise (100D) + Label Embedding (16D)
  ↓
Linear Projection → Reshape
  ↓
ConvTranspose2d (512 → 256) + BatchNorm + ReLU
  ↓
ConvTranspose2d (256 → 128) + BatchNorm + ReLU
  ↓
ConvTranspose2d (128 → 64) + BatchNorm + ReLU
  ↓
ConvTranspose2d (64 → 1) + ReLU
  ↓
Output: Log-Mel Spectrogram (1×128×256)
```

### 2. Critic (The Detective 🕵️)
The WGAN_Critic evaluates the quality of spectrograms:
```
Input: Spectrogram (1×128×256) + Label Embedding (16D)
  ↓
Concatenate → (2×128×256)
  ↓
Conv2d (2 → 32) + LeakyReLU
  ↓
Conv2d (32 → 64) + LeakyReLU
  ↓
Conv2d (64 → 128) + LeakyReLU
  ↓
Conv2d (128 → 256) + LeakyReLU
  ↓
Conv2d (256 → 1)
  ↓
Output: Wasserstein Distance (scalar)
```

### WGAN-GP Training Details

* Gradient Penalty: Enforces Lipschitz constraint for stable training (weight: 10.0)
* Critic Updates: 5 critic updates per generator update
* Optimizer: Adam with learning rate 1e-4, betas (0.5, 0.9)
* Mixed Precision: Automatic mixed precision (AMP) for faster training on modern GPUs

---

### Prerequisites
* A Google Account (for Google Colab and Google Drive).
* Your audio dataset organized by category.

## 📂 Dataset Structure
```
drive/MyDrive/decibel_duel/
├── train/
│   ├── train/                    # Raw audio files
│   │   ├── dog_bark/
│   │   │   ├── audio_001.wav
│   │   │   ├── audio_002.wav
│   │   │   └── ...
│   │   ├── drilling/
│   │   │   ├── sound_A.wav
│   │   │   └── ...
│   │   ├── engine_idling/
│   │   ├── siren/
│   │   └── street_music/
│   │
│   └── precompute/              # Pre-computed spectrograms (generated)
│       ├── dog_bark/
│       ├── drilling/
│       └── ...
│
├── checkpoints/                 # Model checkpoints (auto-created)
│   └── wgan_audio.pth.tar
│
└── samples/                     # Generated samples (auto-created)
    ├── epoch_020.png
    └── gan_generated_audio/
```

## 📚 References

* [Base Sample Code](https://github.com/ankush-10010/Decibal-Duel)
* [Improved Training of WGANs](https://arxiv.org/abs/1704.00028)
* [PyTorch Documentation](https://pytorch.org/docs/)
* [Torchaudio Documentation](https://pytorch.org/audio/)

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
