# Build text-to-speech from scratch

This folder contains the recipes for training TTS systems with the popular LJSpeech dataset.
# 🔧 Project Requirements Recap (Confirmed)
| Requirement             | Choice                                             |
| ----------------------- | -------------------------------------------------- |
| Language support        | **Single language** (assume English for now)       |
| Emotions/Styles/Accents | ✅ Yes                                              |
| Real-time synthesis     | ✅ Yes (also supports non-real-time)                |
| Target platform         | **Web** (browser-based UI, backend locally hosted) |
| Deployment mode         | **Local-only** (no cloud inference)                |
| Build from scratch      | ✅ Yes (not using full toolkits like Coqui)         |

# TECH STACK
| Component           | Recommended Model             | Justification                                                        |
| ------------------- | ----------------------------- | -------------------------------------------------------------------- |
| **Text → Phonemes** | `g2p-seq2seq` or `phonemizer` | Converts text to phonemes, required for better pronunciation control |
| **Acoustic Model**  | `FastSpeech 2`                | High-quality, fast, supports prosody and style control               |
| **Vocoder**         | `HiFi-GAN v1`                 | Real-time-capable, lightweight, produces very high-quality audio     |
| **Frontend**        | `React + Web Audio API`       | Web interface to record/play voice, send text to local backend       |
| **Backend**         | `FastAPI + PyTorch`           | Async API for serving the model locally                              |

# Dataset
The dataset can be downloaded from here:

# Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
