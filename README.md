<h1 align="center">audio-deepfake-guard</h1>

<p align="center">
  <strong>Desktop tool to detect audio deepfakes / synthetic speech</strong><br>
  <em>48 forensic audio features + XGBoost/LightGBM ensemble → segment-level probability + visual report</em>
</p>

<p align="center">
  Lightweight • Interpretable • No heavy deep learning models
</p>

<br>

## What it does

A PyQt5-based GUI application that analyzes audio files for signs of synthetic manipulation (deepfakes, voice cloning, TTS generation).

**Key features**

- Splits audio into short segments (~10 ms)
- Extracts **48 hand-crafted forensic features** (MFCC, spectral, pitch, formant, jitter/shimmer, band energy ratios, entropy, etc.)
- Classifies each segment with an **ensemble** of XGBoost + LightGBM
- Visualizes probabilities over time (line plot + heatmap)
- Highlights **top 5 most suspicious segments**
- Shows feature distributions + model consistency
- Exports full forensic report as **PDF** (with embedded plots)
- Supports CPU / CUDA / OpenCL (if available)

Built for scenarios where interpretability, low resource usage and clear reporting matter more than bleeding-edge EER.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your trained models
#    → xgb_model.pkl
#    → lgb_model.pkl
#    (in the same folder as main.py)

# 3. Run
python main.py
