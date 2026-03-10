<h1 align="center">voice-deepfake-detector-gui</h1>

<p align="center">
  <strong>Desktop forensic tool to hunt synthetic voices</strong><br>
  <em>48 hand-crafted forensic features → XGBoost + LightGBM ensemble → segment-level deepfake scoring + PDF forensic report</em>
</p>

<p align="center">
  <strong>NO DEEP LEARNING. NO BULLSHIT. JUST HARD EVIDENCE.</strong><br>
  Lightweight • interpretable • pinpoints suspicious parts • runs on almost anything
</p>

<br>

## What it does

A clean PyQt5 GUI that tears audio files apart and tells you exactly which parts smell like synthetic speech (TTS, voice cloning, deepfake audio).

**Core features**

- Splits audio into ~10 ms segments  
- Extracts **48 forensic audio features** (MFCCs, jitter/shimmer, formants, band energy ratios, spectral entropy, modulation index, HNR, crest factor, …)  
- Classifies every segment using an **XGBoost + LightGBM ensemble** (averaged probabilities)  
- Visualizes: probability timeline, heatmap (RdYlGn_r), top-5 most suspicious segments, feature distributions, model agreement  
- Exports a **complete PDF forensic report** with embedded plots (ready for court, journalism, security, call-center audits)  
- Supports CPU / CUDA / OpenCL acceleration for feature extraction where available  
- Built-in **training tab**: load your SQLite database → extract features → train new models → save .pkl files → use instantly  

No 2 GB models. No internet dependency. No black-box nonsense. Just features + boosting.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Provide pre-trained models (or train your own)
#    → xgb_model.pkl
#    → lgb_model.pkl
#    (place them next to main.py)

# 3. Launch
python main.py
