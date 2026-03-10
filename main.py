import sys
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.feature
import numpy as np
import sqlite3
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QMessageBox, QProgressBar, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdfcanvas
import tempfile
import warnings
import psutil
import traceback

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

try:
    import pyopencl as cl
    opencl_available = True
except ImportError:
    opencl_available = False
    print("OpenCL nicht verfügbar (pyopencl nicht installiert)")

cuda_available = torch.cuda.is_available()
print(f"CUDA verfügbar: {cuda_available}")
print(f"OpenCL verfügbar: {opencl_available}")

XGB_MODEL_PATH = "xgb_model.pkl"
LGB_MODEL_PATH = "lgb_model.pkl"

feature_names = [
    "avg_amplitude", "zcr", "spectral_centroid", "spectral_bandwidth", "rolloff", "rmse",
    "mfcc_mean_0", "mfcc_mean_1", "mfcc_mean_2",
    "mfcc_var_0", "mfcc_var_1", "mfcc_var_2",
    "chroma_mean", "chroma_var", "tonnetz",
    "tempo", "onset_strength", "energy_entropy", "spectral_flatness", "spectral_contrast",
    "band_energy_ratio", "hnr", "temporal_centroid", "fundamental_frequency",
    "crest_factor", "silence_ratio", "perceptual_sharpness", "modulation_index",
    "spectral_entropy", "spectral_flux", "peak_amplitude", "skewness", "kurtosis",
    "harmonic_energy", "noise_energy", "mel_mean", "mel_var", "spectral_spread",
    "energy_50_150_hz", "energy_150_300_hz", "energy_300_1000_hz", "energy_1000_4000_hz",
    "formant_1", "formant_2", "pitch_std", "spectral_flux_bands_0_500_hz", "jitter", "shimmer"
]

def select_device(platform):
    print(f"[DEBUG] Wähle Gerät für Plattform: {platform}")
    if platform == "CUDA" and cuda_available:
        return torch.device("cuda")
    elif platform == "OpenCL" and opencl_available:
        platforms = cl.get_platforms()
        if platforms:
            platform = platforms[0]
            devices = platform.get_devices()
            context = cl.Context(devices)
            queue = cl.CommandQueue(context)
            return {"context": context, "queue": queue, "device": devices[0]}
    return torch.device("cpu")

def extract_48_features(seg_y, sr, device_info, n_fft=256):
    print(f"[DEBUG] extract_48_features: Segmentlänge: {len(seg_y)}, sr: {sr}, n_fft: {n_fft}")
    if not np.any(seg_y) or len(seg_y) < 2 or np.max(np.abs(seg_y)) < 1e-6:
        print(f"[ERROR] Segment ungültig: leer, konstant oder zu schwach")
        raise ValueError("Segment ist leer, konstant oder hat kein ausreichendes Signal.")
    
    n_fft = min(n_fft, len(seg_y))
    if n_fft < 16:
        print(f"[ERROR] n_fft zu klein: {n_fft}, Segmentlänge: {len(seg_y)}")
        raise ValueError(f"Segmentlänge {len(seg_y)} zu kurz für sinnvolle Feature-Extraktion.")
    
    try:
        waveform = torch.tensor(seg_y, dtype=torch.float32).unsqueeze(0).to(device_info)
        print(f"[DEBUG] Waveform shape: {waveform.shape}")
        
        n_mels = 40
        n_freqs = n_fft // 2 + 1
        if n_mels > n_freqs:
            n_mels = n_freqs
        print(f"[DEBUG] n_mels: {n_mels}, n_freqs: {n_freqs}")
        mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": n_fft, "n_mels": n_mels}).to(device_info)
        mfcc = mfcc_transform(waveform).squeeze(0).cpu().numpy()
        print(f"[DEBUG] MFCC shape: {mfcc.shape}")
        
        spectrogram_transform = T.Spectrogram(n_fft=n_fft).to(device_info)
        spectrogram = spectrogram_transform(waveform).squeeze(0).cpu().numpy()
        print(f"[DEBUG] Spectrogram shape: {spectrogram.shape}")
        power_spectrogram = spectrogram ** 2
        
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, n_mels=n_mels).to(device_info)
        mel_spectrogram = mel_spectrogram_transform(waveform).squeeze(0).cpu().numpy()
        print(f"[DEBUG] Mel-Spectrogram shape: {mel_spectrogram.shape}")
        
        features = {}
        features["avg_amplitude"] = np.mean(np.abs(seg_y))
        features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(seg_y))
        
        features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(S=power_spectrogram, sr=sr))
        features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(S=power_spectrogram, sr=sr))
        features["rolloff"] = np.mean(librosa.feature.spectral_rolloff(S=power_spectrogram, sr=sr, roll_percent=0.85))
        features["rmse"] = np.mean(librosa.feature.rms(y=seg_y))
        
        for i in range(3):
            features[f"mfcc_mean_{i}"] = np.mean(mfcc[i]) if mfcc.shape[0] > i else 0.0
            features[f"mfcc_var_{i}"] = np.var(mfcc[i]) if mfcc.shape[0] > i else 0.0
        
        features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(S=power_spectrogram, sr=sr))
        features["chroma_var"] = np.var(librosa.feature.chroma_stft(S=power_spectrogram, sr=sr))
        
        features["tonnetz"] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(seg_y), sr=sr))
        features["tempo"], _ = librosa.beat.beat_track(y=seg_y, sr=sr)
        features["onset_strength"] = np.mean(librosa.onset.onset_strength(y=seg_y, sr=sr, n_fft=n_fft))
        
        sum_abs_seg = np.sum(np.abs(seg_y))
        if sum_abs_seg < 1e-6:
            print(f"[ERROR] Summe der absoluten Werte zu klein: {sum_abs_seg}")
            raise ValueError("Summe der absoluten Werte zu klein, Energy Entropy nicht berechenbar.")
        prob = np.abs(seg_y) / sum_abs_seg
        features["energy_entropy"] = -np.sum(prob * np.log2(prob + 1e-6))
        
        features["spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(S=power_spectrogram))
        features["spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(S=power_spectrogram, sr=sr))
        
        low_energy = np.sum(np.abs(seg_y[:len(seg_y) // 2]) ** 2)
        high_energy = np.sum(np.abs(seg_y[len(seg_y) // 2:]) ** 2)
        features["band_energy_ratio"] = low_energy / (high_energy + 1e-6)
        
        harmonic = librosa.effects.harmonic(seg_y)
        noise = seg_y - harmonic
        features["hnr"] = np.mean(np.abs(harmonic) / (np.abs(noise) + 1e-6))
        
        sum_seg_squared = np.sum(seg_y ** 2)
        if sum_seg_squared < 1e-6:
            print(f"[ERROR] Summe der quadrierten Werte zu klein: {sum_seg_squared}")
            raise ValueError("Summe der quadrierten Werte zu klein, Temporal Centroid nicht berechenbar.")
        features["temporal_centroid"] = np.sum(np.arange(len(seg_y)) * (seg_y ** 2)) / sum_seg_squared
        
        pitches, magnitudes = librosa.piptrack(y=seg_y, sr=sr, n_fft=n_fft)
        features["fundamental_frequency"] = np.mean(pitches[pitches > 0]) if pitches.any() else 0
        
        features["crest_factor"] = np.max(np.abs(seg_y)) / (features["rmse"] + 1e-6)
        features["silence_ratio"] = np.sum(np.abs(seg_y) < 0.01) / len(seg_y)
        
        features["perceptual_sharpness"] = np.mean(features["spectral_flatness"] * features["spectral_centroid"])
        mean_abs_seg = np.mean(np.abs(seg_y))
        if mean_abs_seg < 1e-6:
            print(f"[ERROR] Mittelwert der absoluten Werte zu klein: {mean_abs_seg}")
            raise ValueError("Mittelwert der absoluten Werte zu klein, Modulation Index nicht berechenbar.")
        features["modulation_index"] = np.std(np.abs(seg_y)) / mean_abs_seg
        
        features["spectral_entropy"] = -np.sum(power_spectrogram * np.log2(power_spectrogram + 1e-6))
        features["spectral_flux"] = np.mean(np.diff(seg_y) ** 2)
        
        features["peak_amplitude"] = np.max(np.abs(seg_y))
        
        std_seg = np.std(seg_y)
        if std_seg < 1e-6:
            print(f"[ERROR] Standardabweichung zu klein: {std_seg}")
            raise ValueError("Standardabweichung zu klein, Skewness und Kurtosis nicht berechenbar.")
        features["skewness"] = np.mean(((seg_y - np.mean(seg_y)) / std_seg) ** 3)
        features["kurtosis"] = np.mean(((seg_y - np.mean(seg_y)) / std_seg) ** 4) - 3
        
        features["harmonic_energy"] = np.sum(harmonic ** 2)
        features["noise_energy"] = np.sum(noise ** 2)
        
        features["mel_mean"] = np.mean(mel_spectrogram)
        features["mel_var"] = np.var(mel_spectrogram)
        
        features["spectral_spread"] = np.sqrt(np.mean((seg_y - np.mean(seg_y)) ** 2))
        
        n_fft = min(n_fft, len(seg_y))
        fft = np.fft.rfft(seg_y, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1/sr)
        fft_magnitude = np.abs(fft) ** 2
        print(f"[DEBUG] fft_magnitude shape: {fft_magnitude.shape}, freqs shape: {freqs.shape}")
        
        total_energy = np.sum(fft_magnitude)
        if total_energy < 1e-6:
            print(f"[ERROR] Gesamtenergie zu klein: {total_energy}")
            raise ValueError("Gesamtenergie zu klein, Frequenzband-Energien nicht berechenbar.")
        
        features["energy_50_150_hz"] = np.sum(fft_magnitude[(freqs >= 50) & (freqs <= 150)]) / total_energy
        features["energy_150_300_hz"] = np.sum(fft_magnitude[(freqs >= 150) & (freqs <= 300)]) / total_energy
        features["energy_300_1000_hz"] = np.sum(fft_magnitude[(freqs >= 300) & (freqs <= 1000)]) / total_energy
        features["energy_1000_4000_hz"] = np.sum(fft_magnitude[(freqs >= 1000) & (freqs <= 4000)]) / total_energy
        
        f0, voiced_flag, _ = librosa.pyin(seg_y, fmin=50, fmax=500, sr=sr)
        features["pitch_std"] = np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0
        if np.any(voiced_flag) and len(f0[voiced_flag]) > 1:
            features["jitter"] = np.mean(np.abs(np.diff(f0[voiced_flag]))) / np.mean(f0[voiced_flag])
        else:
            features["jitter"] = 0.0
        
        rms_frames = librosa.feature.rms(y=seg_y, frame_length=n_fft, hop_length=n_fft//2)[0]
        if len(rms_frames) > 1 and np.mean(rms_frames) > 1e-6:
            features["shimmer"] = np.mean(np.abs(np.diff(rms_frames))) / np.mean(rms_frames)
        else:
            features["shimmer"] = 0.0
        
        lpc_coeffs = librosa.lpc(seg_y, order=2 * int(sr / 1000))
        roots = np.roots(lpc_coeffs)
        formants = sorted([abs(r) * sr / (2 * np.pi) for r in roots if abs(r.imag) > 0 and r.real > 0])
        features["formant_1"] = formants[0] if len(formants) > 0 else 0
        features["formant_2"] = formants[1] if len(formants) > 1 else 0
        
        features["spectral_flux_bands_0_500_hz"] = np.mean(np.diff(np.abs(fft[(freqs >= 0) & (freqs <= 500)]) ** 2)) if len(fft[(freqs >= 0) & (freqs <= 500)]) > 1 else 0
        
        feature_array = np.array([features[name] for name in feature_names])
        print(f"[DEBUG] Feature array shape: {feature_array.shape}")
        
        if len(feature_array) != 48:
            print(f"[ERROR] Feature-Länge: {len(feature_array)}, erwartet: 48")
            raise ValueError(f"Fehler: {len(feature_array)} Merkmale extrahiert, erwartet 48.")
        if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
            problematic_indices = np.where(np.isnan(feature_array) | np.isinf(feature_array))[0]
            print(f"[ERROR] Problematic features: {[feature_names[idx] for idx in problematic_indices]}")
            raise ValueError("Ungültige Werte (NaN oder inf) in den Features erkannt.")
        
        return feature_array
    except Exception as e:
        print(f"[ERROR] Fehler in extract_48_features: {str(e)}")
        print(traceback.format_exc())
        raise

def extract_all_metrics(audio_path, segment_length=0.01, hop_length=None, platform="CPU", device_info=None, max_segments=500, progress_callback=None):
    print(f"[DEBUG] extract_all_metrics: Audio: {audio_path}, segment_length: {segment_length}, max_segments: {max_segments}")
    try:
        waveform, sr = torchaudio.load(audio_path)
        print(f"[DEBUG] Waveform shape: {waveform.shape}, sr: {sr}, Audio-Länge: {waveform.shape[-1]/sr:.2f} Sekunden")
        if platform == "CUDA" and cuda_available:
            waveform = waveform.to(device_info)
        y = waveform.squeeze().cpu().numpy()
        
        segment_samples = int(segment_length * sr)
        hop_samples = segment_samples if hop_length is None else int(hop_length * sr)
        num_segments = (len(y) - segment_samples) // hop_samples + 1
        print(f"[DEBUG] segment_samples: {segment_samples}, hop_samples: {hop_samples}, num_segments: {num_segments}")
        
        segments = []
        for i in range(min(num_segments, max_segments)):
            start = i * hop_samples
            end = start + segment_samples
            if end <= len(y):
                seg = y[start:end]
            else:
                seg = np.pad(y[start:], (0, segment_samples - (len(y) - start)), mode='constant')[:segment_samples]
            if len(seg) != segment_samples:
                print(f"[ERROR] Segment {i} hat falsche Länge: {len(seg)}, erwartet: {segment_samples}")
                seg = np.pad(seg, (0, segment_samples - len(seg)), mode='constant')[:segment_samples]
            segments.append(seg)
            print(f"[DEBUG] Segment {i} Länge: {len(seg)}")
        
        if not segments:
            print(f"[ERROR] Keine Segmente extrahiert, Audio zu kurz")
            segments = [np.pad(y[:segment_samples], (0, max(0, segment_samples - len(y))), mode='constant')[:segment_samples]]
            print(f"[DEBUG] Fallback-Segment Länge: {len(segments[0])}")
        
        # Auffüllen der Segmente auf max_segments
        while len(segments) < max_segments:
            last_seg = segments[-1]
            print(f"[DEBUG] Länge von last_seg: {len(last_seg)}")
            if len(last_seg) != segment_samples:
                print(f"[ERROR] last_seg hat falsche Länge: {len(last_seg)}, erwartet: {segment_samples}")
                last_seg = np.pad(last_seg, (0, segment_samples - len(last_seg)), mode='constant')[:segment_samples]
            segments.append(last_seg.copy())
        
        print(f"[DEBUG] Anzahl Segmente: {len(segments)}, Segmentlänge: {segment_samples}")
        
        # Sicherstellen, dass alle Segmente die gleiche Länge haben
        segments = [seg[:segment_samples] if len(seg) > segment_samples else np.pad(seg, (0, segment_samples - len(seg)), mode='constant')[:segment_samples] for seg in segments]
        for i, seg in enumerate(segments):
            if len(seg) != segment_samples:
                print(f"[ERROR] Segment {i} hat nach Korrektur falsche Länge: {len(seg)}")
                raise ValueError(f"Segment {i} hat falsche Länge: {len(seg)}, erwartet: {segment_samples}")
        
        # Batch-Verarbeitung auf GPU
        waveform_batch = torch.stack([torch.tensor(seg, dtype=torch.float32) for seg in segments]).to(device_info)
        print(f"[DEBUG] Waveform batch shape: {waveform_batch.shape}")
        
        n_fft = min(256, segment_samples)
        n_mels = 40
        n_freqs = n_fft // 2 + 1
        if n_mels > n_freqs:
            n_mels = n_freqs
        print(f"[DEBUG] n_fft: {n_fft}, n_mels: {n_mels}, n_freqs: {n_freqs}")
        
        mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": n_fft, "n_mels": n_mels}).to(device_info)
        mfccs = mfcc_transform(waveform_batch).cpu().numpy()
        print(f"[DEBUG] MFCCs shape: {mfccs.shape}")
        
        spectrogram_transform = T.Spectrogram(n_fft=n_fft).to(device_info)
        spectrograms = spectrogram_transform(waveform_batch).cpu().numpy()
        print(f"[DEBUG] Spectrograms shape: {spectrograms.shape}")
        
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, n_mels=n_mels).to(device_info)
        mel_spectrograms = mel_spectrogram_transform(waveform_batch).cpu().numpy()
        print(f"[DEBUG] Mel-Spectrograms shape: {mel_spectrograms.shape}")
        
        all_segment_features = []
        for i, seg in enumerate(segments):
            try:
                print(f"[DEBUG] Verarbeite Segment {i}")
                power_spectrogram = spectrograms[i] ** 2
                features = extract_48_features(seg, sr, device_info, n_fft=n_fft)
                features["mfcc_mean_0"] = np.mean(mfccs[i, 0]) if mfccs.shape[1] > 0 else 0.0
                features["mfcc_var_0"] = np.var(mfccs[i, 0]) if mfccs.shape[1] > 0 else 0.0
                features["mfcc_mean_1"] = np.mean(mfccs[i, 1]) if mfccs.shape[1] > 1 else 0.0
                features["mfcc_var_1"] = np.var(mfccs[i, 1]) if mfccs.shape[1] > 1 else 0.0
                features["mfcc_mean_2"] = np.mean(mfccs[i, 2]) if mfccs.shape[1] > 2 else 0.0
                features["mfcc_var_2"] = np.var(mfccs[i, 2]) if mfccs.shape[1] > 2 else 0.0
                features["mel_mean"] = np.mean(mel_spectrograms[i])
                features["mel_var"] = np.var(mel_spectrograms[i])
                features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(S=power_spectrogram, sr=sr))
                features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(S=power_spectrogram, sr=sr))
                features["rolloff"] = np.mean(librosa.feature.spectral_rolloff(S=power_spectrogram, sr=sr, roll_percent=0.85))
                features["spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(S=power_spectrogram))
                features["spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(S=power_spectrogram, sr=sr))
                features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(S=power_spectrogram, sr=sr))
                features["chroma_var"] = np.var(librosa.feature.chroma_stft(S=power_spectrogram, sr=sr))
                features["spectral_entropy"] = -np.sum(power_spectrogram * np.log2(power_spectrogram + 1e-6))
                
                feature_array = np.array([features[name] for name in feature_names])
                print(f"[DEBUG] Segment {i} feature array shape: {feature_array.shape}")
                if len(feature_array) != 48:
                    print(f"[ERROR] Segment {i} Feature-Länge: {len(feature_array)}, erwartet: 48")
                    raise ValueError(f"Fehler: {len(feature_array)} Merkmale extrahiert, erwartet 48.")
                if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                    problematic_indices = np.where(np.isnan(feature_array) | np.isinf(feature_array))[0]
                    print(f"[ERROR] Segment {i} ungültige Werte: {[feature_names[idx] for idx in problematic_indices]}")
                    raise ValueError("Ungültige Werte (NaN oder inf) in den Features erkannt.")
                
                all_segment_features.append(feature_array)
                if progress_callback:
                    progress = int((i + 1) / len(segments) * 50)
                    progress_callback.emit(progress)
            except Exception as e:
                print(f"[ERROR] Segment {i} von {audio_path} übersprungen: {str(e)}")
                print(traceback.format_exc())
                continue
        
        if not all_segment_features:
            print(f"[ERROR] Keine validen Segmente für {audio_path} extrahiert")
            raise ValueError(f"Keine validen Segmente für {audio_path} extrahiert.")
        
        segment_features = np.array(all_segment_features)
        aggregated_features = segment_features.mean(axis=0)
        print(f"[DEBUG] Segment features shape: {segment_features.shape}, Aggregated features shape: {aggregated_features.shape}")
        
        return segment_features, aggregated_features
    except Exception as e:
        print(f"[ERROR] Fehler in extract_all_metrics: {str(e)}")
        print(traceback.format_exc())
        raise

def load_data_from_db():
    print("[DEBUG] Lade Daten aus Datenbank")
    try:
        deepfake_conn = sqlite3.connect("deepfakes.db")
        non_deepfake_conn = sqlite3.connect("non_deepfakes.db")
        
        deepfake_cursor = deepfake_conn.cursor()
        non_deepfake_cursor = non_deepfake_conn.cursor()
        
        deepfake_cursor.execute(f"SELECT {', '.join(feature_names)} FROM segment_features")
        non_deepfake_cursor.execute(f"SELECT {', '.join(feature_names)} FROM segment_features")
        
        deepfake_features = np.array(deepfake_cursor.fetchall())
        non_deepfake_features = np.array(non_deepfake_cursor.fetchall())
        print(f"[DEBUG] Deepfake features shape: {deepfake_features.shape}, Non-Deepfake features shape: {non_deepfake_features.shape}")
        
        if deepfake_features.shape[1] != 48 or non_deepfake_features.shape[1] != 48:
            print(f"[ERROR] Feature-Länge in der Datenbank: {deepfake_features.shape[1]} oder {non_deepfake_features.shape[1]}, erwartet: 48")
            raise ValueError("Feature-Länge in der Datenbank stimmt nicht mit erwarteten 48 überein!")
        
        X = np.vstack((deepfake_features, non_deepfake_features))
        y = np.hstack((np.ones(len(deepfake_features)), np.zeros(len(non_deepfake_features))))
        print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
        
        deepfake_conn.close()
        non_deepfake_conn.close()
        
        return X, y, 48
    except Exception as e:
        print(f"[ERROR] Fehler in load_data_from_db: {str(e)}")
        print(traceback.format_exc())
        raise

def train_models(X, y, platform):
    print("[DEBUG] Starte Modelltraining")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"[DEBUG] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
        if platform == "CUDA" and cuda_available:
            xgb_params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'gpu_hist',
                'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200
            }
            lgb_params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'device': 'gpu',
                'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 31, 'n_estimators': 200
            }
        else:
            xgb_params = {
                'objective': 'binary:logistic', 'eval_metric': 'logloss',
                'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200
            }
            lgb_params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 31, 'n_estimators': 200
            }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        
        xgb_pred = xgb_model.predict(X_test)
        lgb_pred = lgb_model.predict(X_test)
        
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'xgb_accuracy': accuracy_score(y_test, xgb_pred),
            'xgb_auc': roc_auc_score(y_test, xgb_probs),
            'xgb_cm': confusion_matrix(y_test, xgb_pred),
            'lgb_accuracy': accuracy_score(y_test, lgb_pred),
            'lgb_auc': roc_auc_score(y_test, lgb_probs),
            'lgb_cm': confusion_matrix(y_test, lgb_pred),
            'xgb_train_sizes': None,
            'xgb_train_scores': None,
            'xgb_test_scores': None,
            'lgb_train_sizes': None,
            'lgb_train_scores': None,
            'lgb_test_scores': None
        }
        
        train_sizes, xgb_train_scores, xgb_test_scores = learning_curve(
            xgb_model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        metrics['xgb_train_sizes'] = train_sizes
        metrics['xgb_train_scores'] = xgb_train_scores
        metrics['xgb_test_scores'] = xgb_test_scores
        
        train_sizes, lgb_train_scores, lgb_test_scores = learning_curve(
            lgb_model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        metrics['lgb_train_sizes'] = train_sizes
        metrics['lgb_train_scores'] = lgb_train_scores
        metrics['lgb_test_scores'] = lgb_test_scores
        
        with open(XGB_MODEL_PATH, 'wb') as f:
            pickle.dump(xgb_model, f)
        with open(LGB_MODEL_PATH, 'wb') as f:
            pickle.dump(lgb_model, f)
        
        print("[DEBUG] Modelle trainiert und gespeichert")
        return xgb_model, lgb_model, metrics
    except Exception as e:
        print(f"[ERROR] Fehler in train_models: {str(e)}")
        print(traceback.format_exc())
        raise

class FeatureExtractionThread(QThread):
    progress = pyqtSignal(int)
    audio_progress = pyqtSignal(int)
    current_audio = pyqtSignal(str)
    finished = pyqtSignal(list, list)

    def __init__(self, audio_paths, platform, category):
        super().__init__()
        self.audio_paths = audio_paths
        self.platform = platform
        self.category = category
        self.device_info = select_device(platform)
        self.setPriority(QThread.LowPriority)

    def run(self):
        print("[DEBUG] Starte FeatureExtractionThread")
        segment_features_list = []
        aggregated_features_list = []
        total_files = len(self.audio_paths)
        
        db_name = "deepfakes.db" if self.category == "Deepfakes" else "non_deepfakes.db"
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        feature_columns = ", ".join([f"{name} REAL" for name in feature_names])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS segment_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT,
                segment_index INTEGER,
                {feature_columns}
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS audio_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT UNIQUE,
                {feature_columns},
                valid_segments INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT,
                segment_index INTEGER,
                error_message TEXT
            )
        """)
        
        cursor.execute("SELECT audio_path FROM audio_features")
        processed_paths = set(row[0] for row in cursor.fetchall())
        
        current_run_paths = set()
        processed_count = 0
        
        unique_audio_paths = [path for path in self.audio_paths if path not in processed_paths]
        if len(unique_audio_paths) < len(self.audio_paths):
            skipped_count = len(self.audio_paths) - len(unique_audio_paths)
            print(f"[DEBUG] {skipped_count} Audios übersprungen, da bereits in der Datenbank")
        
        for i, audio_path in enumerate(unique_audio_paths):
            if audio_path in current_run_paths:
                processed_count += 1
                progress = int(processed_count / total_files * 100)
                self.progress.emit(progress)
                self.current_audio.emit(f"Überspringe: {audio_path} (bereits in dieser Sitzung analysiert)")
                continue
            
            self.current_audio.emit(f"Analysiere: {audio_path}")
            current_run_paths.add(audio_path)
            
            try:
                segment_features, aggregated_features = extract_all_metrics(
                    audio_path, platform=self.platform, device_info=self.device_info, progress_callback=self.audio_progress
                )
                valid_segments = len(segment_features)
                if valid_segments < max(1, len(segment_features) * 0.1):
                    print(f"[ERROR] Nur {valid_segments} von {len(segment_features)} Segmenten valide")
                    raise ValueError(f"Nur {valid_segments} von {len(segment_features)} Segmenten valide, Audio unbrauchbar.")
                
                segment_features_list.append(segment_features)
                aggregated_features_list.append(aggregated_features)
                
                placeholders = ", ".join(["?" for _ in range(48)])
                cursor.execute(
                    f"INSERT INTO audio_features (audio_path, {', '.join(feature_names)}, valid_segments) "
                    f"VALUES (?, {placeholders}, ?)",
                    (audio_path, *aggregated_features, valid_segments)
                )
                
                for idx, features in enumerate(segment_features):
                    cursor.execute(
                        f"INSERT INTO segment_features (audio_path, segment_index, {', '.join(feature_names)}) "
                        f"VALUES (?, ?, {placeholders})",
                        (audio_path, idx, *features)
                    )
                
                processed_paths.add(audio_path)
            except Exception as e:
                print(f"[ERROR] Fehler bei {audio_path}: {str(e)}")
                print(traceback.format_exc())
                cursor.execute(
                    "INSERT INTO error_log (audio_path, segment_index, error_message) VALUES (?, ?, ?)",
                    (audio_path, -1, str(e))
                )
                self.audio_progress.emit(100)
                continue
            
            processed_count += 1
            progress = int(processed_count / total_files * 100)
            self.progress.emit(progress)
            conn.commit()
        
        conn.close()
        self.finished.emit(segment_features_list, aggregated_features_list)

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, dict, int)
    
    def __init__(self, platform):
        super().__init__()
        self.platform = platform
        self.setPriority(QThread.LowPriority)
    
    def run(self):
        print("[DEBUG] Starte TrainingThread")
        self.progress.emit(10)
        try:
            X, y, target_length = load_data_from_db()
            self.progress.emit(30)
            xgb_model, lgb_model, metrics = train_models(X, y, self.platform)
            self.progress.emit(90)
            self.finished.emit(xgb_model, lgb_model, metrics, target_length)
        except Exception as e:
            print(f"[ERROR] Fehler in TrainingThread: {str(e)}")
            print(traceback.format_exc())
            self.finished.emit(None, None, {'error': str(e)}, 0)

class PredictionThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, float, float, float, str, np.ndarray, np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, audio_path, platform, xgb_model, lgb_model, target_length):
        super().__init__()
        self.audio_path = audio_path
        self.platform = platform
        self.device_info = select_device(platform)
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.target_length = target_length
        self.setPriority(QThread.LowPriority)

    def run(self):
        print("[DEBUG] Starte PredictionThread")
        try:
            segment_features, aggregated_features = extract_all_metrics(
                self.audio_path, platform=self.platform, device_info=self.device_info, progress_callback=self.progress
            )
            self.progress.emit(50)
            
            if segment_features.shape[1] != self.target_length:
                print(f"[ERROR] Feature-Länge: {segment_features.shape[1]}, erwartet: {self.target_length}")
                raise ValueError(f"Feature-Länge {segment_features.shape[1]} stimmt nicht mit erwarteten 48 überein!")
            
            xgb_probs = self.xgb_model.predict_proba(segment_features)[:, 1]
            self.progress.emit(75)
            lgb_probs = self.lgb_model.predict_proba(segment_features)[:, 1]
            self.progress.emit(90)
            
            avg_xgb_prob = np.mean(xgb_probs)
            avg_lgb_prob = np.mean(lgb_probs)
            avg_pred = (avg_xgb_prob + avg_lgb_prob) / 2
            result = "Deepfake" if avg_pred > 0.5 else "Kein Deepfake"
            
            self.progress.emit(100)
            self.finished.emit(self.audio_path, avg_xgb_prob, avg_lgb_prob, avg_pred, result, xgb_probs, lgb_probs, segment_features)
        except Exception as e:
            print(f"[ERROR] Fehler in PredictionThread: {str(e)}")
            print(traceback.format_exc())
            self.error.emit(f"Fehler bei der Deepfake-Prüfung: {str(e)}")

class DeepfakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Audio Scanner (48 Features - Segmentbasis)")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #2C2F33; }
            QWidget { background-color: #2C2F33; }
            QLabel { color: #FFFFFF; }
            QPushButton { background-color: #7289DA; color: white; border-radius: 8px; padding: 8px; }
            QPushButton:hover { background-color: #677BC4; }
            QComboBox { background-color: #36393F; color: #FFFFFF; border: 1px solid #7289DA; }
            QProgressBar { background-color: #36393F; color: #FFFFFF; border: 1px solid #7289DA; }
            QTabWidget { background-color: #2C2F33; }
            QTabBar::tab { background: #36393F; color: #FFFFFF; padding: 8px; }
            QTabBar::tab:selected { background: #7289DA; }
        """)
        self.xgb_model = None
        self.lgb_model = None
        self.target_length = 48
        self.last_prediction = None
        self.load_existing_models()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Tab 1: Steuerung
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        title = QLabel("Deepfake Audio Scanner (48 Features)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        control_layout.addWidget(title)
        
        self.load_btn = QPushButton("Audios laden")
        self.load_btn.clicked.connect(self.load_audios)
        control_layout.addWidget(self.load_btn)
        
        self.train_btn = QPushButton("Modelle trainieren")
        self.train_btn.clicked.connect(self.train_models)
        control_layout.addWidget(self.train_btn)
        
        self.predict_btn = QPushButton("Deepfake prüfen")
        self.predict_btn.clicked.connect(self.predict_audio)
        control_layout.addWidget(self.predict_btn)
        
        self.audio_label = QLabel("Keine Audios geladen")
        self.audio_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.audio_label)
        
        self.resource_label = QLabel("CPU: 0% | GPU: Inaktiv")
        self.resource_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.resource_label)
        
        self.platform_combo = QComboBox()
        platforms = ["CPU"]
        if cuda_available:
            platforms.append("CUDA")
        if opencl_available:
            platforms.append("OpenCL")
        self.platform_combo.addItems(platforms)
        control_layout.addWidget(QLabel("Berechnungsplattform:"))
        control_layout.addWidget(self.platform_combo)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Deepfakes", "Non-Deepfakes"])
        control_layout.addWidget(QLabel("Kategorie:"))
        control_layout.addWidget(self.category_combo)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(QLabel("Gesamtfortschritt:"))
        control_layout.addWidget(self.progress_bar)
        
        self.audio_progress_bar = QProgressBar()
        self.audio_progress_bar.setValue(0)
        control_layout.addWidget(QLabel("Fortschritt aktuelles Audio:"))
        control_layout.addWidget(self.audio_progress_bar)
        
        control_layout.addStretch()
        self.tabs.addTab(control_widget, "Steuerung")
        
        # Tab 2: Reporting
        self.report_widget = QWidget()
        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_scroll.setWidget(self.report_widget)
        self.report_layout = QVBoxLayout(self.report_widget)
        self.report_label = QLabel("Keine Prüfung durchgeführt. Bitte prüfe ein Audio, um das Reporting zu sehen.")
        self.report_label.setAlignment(Qt.AlignCenter)
        self.report_layout.addWidget(self.report_label)
        self.tabs.addTab(self.report_scroll, "Reporting")

    def load_existing_models(self):
        print("[DEBUG] Lade bestehende Modelle")
        if os.path.exists(XGB_MODEL_PATH):
            with open(XGB_MODEL_PATH, 'rb') as f:
                self.xgb_model = pickle.load(f)
            print(f"[DEBUG] XGBoost-Modell geladen von {XGB_MODEL_PATH}")
        
        if os.path.exists(LGB_MODEL_PATH):
            with open(LGB_MODEL_PATH, 'rb') as f:
                self.lgb_model = pickle.load(f)
            print(f"[DEBUG] LightGBM-Modell geladen von {LGB_MODEL_PATH}")

    def load_audios(self):
        print("[DEBUG] Starte load_audios")
        audio_paths, _ = QFileDialog.getOpenFileNames(self, "Audios auswählen", "", "Audio files (*.wav *.mp3 *.aac)")
        if audio_paths:
            self.audio_paths = list(set(audio_paths))
            self.audio_label.setText(f"Geladen: {len(self.audio_paths)} Audios")
            platform = self.platform_combo.currentText()
            category = self.category_combo.currentText()
            self.thread = FeatureExtractionThread(self.audio_paths, platform, category)
            self.thread.progress.connect(self.update_progress)
            self.thread.audio_progress.connect(self.update_audio_progress)
            self.thread.current_audio.connect(self.update_current_audio)
            self.thread.finished.connect(self.on_extraction_finished)
            self.thread.start()

    def train_models(self):
        print("[DEBUG] Starte train_models")
        platform = self.platform_combo.currentText()
        self.train_thread = TrainingThread(platform)
        self.train_thread.progress.connect(self.update_progress)
        self.train_thread.finished.connect(self.on_training_finished)
        self.train_thread.start()

    def predict_audio(self):
        print("[DEBUG] Starte predict_audio")
        audio_path, _ = QFileDialog.getOpenFileName(self, "Audio auswählen", "", "Audio files (*.wav *.mp3 *.aac)")
        if audio_path:
            self.audio_label.setText(f"Prüfe: {os.path.basename(audio_path)}")
            platform = self.platform_combo.currentText()
            self.predict_thread = PredictionThread(audio_path, platform, self.xgb_model, self.lgb_model, self.target_length)
            self.predict_thread.progress.connect(self.update_progress)
            self.predict_thread.finished.connect(self.on_prediction_finished)
            self.predict_thread.error.connect(self.on_prediction_error)
            self.predict_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        cpu_usage = psutil.cpu_percent()
        gpu_usage = "Aktiv" if cuda_available and self.platform_combo.currentText() == "CUDA" else "Inaktiv"
        self.resource_label.setText(f"CPU: {cpu_usage}% | GPU: {gpu_usage}")

    def update_audio_progress(self, value):
        self.audio_progress_bar.setValue(value)
        cpu_usage = psutil.cpu_percent()
        gpu_usage = "Aktiv" if cuda_available and self.platform_combo.currentText() == "CUDA" else "Inaktiv"
        self.resource_label.setText(f"CPU: {cpu_usage}% | GPU: {gpu_usage}")

    def update_current_audio(self, audio_path):
        self.audio_label.setText(audio_path)

    def on_extraction_finished(self, segment_features_list, aggregated_features_list):
        print("[DEBUG] Feature-Extraktion abgeschlossen")
        msg = QMessageBox()
        msg.setWindowTitle("Fertig")
        msg.setText(f"Feature-Extraktion (48 Features) für {len(segment_features_list)} von {len(self.audio_paths)} Audios abgeschlossen und in der Datenbank gespeichert!")
        msg.setStyleSheet("QLabel { color: #FFFFFF; } QMessageBox { background-color: #2C2F33; }")
        msg.exec_()
        self.audio_label.setText(f"Abgeschlossen: {len(segment_features_list)} Audios")
        self.progress_bar.setValue(100)
        self.audio_progress_bar.setValue(0)

    def on_training_finished(self, xgb_model, lgb_model, metrics, target_length):
        print("[DEBUG] Training abgeschlossen")
        if xgb_model is None or lgb_model is None:
            QMessageBox.critical(self, "Fehler", f"Training fehlgeschlagen!\nDetails: {metrics.get('error', 'Unbekannter Fehler')}")
            return
        
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.target_length = target_length
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(metrics['xgb_cm'], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Non-Deepfake", "Deepfake"], yticklabels=["Non-Deepfake", "Deepfake"])
        plt.title("Confusion Matrix for XGBoost (Segmentbasis)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(metrics['lgb_cm'], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Non-Deepfake", "Deepfake"], yticklabels=["Non-Deepfake", "Deepfake"])
        plt.title("Confusion Matrix for LightGBM (Segmentbasis)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        if metrics['xgb_train_sizes'] is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['xgb_train_sizes'], np.mean(metrics['xgb_train_scores'], axis=1), label="Training Score", color="blue")
            plt.fill_between(metrics['xgb_train_sizes'], 
                             np.mean(metrics['xgb_train_scores'], axis=1) - np.std(metrics['xgb_train_scores'], axis=1),
                             np.mean(metrics['xgb_train_scores'], axis=1) + np.std(metrics['xgb_train_scores'], axis=1), 
                             alpha=0.1, color="blue")
            plt.plot(metrics['xgb_train_sizes'], np.mean(metrics['xgb_test_scores'], axis=1), label="Cross-Validation Score", color="orange")
            plt.fill_between(metrics['xgb_train_sizes'], 
                             np.mean(metrics['xgb_test_scores'], axis=1) - np.std(metrics['xgb_test_scores'], axis=1),
                             np.mean(metrics['xgb_test_scores'], axis=1) + np.std(metrics['xgb_test_scores'], axis=1), 
                             alpha=0.1, color="orange")
            plt.title("Learning Curve for XGBoost (Segmentbasis)")
            plt.xlabel("Training Examples")
            plt.ylabel("Accuracy")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()
        
        if metrics['lgb_train_sizes'] is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['lgb_train_sizes'], np.mean(metrics['lgb_train_scores'], axis=1), label="Training Score", color="blue")
            plt.fill_between(metrics['lgb_train_sizes'], 
                             np.mean(metrics['lgb_train_scores'], axis=1) - np.std(metrics['lgb_train_scores'], axis=1),
                             np.mean(metrics['lgb_train_scores'], axis=1) + np.std(metrics['lgb_train_scores'], axis=1), 
                             alpha=0.1, color="blue")
            plt.plot(metrics['lgb_train_sizes'], np.mean(metrics['lgb_test_scores'], axis=1), label="Cross-Validation Score", color="orange")
            plt.fill_between(metrics['lgb_train_sizes'], 
                             np.mean(metrics['lgb_test_scores'], axis=1) - np.std(metrics['lgb_test_scores'], axis=1),
                             np.mean(metrics['lgb_test_scores'], axis=1) + np.std(metrics['lgb_test_scores'], axis=1), 
                             alpha=0.1, color="orange")
            plt.title("Learning Curve for LightGBM (Segmentbasis)")
            plt.xlabel("Training Examples")
            plt.ylabel("Accuracy")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()
        
        msg = QMessageBox()
        msg.setWindowTitle("Training abgeschlossen")
        msg.setText(
            f"Training fertig! Modelle wurden gespeichert.\n"
            f"XGBoost - Accuracy (Segmentbasis): {metrics['xgb_accuracy']:.4f}, AUC: {metrics['xgb_auc']:.4f}\n"
            f"LightGBM - Accuracy (Segmentbasis): {metrics['lgb_accuracy']:.4f}, AUC: {metrics['lgb_auc']:.4f}"
        )
        msg.setStyleSheet("QLabel { color: #FFFFFF; } QMessageBox { background-color: #2C2F33; }")
        msg.exec_()
        self.progress_bar.setValue(100)

    def on_prediction_finished(self, audio_path, avg_xgb_prob, avg_lgb_prob, avg_pred, result, xgb_probs, lgb_probs, segment_features):
        print("[DEBUG] Vorhersage abgeschlossen")
        self.audio_label.setText(f"Geprüft: {os.path.basename(audio_path)}")
        self.last_prediction = (audio_path, avg_xgb_prob, avg_lgb_prob, avg_pred, result, xgb_probs, lgb_probs, segment_features)
        self.update_reporting_tab()
        
        msg = QMessageBox()
        msg.setWindowTitle("Ergebnis")
        msg.setText(
            f"Datei: {os.path.basename(audio_path)}\n"
            f"XGBoost Wahrscheinlichkeit (Mittelwert): {avg_xgb_prob:.4f}\n"
            f"LightGBM Wahrscheinlichkeit (Mittelwert): {avg_lgb_prob:.4f}\n"
            f"Durchschnitt: {avg_pred:.4f}\n"
            f"Ergebnis: {result}"
        )
        msg.setStyleSheet("QLabel { color: #FFFFFF; } QMessageBox { background-color: #2C2F33; }")
        msg.exec_()
        self.progress_bar.setValue(100)

    def on_prediction_error(self, error_message):
        print(f"[ERROR] Vorhersagefehler: {error_message}")
        QMessageBox.warning(self, "Fehler", error_message)
        self.progress_bar.setValue(0)

    def update_reporting_tab(self):
        if self.last_prediction is None:
            return
        
        audio_path, avg_xgb_prob, avg_lgb_prob, avg_pred, result, xgb_probs, lgb_probs, segment_features = self.last_prediction
        
        for i in reversed(range(self.report_layout.count())): 
            item = self.report_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        
        title_label = QLabel("<h1>Deepfake-Analyse-Bericht</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFFFFF;")
        self.report_layout.addWidget(title_label)
        
        file_label = QLabel(f"<h2>Datei: {os.path.basename(audio_path)}</h2>")
        file_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(file_label)
        
        overview_label = QLabel(f"<h2>Ergebnisübersicht</h2>"
                                f"Das Audio wurde analysiert und als <b>{result}</b> klassifiziert.<br>"
                                f"Wahrscheinlichkeit: {avg_pred*100:.1f}% ({'Sehr sicher' if avg_pred > 0.9 or avg_pred < 0.1 else 'Sicher' if avg_pred > 0.7 or avg_pred < 0.3 else 'Unsicher'})")
        overview_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(overview_label)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(['XGBoost', 'LightGBM', 'Durchschnitt'], [avg_xgb_prob, avg_lgb_prob, avg_pred], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylim(0, 1)
        ax.set_title("Gesamtwahrscheinlichkeiten", fontsize=14, color="white")
        ax.set_ylabel("Wahrscheinlichkeit", color="white")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', color="white", weight='bold')
        ax.set_facecolor('#555555')
        fig.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        self.report_layout.addWidget(canvas)
        
        segment_label = QLabel("<h2>Segment-Analyse</h2>"
                               "Wie sich die Wahrscheinlichkeit über die Zeit des Audios verändert.")
        segment_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(segment_label)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xgb_probs, label="XGBoost", color="#1f77b4", alpha=0.8, linewidth=2)
        ax.plot(lgb_probs, label="LightGBM", color="#ff7f0e", alpha=0.8, linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_title("Wahrscheinlichkeit pro Segment", fontsize=14, color="white")
        ax.set_xlabel("Segment (Zeit)", color="white")
        ax.set_ylabel("Wahrscheinlichkeit", color="white")
        ax.legend(facecolor='#555555', edgecolor='white', labelcolor='white')
        ax.set_facecolor('#555555')
        fig.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        ax.grid(True, color="gray", linestyle="--", alpha=0.3)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(500)
        self.report_layout.addWidget(canvas)
        
        heatmap_label = QLabel("<h2>Verdächtige Stellen</h2>"
                               "Rot = hohe Wahrscheinlichkeit für Deepfake, Grün = wahrscheinlich echt.")
        heatmap_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(heatmap_label)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        avg_probs = (xgb_probs + lgb_probs) / 2
        sns.heatmap([avg_probs], cmap="RdYlGn_r", cbar=True, ax=ax, xticklabels=50, yticklabels=False)
        ax.set_title("Deepfake-Wahrscheinlichkeit über Segmente", fontsize=14, color="white")
        ax.set_xlabel("Segment (Zeit)", color="white")
        ax.set_facecolor('#555555')
        fig.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in cbar.get_ticks()], color='white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(300)
        self.report_layout.addWidget(canvas)
        
        top_label = QLabel("<h2>Top 5 verdächtige Segmente</h2>"
                           "Die Segmente mit der höchsten Deepfake-Wahrscheinlichkeit.")
        top_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(top_label)
        
        top_indices = np.argsort(avg_probs)[-5:][::-1]
        top_values = avg_probs[top_indices]
        top_times = [i * 0.01 for i in top_indices]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([f"Segment {i+1} ({t:.2f}s)" for i, t in enumerate(top_times)], top_values, color="#ff4d4d")
        ax.set_ylim(0, 1)
        ax.set_title("Top 5 Verdächtige Segmente", fontsize=14, color="white")
        ax.set_ylabel("Wahrscheinlichkeit", color="white")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', color="white", weight='bold')
        ax.set_facecolor('#555555')
        fig.set_facecolor('#2C2F33')
        ax.tick_params(colors='white', labelrotation=45)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        self.report_layout.addWidget(canvas)
        
        feature_label = QLabel("<h2>Feature-Verteilung</h2>"
                               "Verteilung ausgewählter Audio-Merkmale über alle Segmente.")
        feature_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(feature_label)
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        feature_indices = [0, 1, 2, 3]
        for i, ax in enumerate(axs.flat):
            sns.histplot(segment_features[:, feature_indices[i]], bins=30, ax=ax, color="#1f77b4", kde=True, edgecolor='white')
            ax.set_title(f"{feature_names[feature_indices[i]]}", fontsize=12, color="white")
            ax.set_xlabel("Wert", color="white")
            ax.set_ylabel("Häufigkeit", color="white")
            ax.set_facecolor('#555555')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
        fig.set_facecolor('#2C2F33')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(600)
        self.report_layout.addWidget(canvas)
        
        consistency_label = QLabel("<h2>Segment-Konsistenz</h2>"
                                   "Wie gleichmäßig ist die Einschätzung über die Segmente?")
        consistency_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(consistency_label)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([xgb_probs, lgb_probs], labels=["XGBoost", "LightGBM"], patch_artist=True, 
                   boxprops=dict(facecolor="#1f77b4"), medianprops=dict(color="white"))
        ax.set_title("Verteilung der Segment-Wahrscheinlichkeiten", fontsize=14, color="white")
        ax.set_ylabel("Wahrscheinlichkeit", color="white")
        ax.set_facecolor('#555555')
        fig.set_facecolor('#2C2F33')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        self.report_layout.addWidget(canvas)
        
        num_segments = len(xgb_probs)
        suspicious_segments = sum(1 for p in avg_probs if p > 0.7)
        summary_label = QLabel(f"<h2>Zusammenfassung</h2>"
                               f"Anzahl Segmente: {num_segments}<br>"
                               f"Verdächtige Segmente (über 70%): {suspicious_segments} ({suspicious_segments/num_segments*100:.1f}%)<br>"
                               f"Durchschnittliche Wahrscheinlichkeit: {avg_pred:.2f}<br>"
                               f"Varianz: {np.var(avg_probs):.3f} ({'Gleichmäßig' if np.var(avg_probs) < 0.05 else 'Variabel'})<br>"
                               f"Unsicherheit (Abweichung XGBoost/LightGBM): {abs(avg_xgb_prob - avg_lgb_prob):.3f}")
        summary_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")
        self.report_layout.addWidget(summary_label)
        
        export_btn = QPushButton("Bericht als PDF exportieren")
        export_btn.clicked.connect(self.export_to_pdf)
        export_btn.setStyleSheet("background-color: #7289DA; color: white; border-radius: 8px; padding: 8px;")
        self.report_layout.addWidget(export_btn)
        
        self.report_layout.addStretch()

    def export_to_pdf(self):
        if self.last_prediction is None:
            return
        
        audio_path, avg_xgb_prob, avg_lgb_prob, avg_pred, result, xgb_probs, lgb_probs, segment_features = self.last_prediction
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Bericht speichern", "", "PDF Files (*.pdf)")
        if not file_path:
            return
        
        c = pdfcanvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        y_pos = height - 50
        
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, y_pos, "Deepfake-Analyse-Bericht")
        y_pos -= 40
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, f"Datei: {os.path.basename(audio_path)}")
        y_pos -= 20
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_pos, "Ergebnisübersicht")
        y_pos -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, f"Ergebnis: {result}")
        y_pos -= 15
        c.drawString(50, y_pos, f"Wahrscheinlichkeit: {avg_pred*100:.1f}%")
        y_pos -= 15
        c.drawString(50, y_pos, f"XGBoost: {avg_xgb_prob:.2f}, LightGBM: {avg_lgb_prob:.2f}")
        y_pos -= 30
        
        temp_files = []
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(['XGBoost', 'LightGBM', 'Durchschnitt'], [avg_xgb_prob, avg_lgb_prob, avg_pred], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylim(0, 1)
        ax.set_title("Gesamtwahrscheinlichkeiten")
        ax.set_ylabel("Wahrscheinlichkeit")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, bbox_inches='tight')
        plt.close(fig)
        c.drawImage(temp_file.name, 50, y_pos - 300, width=400, height=300)
        temp_files.append(temp_file.name)
        c.showPage()
        
        y_pos = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_pos, "Segment-Analyse")
        y_pos -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, "Wahrscheinlichkeit pro Segment:")
        y_pos -= 20
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xgb_probs, label="XGBoost", color="#1f77b4")
        ax.plot(lgb_probs, label="LightGBM", color="#ff7f0e")
        ax.set_ylim(0, 1)
        ax.set_title("Wahrscheinlichkeit pro Segment")
        ax.set_xlabel("Segment (Zeit)")
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.legend()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, bbox_inches='tight')
        plt.close(fig)
        c.drawImage(temp_file.name, 50, y_pos - 300, width=500, height=300)
        temp_files.append(temp_file.name)
        c.showPage()
        
        y_pos = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_pos, "Verdächtige Stellen")
        y_pos -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, "Rot = hohe Wahrscheinlichkeit für Deepfake")
        y_pos -= 20
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap([avg_probs], cmap="RdYlGn_r", cbar=True, ax=ax)
        ax.set_title("Deepfake-Wahrscheinlichkeit über Segmente")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, bbox_inches='tight')
        plt.close(fig)
        c.drawImage(temp_file.name, 50, y_pos - 150, width=500, height=150)
        temp_files.append(temp_file.name)
        c.showPage()
        
        y_pos = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_pos, "Top 5 Verdächtige Segmente")
        y_pos -= 20
        c.setFont("Helvetica", 12)
        top_indices = np.argsort(avg_probs)[-5:][::-1]
        top_values = avg_probs[top_indices]
        top_times = [i * 0.01 for i in top_indices]
        for i, (time, value) in enumerate(zip(top_times, top_values)):
            c.drawString(50, y_pos, f"Segment {i+1} ({time:.2f}s): {value:.2f}")
            y_pos -= 15
        c.showPage()
        
        y_pos = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_pos, "Zusammenfassung")
        y_pos -= 20
        c.setFont("Helvetica", 12)
        num_segments = len(xgb_probs)
        suspicious_segments = sum(1 for p in avg_probs if p > 0.7)
        c.drawString(50, y_pos, f"Anzahl Segmente: {num_segments}")
        y_pos -= 15
        c.drawString(50, y_pos, f"Verdächtige Segmente (>70%): {suspicious_segments}")
        y_pos -= 15
        c.drawString(50, y_pos, f"Durchschnittliche Wahrscheinlichkeit: {avg_pred:.2f}")
        y_pos -= 15
        c.drawString(50, y_pos, f"Varianz: {np.var(avg_probs):.3f}")
        
        c.save()
        
        for temp_file in temp_files:
            os.unlink(temp_file)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeApp()
    window.show()
    sys.exit(app.exec_())