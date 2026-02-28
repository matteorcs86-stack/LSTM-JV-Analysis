import logging
from typing import Any, Dict, List, Tuple

# ------------------------------------------------------------
# Matplotlib: backend e silenziamento warning sui font
# ------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # backend non interattivo

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
from pathlib import Path
from scipy.io import savemat

# Stili scienceplots
import scienceplots

# Silenzia i WARNING del font manager (che escono come logging, non come warnings)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Applica stile IEEE e POI sovrascrivi i font per evitare "Times"
plt.style.use('ieee')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.serif": [],                 # ← svuota serif per evitare ricerca di Times
    "mathtext.fontset": "dejavusans", # formula inline coerente con sans-serif
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------
# Export TikZ: disabilitato per evitare errori (riabilita quando serve)
# ------------------------------------------------------------
SAVE_TIKZ = False
try:
    import tikzplotlib
    _tikz_ok = True
except Exception:
    _tikz_ok = False

def safe_save_plot(filename_base: str):
    """Salva PNG sempre. TikZ disabilitato per default (niente errori in console)."""
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
    if SAVE_TIKZ and _tikz_ok:
        try:
            tikzplotlib.save(f"{filename_base}.tex")
        except Exception:
            # Silenzioso: non stampiamo nulla in console
            pass

# ------------------------------------------------------------
# Resto delle dipendenze del progetto
# ------------------------------------------------------------
from scipy.signal import find_peaks
from src.dataloader.csv_dataset import CSVDataset
from src.utils import denormalise

logger = logging.getLogger('pytorch_lightning')

# ============================================================
# 1) Metriche base
# ============================================================
def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return np.sqrt(np.mean((actuals - predictions) ** 2))

def fit_index(predictions: np.ndarray, actuals: np.ndarray) -> List[float]:
    fit_values = []
    for ii in range(1, len(predictions) + 1):
        current_pred = predictions[:ii]
        current_act = actuals[:ii]
        denom = np.linalg.norm(current_act - np.mean(current_act))
        if denom == 0:
            fit = 0
        else:
            num = np.linalg.norm(current_act - current_pred)
            fit = 100.0 * (1 - (num / denom))
        fit_values.append(fit)
    return fit_values

# ============================================================
# 2) Estremi locali e segmenti soglia 75%
# ============================================================
def find_all_threshold_segments(real: np.ndarray, prominence: float = 0.01
                               ) -> Tuple[List[Tuple[int,int,int]], List[Tuple[int,int,int]]]:
    peaks  = find_peaks(real,  prominence=prominence)[0]
    nadirs = find_peaks(-real, prominence=prominence)[0]
    extrema = sorted([(i,'peak') for i in peaks] + [(i,'nadir') for i in nadirs], key=lambda x: x[0])

    up, down = [], []
    for i in range(len(extrema) - 1):
        idx0, t0 = extrema[i]
        idx1, t1 = extrema[i+1]
        if t0 == 'nadir' and t1 == 'peak':
            t_nadir, t_peak = idx0, idx1
            t_p75 = int(round(t_nadir + 0.75*(t_peak - t_nadir)))
            up.append((t_nadir, t_peak, t_p75))
        elif t0 == 'peak' and t1 == 'nadir':
            t_peak, t_nadir = idx0, idx1
            t_n75 = int(round(t_peak - 0.75*(t_peak - t_nadir)))
            down.append((t_peak, t_nadir, t_n75))
    return up, down

# ============================================================
# 3) Ritardi segmentali/medi
# ============================================================
def compute_upward_delay_segment(real, pred, t_nadir, t_p75, max_delay=10) -> int:
    best_d, best_mse = 0, float('inf')
    if t_p75 - t_nadir <= 0:
        return 0
    for d in range(max_delay + 1):
        if t_p75 + d > len(pred):
            break
        mse = np.mean((real[t_nadir:t_p75] - pred[t_nadir + d:t_p75 + d]) ** 2)
        if mse < best_mse:
            best_mse, best_d = mse, d
    return best_d

def compute_downward_delay_segment(real, pred, t_peak, t_n75, max_delay=10) -> int:
    best_d, best_mse = 0, float('inf')
    if t_n75 - t_peak <= 0:
        return 0
    for d in range(max_delay + 1):
        if t_n75 + d > len(pred):
            break
        mse = np.mean((real[t_peak:t_n75] - pred[t_peak + d:t_n75 + d]) ** 2)
        if mse < best_mse:
            best_mse, best_d = mse, d
    return best_d

def compute_average_upward_delay(real, pred, max_delay=8, prominence=0.01) -> float:
    up, _ = find_all_threshold_segments(real, prominence=prominence)
    delays = [compute_upward_delay_segment(real, pred, t0, t75, max_delay) for (t0, _, t75) in up]
    return np.mean(delays) if delays else 0.0

def compute_average_downward_delay(real, pred, max_delay=8, prominence=0.01) -> float:
    _, down = find_all_threshold_segments(real, prominence=prominence)
    delays = [compute_downward_delay_segment(real, pred, tp, tn75, max_delay) for (tp, _, tn75) in down]
    return np.mean(delays) if delays else 0.0

# ============================================================
# 4) Plot predizioni
# ============================================================
def plot_predictions(experiment_path: str,
                     dataset: CSVDataset,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     output_features: list,
                     t_initial: int, t_final: int,
                     save_folder_name: str = "predictions",
                     data_split: str = "test") -> None:

    logger.info("Checking input data:")
    logger.info(f"predictions shape: {predictions.shape}")
    logger.info(f"targets shape: {targets.shape}")
    logger.info(f"output_features: {output_features}")
    logger.info(f"t_initial: {t_initial}, t_final: {t_final}")

    save_dir = Path(experiment_path) / save_folder_name / data_split
    save_dir.mkdir(parents=True, exist_ok=True)

    prediction_sample = predictions.shape[1] - 1

    scaling_factors_min = dataset.scaling_factors_min[dataset.indices_output_features]
    scaling_factors_max = dataset.scaling_factors_max[dataset.indices_output_features]

    profiles_hat = denormalise(
        predictions[:, prediction_sample, :],
        scaling_factors_min,
        scaling_factors_max,
    )
    profiles_targets = denormalise(
        targets[:, prediction_sample, :],
        scaling_factors_min,
        scaling_factors_max,
    )

    # Denormalise the full sequences for export
    profiles_hat_full = denormalise(
        predictions,
        scaling_factors_min,
        scaling_factors_max,
    )
    profiles_targets_full = denormalise(
        targets,
        scaling_factors_min,
        scaling_factors_max,
    )

    
    # Ricava lo site_id dal dataset
    site_id = getattr(dataset, "site_id", None)

    if site_id is None:
        # fallback, nel raro caso che site_id non sia salvato come attributo
        site_id = "unknown"

    mat_filename = save_dir / f"predictions_vs_targets_site_{site_id:03d}.mat"
    
    # Save full sequences to MATLAB format
    savemat(
        mat_filename,
       {
            "profiles_hat": profiles_hat_full.detach().cpu().numpy(),
            "profiles_targets": profiles_targets_full.detach().cpu().numpy(),
       },
    )

    if len(output_features) == 1:
        feature = output_features[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(feature)
        ax.grid()

        real = profiles_targets[t_initial:t_final, 0].detach().cpu().numpy()
        pred = profiles_hat[t_initial:t_final, 0].detach().cpu().numpy()

        ax.plot(real, label="Ground Truth", color="blue", linewidth=1.0)
        ax.plot(pred, label="Prediction",  color="green", linewidth=1.0)
        ax.legend()

        safe_save_plot(save_dir / f"predictions_{feature}")
        plt.close(fig)

        fit_vals = fit_index(pred, real)
        logger.info(f"[{feature}] RMSE: {calculate_rmse(pred, real):.4f}")
        logger.info(f"[{feature}] Average FIT: {np.mean(fit_vals):.2f}% | Final FIT: {fit_vals[-1]:.2f}%")
        logger.info(f"[{feature}] Average Upward Delay: {compute_average_upward_delay(real, pred):.2f} samples")
        logger.info(f"[{feature}] Average Downward Delay: {compute_average_downward_delay(real, pred):.2f} samples")

    else:
        fig, axs = plt.subplots(len(output_features), 1, figsize=(10, 6 * len(output_features)))
        fig.suptitle("Predictions vs. Ground Truth")
        for i, feature in enumerate(output_features):
            axs[i].set_title(feature)
            axs[i].grid()
            real = profiles_targets[t_initial:t_final, i].detach().cpu().numpy()
            pred = profiles_hat[t_initial:t_final, i].detach().cpu().numpy()
            axs[i].plot(real, label="Ground Truth", color="blue", linewidth=1.0)
            axs[i].plot(pred, label="Prediction",  color="green", linewidth=1.0)
            axs[i].legend()
        safe_save_plot(save_dir / "predictions_multifeature")
        plt.close(fig)

# ============================================================
# 5) Traiettoria glicemica
# ============================================================
def plot_blood_glucose_trajectory(experiment_path: str,
                                  state: torch.Tensor,
                                  t_initial: int, t_final: int,
                                  save_folder_name: str = "trajectory") -> None:
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    if state.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title('Blood Glucose Level')
        ax.plot(state[t_initial:t_final, 0].cpu().numpy(), label='BloodGlucose')
        ax.grid()
        ax.legend()
    else:
        fig, axs = plt.subplots(state.shape[1], figsize=(15, 10))
        for i in range(state.shape[1]):
            axs[i].set_title('Blood Glucose Level')
            axs[i].plot(state[t_initial:t_final, i].cpu().numpy(), label='BloodGlucose')
            axs[i].grid()
            axs[i].legend()

    safe_save_plot(os.path.join(save_directory, "blood_glucose_trajectory"))
    plt.close()

# ============================================================
# 6) Loss di training
# ============================================================
def plot_training_loss(experiment_path: str, 
                       training_losses: list,
                       save_folder_name: str = "training_loss") -> None:
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time Steps')
    plt.legend()
    plt.grid(True)

    safe_save_plot(os.path.join(save_directory, 'training_loss_plot'))
    plt.close()