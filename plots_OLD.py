import logging
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
from pathlib import Path
from src.dataloader.csv_dataset import CSVDataset
from src.utils import denormalise
import pandas as pd
import os
import tikzplotlib

logger = logging.getLogger('pytorch_lightning')
plt.style.use('ieee')

# Use scipy to detect local extrema
from scipy.signal import find_peaks

###############################################################################
# 1) Basic Metrics: RMSE & FIT
###############################################################################
def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between predictions and actual values.
    """
    return np.sqrt(np.mean((actuals - predictions) ** 2))


def fit_index(predictions: np.ndarray, actuals: np.ndarray) -> List[float]:
    """
    Compute a running FIT index for each time step using data up to that point.
    Returns a list of FIT percentages.
    """
    fit_values = []
    for ii in range(1, len(predictions) + 1):
        current_pred = predictions[:ii]
        current_act = actuals[:ii]
        denominator = np.linalg.norm(current_act - np.mean(current_act))
        if denominator == 0:
            fit = 0
        else:
            numerator = np.linalg.norm(current_act - current_pred)
            fit = 100.0 * (1 - (numerator / denominator))
        fit_values.append(fit)
    return fit_values

###############################################################################
# 2) Detecting Multiple Local Extrema and Computing 75% Threshold Indices
###############################################################################
def find_all_threshold_segments(real: np.ndarray, prominence: float = 20
                               ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Detect all local extrema (peaks and nadirs) in the time series, and then segment
    the data into upward and downward trends. For each upward segment (from a local minimum
    to the following local maximum), compute the 75% threshold index:
        t_p75 = t_nadir + 0.75 * (t_peak - t_nadir)
    For each downward segment (from a local maximum to the following local minimum), compute:
        t_n75 = t_peak - 0.75 * (t_peak - t_nadir)
    
    Args:
        real: 1D array of actual values.
        prominence: The minimum prominence required to consider a peak/nadir.
        
    Returns:
        upward_segments: List of tuples (t_nadir, t_peak, t_p75) for upward trends.
        downward_segments: List of tuples (t_peak, t_nadir, t_n75) for downward trends.
    """
    # Detect local peaks (maxima)
    peaks = find_peaks(real, prominence=prominence)[0]
    # Detect local nadirs (minima) by finding peaks in -real
    nadirs = find_peaks(-real, prominence=prominence)[0]

    # Merge and sort all extrema by index
    extrema = []
    for idx in peaks:
        extrema.append((idx, 'peak'))
    for idx in nadirs:
        extrema.append((idx, 'nadir'))
    extrema.sort(key=lambda x: x[0])

    upward_segments = []
    downward_segments = []

    # Iterate over consecutive extrema to form segments
    for i in range(len(extrema) - 1):
        current_idx, current_type = extrema[i]
        next_idx, next_type = extrema[i + 1]
        if current_type == 'nadir' and next_type == 'peak':
            # Upward segment
            t_nadir = current_idx
            t_peak = next_idx
            t_p75 = int(round(t_nadir + 0.75 * (t_peak - t_nadir)))
            upward_segments.append((t_nadir, t_peak, t_p75))
        elif current_type == 'peak' and next_type == 'nadir':
            # Downward segment
            t_peak = current_idx
            t_nadir = next_idx
            t_n75 = int(round(t_peak - 0.75 * (t_peak - t_nadir)))
            downward_segments.append((t_peak, t_nadir, t_n75))
    return upward_segments, downward_segments

###############################################################################
# 3) Computing Upward and Downward Delay for a Segment
###############################################################################
def compute_upward_delay_segment(real: np.ndarray, pred: np.ndarray,
                                 t_nadir: int, t_p75: int, max_delay: int = 10) -> int:
    """
    For an upward segment (from a local minimum to the 75% threshold of the following peak),
    find the integer shift (in sample units) that minimizes the Mean Squared Error (MSE)
    between the actual and shifted predicted values.
    
    Args:
        real: Array of actual values.
        pred: Array of predicted values.
        t_nadir: Starting index of the upward segment (local minimum).
        t_p75: Ending index of the upward segment (75% threshold).
        max_delay: Maximum shift (in samples) to consider.
        
    Returns:
        int: The delay (in sample units) that minimizes the MSE.
    """
    best_delay = 0
    best_mse = float('inf')
    window_length = t_p75 - t_nadir
    if window_length <= 0:
        return 0

    for delay in range(max_delay + 1):
        if t_p75 + delay > len(pred):
            break
        mse = np.mean((real[t_nadir:t_p75] - pred[t_nadir + delay:t_p75 + delay]) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_delay = delay
    return best_delay


def compute_downward_delay_segment(real: np.ndarray, pred: np.ndarray,
                                   t_peak: int, t_n75: int, max_delay: int = 10) -> int:
    """
    For a downward segment (from a local maximum to the 75% threshold of the following nadir),
    find the integer shift (in sample units) that minimizes the Mean Squared Error (MSE)
    between the actual and shifted predicted values.
    
    Args:
        real: Array of actual values.
        pred: Array of predicted values.
        t_peak: Starting index of the downward segment (local maximum).
        t_n75: Ending index of the downward segment (75% threshold).
        max_delay: Maximum shift (in samples) to consider.
        
    Returns:
        int: The delay (in sample units) that minimizes the MSE.
    """
    best_delay = 0
    best_mse = float('inf')
    window_length = t_n75 - t_peak
    if window_length <= 0:
        return 0

    for delay in range(max_delay + 1):
        if t_n75 + delay > len(pred):
            break
        mse = np.mean((real[t_peak:t_n75] - pred[t_peak + delay:t_n75 + delay]) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_delay = delay
    return best_delay


def compute_average_upward_delay(real: np.ndarray, pred: np.ndarray,
                                 max_delay: int = 10, prominence: float = 10) -> float:
    """
    Compute the delay for each upward segment and return the average delay.
    """
    upward_segments, _ = find_all_threshold_segments(real, prominence=prominence)
    delays = []
    for (t_nadir, t_peak, t_p75) in upward_segments:
        delay = compute_upward_delay_segment(real, pred, t_nadir, t_p75, max_delay)
        delays.append(delay)
    return np.mean(delays) if delays else 0.0


def compute_average_downward_delay(real: np.ndarray, pred: np.ndarray,
                                   max_delay: int = 10, prominence: float = 10) -> float:
    """
    Compute the delay for each downward segment and return the average delay.
    """
    _, downward_segments = find_all_threshold_segments(real, prominence=prominence)
    delays = []
    for (t_peak, t_nadir, t_n75) in downward_segments:
        delay = compute_downward_delay_segment(real, pred, t_peak, t_n75, max_delay)
        delays.append(delay)
    return np.mean(delays) if delays else 0.0

###############################################################################
# 4) Main Plot Function: Compare Predictions and Ground Truth & Compute Metrics
###############################################################################
def plot_predictions(experiment_path: str,
                     dataset: CSVDataset,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     output_features: list,
                     t_initial: int, t_final: int,
                     save_folder_name: str = "predictions",
                     data_split: str = "test") -> None:
    """
    Plot the comparison between predictions and ground truth within the specified time interval.
    Also compute RMSE, running FIT index, and the average upward and downward delays
    (based on 75% thresholds) for each feature, then log the results.
    """
    logger.info("Checking input data:")
    logger.info(f"predictions shape: {predictions.shape}")
    logger.info(f"targets shape: {targets.shape}")
    logger.info(f"output_features: {output_features}")
    logger.info(f"t_initial: {t_initial}, t_final: {t_final}")

    # Create the directory to save results if it does not exist
    save_dir = Path(experiment_path) / save_folder_name / data_split
    save_dir.mkdir(parents=True, exist_ok=True)

    # Assume that predictions have multiple time steps; use the last time step as final prediction
    prediction_sample = predictions.shape[1] - 1

    # Extract final predictions and corresponding targets
    profiles_hat = predictions[:, prediction_sample, :]
    profiles_targets = targets[:, prediction_sample, :]

    # Denormalise if the dataset provides scaling parameters
    scaling_factors_min = dataset.scaling_factors_min[dataset.indices_output_features]
    scaling_factors_max = dataset.scaling_factors_max[dataset.indices_output_features]
    profiles_hat = denormalise(profiles_hat, scaling_factors_min, scaling_factors_max)
    profiles_targets = denormalise(profiles_targets, scaling_factors_min, scaling_factors_max)

    # Single-feature case
    if len(output_features) == 1:
        feature_name = output_features[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(feature_name)
        ax.grid()

        # Convert to numpy arrays
        real_data = profiles_targets[t_initial:t_final, 0].detach().cpu().numpy()
        pred_data = profiles_hat[t_initial:t_final, 0].detach().cpu().numpy()

        ax.plot(real_data, label="Ground Truth", color="blue", linewidth=1.0)
        ax.plot(pred_data, label="Prediction", color="green", linewidth=1.0)
        ax.legend()

        plt.savefig(save_dir / f"predictions_{feature_name}.png", bbox_inches='tight')
        plt.close(fig)

        # --- Compute Metrics ---
        fit_vals = fit_index(pred_data, real_data)
        avg_fit = np.mean(fit_vals)
        logger.info(f"[{feature_name}] Average FIT: {avg_fit:.2f}%")
        logger.info(f"[{feature_name}] Final FIT: {fit_vals[-1]:.2f}%")

        rmse_val = calculate_rmse(pred_data, real_data)
        logger.info(f"[{feature_name}] RMSE: {rmse_val:.4f}")

        # Compute average delays using the improved multi-segment detection (prominence=20)
        up_delay_avg = compute_average_upward_delay(real_data, pred_data, max_delay=8, prominence=0.01)
        down_delay_avg = compute_average_downward_delay(real_data, pred_data, max_delay=8, prominence=0.01)
        logger.info(f"[{feature_name}] Average Upward Delay: {up_delay_avg:.2f} samples")
        logger.info(f"[{feature_name}] Average Downward Delay: {down_delay_avg:.2f} samples")

    # Multi-feature case
    else:
        fig, axs = plt.subplots(len(output_features), 1, figsize=(10, 6 * len(output_features)))
        fig.suptitle("Predictions vs. Ground Truth")
        for i, feature_name in enumerate(output_features):
            axs[i].set_title(feature_name)
            axs[i].grid()
            real_data = profiles_targets[t_initial:t_final, i].detach().cpu().numpy()
            pred_data = profiles_hat[t_initial:t_final, i].detach().cpu().numpy()
            axs[i].plot(real_data, label="Ground Truth", color="blue", linewidth=1.0)
            axs[i].plot(pred_data, label="Prediction", color="green", linewidth=1.0)
            axs[i].legend()
        plt.savefig(save_dir / "predictions_multifeature.png", bbox_inches='tight')
        plt.close(fig)

        for i, feature_name in enumerate(output_features):
            real_data = profiles_targets[t_initial:t_final, i].detach().cpu().numpy()
            pred_data = profiles_hat[t_initial:t_final, i].detach().cpu().numpy()
            fit_vals = fit_index(pred_data, real_data)
            avg_fit = np.mean(fit_vals)
            logger.info(f"[{feature_name}] Average FIT: {avg_fit:.2f}%")
            logger.info(f"[{feature_name}] Final FIT: {fit_vals[-1]:.2f}%")
            rmse_val = calculate_rmse(pred_data, real_data)
            logger.info(f"[{feature_name}] RMSE: {rmse_val:.4f}")
            up_delay_avg = compute_average_upward_delay(real_data, pred_data, max_delay=8, prominence=0.01)
            down_delay_avg = compute_average_downward_delay(real_data, pred_data, max_delay=8, prominence=0.01)
            logger.info(f"[{feature_name}] Average Upward Delay: {up_delay_avg:.2f} samples")
            logger.info(f"[{feature_name}] Average Downward Delay: {down_delay_avg:.2f} samples")
            
def plot_blood_glucose_trajectory(experiment_path: str,
                                  state: torch.Tensor,
                                  t_initial: int, t_final: int,
                                  save_folder_name: str = "trajectory") -> None:
    """This function is used to plot the blood glucose trajectory and associated inputs (Insulin, CHO).

    Args:
        experiment_path (str): Path to the experiment folder.
        state (torch.Tensor): State trajectory (e.g., BloodGlucose).
        t_initial (int): Initial time step to plot.
        t_final (int): Final time step to plot.
        save_folder_name (str): Name of the folder where to save the plots.
    """
    # Create trajectory folder
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    # Create figure
    if state.shape[1] == 1:
        fig, axs = plt.subplots(figsize=(15, 10))
        axs.set_title('Blood Glucose Level')
        axs.plot(state[t_initial:t_final, 0].cpu().numpy(), label='BloodGlucose')
        axs.grid()
        axs.legend()
    else:
        fig, axs = plt.subplots(state.shape[1], figsize=(15, 10))
        for i in range(state.shape[1]):
            axs[i].set_title('Blood Glucose Level')
            axs[i].plot(state[t_initial:t_final, i].cpu().numpy(), label='BloodGlucose')
            axs[i].grid()
            axs[i].legend()

    # Save figure
    plt.savefig(os.path.join(save_directory, "blood_glucose_trajectory.png"), bbox_inches='tight')
    plt.close()

def plot_different_runs_and_metrics(results_df: pd.DataFrame, experiment_path: str) -> None:
    """Plot each different run separately on the same graph with respect to the iteration and all the logged metrics."""
    # Get unique runs
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRICS_TENDENCY dictionary's keys
    unique_metrics = [column for column in results_df.columns if column in METRICS_TENDENCY]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_runs)))

    for metric in unique_metrics:
        last_value = None
        best_id = None
        ymin = float("inf")
        ymax = float("-inf")

        for i, run in enumerate(unique_runs):
            run_df = results_df[results_df["trial_id"] == run]
            x = np.arange(0, len(run_df[metric].values))
            plt.plot(x, run_df[metric].values, color=colors[i])
            ymin = min(ymin, np.min(run_df[metric].values))
            ymax = max(ymax, np.max(run_df[metric].values))

            if last_value is None:
                last_value = run_df[metric].values[-1]
                best_id = run
            else:
                if METRICS_TENDENCY[metric] == "min":
                    if run_df[metric].values[-1] < last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run
                else:
                    if run_df[metric].values[-1] > last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run

        # Plot the best run with respect to black color
        best_run = results_df[results_df["trial_id"] == best_id]
        x = np.arange(0, len(best_run[metric].values))
        plt.plot(x, best_run[metric].values, color="black")
        plt.xlabel("Tuning step")
        plt.ylabel(metric)
        plt.grid()
        plt.ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)
        plt.title(f"Individual runs, All runs: {len(unique_runs)}, Best run id: {best_id}", fontsize=10)
        plt.savefig(os.path.join(experiment_path, f"{metric}_individual_runs.png"), bbox_inches="tight")
        plt.close()
        plt.clf()

def plot_training_loss(experiment_path: str, 
                       training_losses: list,
                       save_folder_name: str = "training_loss") -> None:
    """
    Function to plot training loss over time steps.

    Args:
        experiment_path (str): Path to save the plot.
        training_losses (list of float): List of training loss values over time steps.
        save_folder_name (str, optional): Folder name to save the plot. Defaults to "training_loss".
    """
    # Create the directory to save the plot
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time Steps')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(save_directory, 'training_loss_plot.png'), bbox_inches='tight')
    plt.close()
