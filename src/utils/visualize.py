import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.calibration import calibration_curve
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def plot_calibration_curve(y_true, y_prob, output_path='reports/figures/calibration_plot.png', n_bins=10):
    """
    Plots the calibration curve and the histogram of predicted probabilities.
    
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities for the positive class.
        output_path (str): Path to save the plot image.
        n_bins (int): Number of bins for calibration curve and histogram.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot perfectly calibrated line
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        # Plot model calibration curve
        ax1.plot(prob_pred, prob_true, "s-", label="Model")
        
        ax1.set_ylabel("Fraction of positives")
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title("Calibration Plot (Reliability Curve)")
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot histogram of predicted probabilities as a secondary axis
        ax2 = ax1.twinx()
        ax2.hist(y_prob, range=(0, 1), bins=n_bins, alpha=0.3, color='gray', label='Prediction count')
        ax2.set_ylabel("Count")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Calibration plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot calibration curve: {e}")
