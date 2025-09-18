# src/utils/logger.py
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import logging
import torch
import pandas as pd 

def setup_logger(log_file: str, logger_name: str = None):
    """
    Sets up a logger that logs to both a file and the console.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Avoid duplicate handlers if the logger is already configured
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger

class TrainingLogger:
    def __init__(self, output_dir, experiment_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or timestamp
        
        self.iteration_log_file = self.output_dir / f"iteration_log_{self.experiment_name}.csv"
        self.epoch_log_file = self.output_dir / f"epoch_log_{self.experiment_name}.csv"
        self.iteration_writer = None
        self.epoch_writer = None
        
        self.current_epoch_losses = []
        
        self.loss_plot_dir = self.output_dir / "loss_plots"
        os.makedirs(self.loss_plot_dir, exist_ok=True)

        event_log_file = self.output_dir / f"events_{self.experiment_name}.log"
        self.event_logger = setup_logger(str(event_log_file), logger_name=self.experiment_name)
        self.event_logger.info("TrainingLogger initialized. Event logs will be saved to %s", event_log_file)

    
    def log_iteration(self, iteration: int, metrics: dict, epoch: int):
        """Log all scalar metrics for a single training iteration."""
        # Prepare the data row for the CSV
        row_data = {'iteration': iteration, 'epoch': epoch}
        
        # Extract scalar values from the metrics dictionary
        for key, value in metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                row_data[key] = value.item()
            elif isinstance(value, (int, float)):
                row_data[key] = value
        
        if self.iteration_writer is None:
            fieldnames = list(row_data.keys())
            self.csv_file = open(self.iteration_log_file, 'w', newline='')
            self.iteration_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.iteration_writer.writeheader()
        
        self.iteration_writer.writerow(row_data)
        
        if 'original_loss' in row_data:
            self.current_epoch_losses.append(row_data['original_loss'])
    
    def log_epoch(self, epoch: int, metrics: dict = None):
        """Log average loss for a completed epoch."""
        if self.current_epoch_losses:
            avg_loss = np.mean(self.current_epoch_losses)
            row_data = {'epoch': epoch, 'average_loss': avg_loss}
            self.current_epoch_losses = [] 
            
            if metrics:
                row_data.update(metrics)
            
            if self.epoch_writer is None:
                fieldnames = list(row_data.keys())
                self.epoch_csv_file = open(self.epoch_log_file, 'w', newline='')
                self.epoch_writer = csv.DictWriter(self.epoch_csv_file, fieldnames=fieldnames)
                self.epoch_writer.writeheader()
            
            self.epoch_writer.writerow(row_data)
    
    def plot_losses(self, save_path=None):
        """Plots the training progress by reading the log files."""
        if not self.iteration_log_file.exists():
            self.event_logger.warning("Log file not found, skipping plot.")
            return

        # Use pandas to easily read and plot the data
        df = pd.read_csv(self.iteration_log_file)
        
        plt.figure(figsize=(15, 7))
        
        # Smooth the loss curve to make it easier to see the trend
        # A window of 50 iterations is a good starting point
        window_size = min(50, len(df) // 10) 
        if window_size > 0:
            df['loss_smoothed'] = df['original_loss'].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['iteration'], df['loss_smoothed'], label='Smoothed Training Loss')
        else:
            plt.plot(df['iteration'], df['original_loss'], label='Training Loss')

        plt.title(f'Training Loss Over Time for {self.experiment_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.event_logger.warning(f"Loss plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def __del__(self):
        # Ensure files are closed when the object is destroyed
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        if hasattr(self, 'epoch_csv_file') and not self.epoch_csv_file.closed:
            self.epoch_csv_file.close()
            
    def plot_epoch_metrics(self, save_path=None):
        """Plots all collected metrics against epochs."""
        if not self.epoch_log_file.exists():
            self.event_logger.warning("Epoch log file not found, skipping epoch metrics plot.")
            return

        df = pd.read_csv(self.epoch_log_file)
        
        plt.figure(figsize=(12, 6))
        
        for col in df.columns:
            if col != 'epoch':
                plt.plot(df['epoch'], df[col], marker='o', linestyle='-', label=col.replace('_', ' ').title())

        plt.title(f'Epoch Metrics for {self.experiment_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = save_path or self.plots_dir / f'epoch_metrics_curve_{self.experiment_name}.png'
        plt.savefig(save_path)
        self.event_logger.info(f"Epoch metrics plot saved to {save_path}")
        plt.close()