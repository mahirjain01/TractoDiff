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
    def __init__(self, output_dir, experiment_name=None, tensorboard_writer=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or timestamp
        
        self.iteration_log_file = self.output_dir / f"train_iteration_log.csv"
        self.epoch_log_file = self.output_dir / f"epoch_log.csv"
        self.iteration_writer = None
        self.epoch_writer = None

        self.writer = tensorboard_writer
        
        event_log_file = self.output_dir / f"training.log"
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
        self.csv_file.flush() 
        
    def log_epoch(self, epoch: int, metrics: dict, prefix: str):
        """
        Logs a summary for a completed epoch to the console, a CSV file, and TensorBoard.
        """
        self.event_logger.info(f"--- Epoch {epoch} {prefix} Summary ---")
        for key, value in metrics.items():
            self.event_logger.info(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        self.event_logger.info("------------------------------------")

        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"Epoch/{prefix}_{key}", value, epoch)
        
        row_data = {'epoch': epoch, 'type': prefix}
        row_data.update(metrics)
        
        if self.epoch_writer is None:
            self.epoch_csv_file = open(self.epoch_log_file, 'w', newline='')
            fieldnames = ['epoch', 'type'] + list(metrics.keys())
            self.epoch_writer = csv.DictWriter(self.epoch_csv_file, fieldnames=fieldnames)
            self.epoch_writer.writeheader()
        
        self.epoch_writer.writerow(row_data)
        self.epoch_csv_file.flush()
        
    def plot_epoch_metrics(self, save_path=None):
            """Plots all collected metrics from the epoch log CSV."""
            if not self.epoch_log_file.exists():
                self.event_logger.warning("Epoch log file not found, skipping epoch metrics plot.")
                return

            df = pd.read_csv(self.epoch_log_file)
            train_df = df[df['type'] == 'Train']
            eval_df = df[df['type'] == 'Validation']

            metrics_to_plot = [col for col in df.columns if col not in ['epoch', 'type']]
            
            output_folder = os.path.join(self.output_dir, 'Plots')
            os.makedirs(output_folder, exist_ok=True)
            
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 5))
                if not train_df.empty and metric in train_df.columns:
                    plt.plot(train_df['epoch'], train_df[metric], marker='o', linestyle='-', label=f'Train {metric}')
                if not eval_df.empty and metric in eval_df.columns:
                    plt.plot(eval_df['epoch'], eval_df[metric], marker='s', linestyle='--', label=f'Eval {metric}')
                
                plt.title(f'{metric.replace("_", " ").title()} vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                plot_path = os.path.join(output_folder, f'{metric}.png')
                plt.savefig(plot_path)
                self.event_logger.info(f"Saved epoch plot for '{metric}' to {plot_path}")
                plt.close()

    def __del__(self):
        if hasattr(self, 'iteration_csv_file') and not self.iteration_csv_file.closed:
            self.iteration_csv_file.close()
        if hasattr(self, 'epoch_csv_file') and not self.epoch_csv_file.closed:
            self.epoch_csv_file.close()