# src/utils/logger.py
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

class TrainingLogger:
    def __init__(self, output_dir, experiment_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp and experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or timestamp
        
        # Initialize storage
        self.iteration_losses = []
        self.epoch_losses = []
        self.current_epoch_losses = []
        
        # Setup log files
        self.iteration_log_file = self.output_dir / f"iteration_losses_{self.experiment_name}.csv"
        self.epoch_log_file = self.output_dir / f"epoch_losses_{self.experiment_name}.csv"
        
        # Initialize CSV files with headers
        with open(self.iteration_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss', 'epoch'])
            
        with open(self.epoch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'average_loss'])
        
        self.loss_plot_dir = f"{output_dir}/loss_plots"
        os.makedirs(self.loss_plot_dir, exist_ok=True)
    
    def log_iteration(self, iteration, loss, epoch):
        """Log loss for a single training iteration"""
        self.iteration_losses.append({
            'iteration': iteration,
            'loss': loss,
            'epoch': epoch
        })
        self.current_epoch_losses.append(loss)
        
        # Write to CSV
        with open(self.iteration_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, loss, epoch])
    
    def log_epoch(self, epoch):
        """Log average loss for an epoch"""
        if self.current_epoch_losses:
            avg_loss = np.mean(self.current_epoch_losses)
            self.epoch_losses.append(avg_loss)
            
            # Write to CSV
            with open(self.epoch_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss])
            
            # Reset current epoch losses
            self.current_epoch_losses = []
            
            return avg_loss
    
    def plot_losses(self, epoch=None, losses=None, save_path=None):
        """Plot losses for a specific epoch or all epochs"""
        plt.figure(figsize=(12, 6))
        
        if losses:  # Plot single epoch losses
            iterations = range(len(losses))
            path_distances = [l['path_dis'] for l in losses]
            plt.plot(iterations, path_distances, label='Path Distance')
            
            if losses[0]['total_loss'] is not None:
                total_losses = [l['total_loss'] for l in losses]
                plt.plot(iterations, total_losses, label='Total Loss')
            
            plt.title(f'Losses for Epoch {epoch}')
            plt.xlabel('Iteration')
            plt.ylabel('Loss Value')
            
        else:  # Plot all epochs
            epochs = range(len(self.epoch_losses))
            avg_losses = self.epoch_losses
            plt.plot(epochs, avg_losses, label='Average Loss per Epoch')
            plt.title('Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
        
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()