import os
import pickle
import torch
import random
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class TractographyDataset(Dataset):
    def __init__(self, cfg, train: bool, logger = None):
        """
        PyTorch Dataset for tractography data
        
        Args:
            bundle (str): Bundle name (e.g., 'AF_L')
            subjects (list): List of subject IDs
            split (str): Data split ('trainset', etc.)
            seq_length (int): Length of point subsequence to extract
            root_path (str): Path to original tractography data
            condition_path (str): Path to processed .pkl files
            shuffle (bool): Whether to shuffle data
        """
        self.bundle = cfg.bundle
        self.logger = logger

        if train: 
            self.split = 'trainset'
            self.subjects = cfg.subjects[:1]
            self.logger.info(f"The subjects are {self.subjects}")
        else:
            self.split = 'trainset'
            self.subjects = cfg.subjects[-1:]

        self.seq_length = cfg.seq_length
        self.root_path = cfg.root_path
        self.condition_path = cfg.condition_path
        
        self.streamlines = []
        self.condition_vectors = []
        self.streamline_indices = []

        self.streamline_subjects = []
        
        self.mean = None
        self.std = None
        
        self._load_data()
        
        if train:
            self._compute_stats()
    
    def _load_data(self):
        """Load both 3D streamlines and condition vectors"""
        for subject in self.subjects:

            tract_fname = f'{self.root_path}/{self.split}/{subject}/tractography/{subject}__{self.bundle}.trk'
            tractogram = nib.streamlines.load(tract_fname)
            streamlines = tractogram.streamlines
            
            # Load condition vectors from .pkl file
            pkl_path = f'{self.condition_path}/{self.bundle}/{subject}.pkl'
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    condition_data = pickle.load(f)
            else:
                print(f"Warning: {pkl_path} not found. Skipping subject {subject}")
                continue
            
            for i, streamline in enumerate(streamlines[:2000]):
                if len(streamline) < self.seq_length:
                    continue
                self.streamlines.append(streamline)
                self.condition_vectors.append(condition_data[i]['observations'])
                self.streamline_subjects.append(subject)
            
            self.logger.info(f"Loaded {len(streamlines)} streamlines from subject {subject} for {self.split}")
        
        self.logger.info(f"Total usable streamlines for {self.split}: {len(self.streamlines)}")
        
    def __len__(self):
        return len(self.streamlines)
    
    def _compute_stats(self):
        """Computes mean and standard deviation for normalization."""
        if not self.streamlines:
            self.logger("Cannot compute stats, no streamlines loaded.")
            self.mean = torch.zeros(3, dtype=torch.float32)
            self.std = torch.ones(3, dtype=torch.float32)
            return

        self.logger.info("Computing normalization statistics...")
        all_points = np.vstack(self.streamlines)
        
        self.mean = torch.tensor(np.mean(all_points, axis=0), dtype=torch.float32)
        self.std = torch.tensor(np.std(all_points, axis=0), dtype=torch.float32)
        
        self.logger.info(f"Computed Mean: {self.mean.tolist()}")
        self.logger.info(f"Computed Std Dev: {self.std.tolist()}")
        
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std
        self.logger.info("Normalization stats set from training data.")
    
    def __getitem__(self, idx):
        """
        Get a random subsequence of 16 points from a streamline and its corresponding first condition vector
        
        Returns:
            points (tensor): Tensor of shape (16, 3) containing 16 consecutive 3D points
            condition (tensor): Tensor of shape (334,) containing the condition vector for the first point
        """

        subject_id = self._get_subject_id_for_index(idx)

        streamline = self.streamlines[idx]
        condition_vectors = self.condition_vectors[idx]
        
        # Generate a random start index for the subsequence
        max_start_idx = len(streamline) - self.seq_length
        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0

        # self.logger.info("The start value is: ", streamline[start_idx])
        # self.logger.info("The streamline is: ", streamline[start_idx:start_idx + self.seq_length])

        # Extract the subsequence of points
        point_seq = streamline[start_idx:start_idx + self.seq_length]
        
        # Get the corresponding condition vector for the first point in the subsequence
        first_point_condition = condition_vectors[start_idx]
        
        # Convert to PyTorch tensors
        points_tensor = torch.tensor(point_seq, dtype=torch.float32)
        condition_tensor = torch.tensor(first_point_condition, dtype=torch.float32)
        
        if self.mean is not None and self.std is not None:
            normalized_points = (points_tensor - self.mean) / self.std
        else:
            # Should not happen for training, but a safeguard
            normalized_points = points_tensor
        
        return {
            'points': normalized_points,  # Shape: (16, 3)
            'condition': condition_tensor,  # Shape: (334,)
            'subject_id': subject_id,
            'bundle': self.bundle
        }

    def _get_subject_id_for_index(self, idx):
        return self.streamline_subjects[idx]
