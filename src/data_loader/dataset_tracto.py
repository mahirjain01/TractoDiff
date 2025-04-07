import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

class TractographyDataset(Dataset):
    def __init__(self, 
                 bundle='AF_L', 
                 subjects=['sub-1030'], 
                 split='trainset',
                 seq_length=16,
                 root_path='/tracto/Gagan/data/TractoInferno/derivatives',
                 output_path='/tracto/TractoDiff/output',
                 shuffle=True):
        """
        PyTorch Dataset for tractography data
        
        Args:
            bundle (str): Bundle name (e.g., 'AF_L')
            subjects (list): List of subject IDs
            split (str): Data split ('trainset', 'testset', etc.)
            seq_length (int): Length of point subsequence to extract
            root_path (str): Path to original tractography data
            output_path (str): Path to processed .pkl files
            shuffle (bool): Whether to shuffle data
        """
        self.bundle = bundle
        self.subjects = subjects
        self.split = split
        self.seq_length = seq_length
        self.root_path = root_path
        self.output_path = output_path
        self.shuffle = shuffle
        
        self.streamlines = []
        self.condition_vectors = []
        self.streamline_indices = []
        
        self._load_data()
    
    def _load_data(self):
        """Load both 3D streamlines and condition vectors"""
        for subject in self.subjects:
            # Load 3D point data from .trk file
            tract_fname = f'{self.root_path}/{self.split}/{subject}/tractography/{subject}__{self.bundle}.trk'
            tractogram = nib.streamlines.load(tract_fname)
            streamlines = tractogram.streamlines
            
            # Load condition vectors from .pkl file
            pkl_path = f'{self.output_path}/{self.bundle}/{subject}.pkl'
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    condition_data = pickle.load(f)
            else:
                print(f"Warning: {pkl_path} not found. Skipping subject {subject}")
                continue
            
            # Add to our dataset
            start_idx = len(self.streamlines)
            for i, streamline in enumerate(streamlines):
                # Only add streamlines that are long enough
                if len(streamline) >= self.seq_length:
                    self.streamlines.append(streamline)
                    self.condition_vectors.append(condition_data[i]['observations'])
                    self.streamline_indices.append((start_idx + i, len(streamline)))
            
            print(f"Loaded {len(streamlines)} streamlines from subject {subject}")
        
        print(f"Total usable streamlines: {len(self.streamlines)}")
        
        if self.shuffle:
            combined = list(zip(self.streamlines, self.condition_vectors, self.streamline_indices))
            random.shuffle(combined)
            self.streamlines, self.condition_vectors, self.streamline_indices = zip(*combined)
    
    def __len__(self):
        return len(self.streamlines)
    
    def __getitem__(self, idx):
        """
        Get a random subsequence of 16 points from a streamline and its corresponding first condition vector
        
        Returns:
            points (tensor): Tensor of shape (16, 3) containing 16 consecutive 3D points
            condition (tensor): Tensor of shape (334,) containing the condition vector for the first point
        """
        streamline = self.streamlines[idx]
        condition_vectors = self.condition_vectors[idx]
        
        # Generate a random start index for the subsequence
        max_start_idx = len(streamline) - self.seq_length
        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        
        # Extract the subsequence of points
        point_seq = streamline[start_idx:start_idx + self.seq_length]
        
        # Get the corresponding condition vector for the first point in the subsequence
        first_point_condition = condition_vectors[start_idx]
        
        # Convert to PyTorch tensors
        points_tensor = torch.tensor(point_seq, dtype=torch.float32)
        condition_tensor = torch.tensor(first_point_condition, dtype=torch.float32)
        
        return {
            'points': points_tensor,  # Shape: (16, 3)
            'condition': condition_tensor  # Shape: (334,)
        }


def get_dataloader(bundle='AF_L', 
                  subjects=['sub-1030'], 
                  split='trainset',
                  batch_size=32,
                  seq_length=16,
                  shuffle=True,
                  num_workers=4):
    """
    Create a PyTorch DataLoader for tractography data
    
    Args:
        bundle (str): Bundle name (e.g., 'AF_L')
        subjects (list): List of subject IDs
        split (str): Data split ('trainset', 'testset', etc.)
        batch_size (int): Batch size
        seq_length (int): Length of point subsequence to extract
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = TractographyDataset(
        bundle=bundle,
        subjects=subjects,
        split=split,
        seq_length=seq_length,
        shuffle=shuffle
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    dataset = TractographyDataset(bundle='AF_L', subjects=['sub-1030'])
    print(f"Dataset size: {len(dataset)}")
    
    # Test extracting a single sample
    sample = dataset[0]
    print(f"Sample points : {sample['points']}")
    print(f"Sample condition shape: {sample['condition'].shape}")
    
    # Test with DataLoader
    dataloader = get_dataloader(bundle='AF_L', subjects=['sub-1030'], batch_size=4)
    for batch in dataloader:
        print(f"Batch points shape: {batch['points'].shape}")
        print(f"Batch condition shape: {batch['condition'].shape}")
        break