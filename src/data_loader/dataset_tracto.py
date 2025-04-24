import os
import copy
import pickle
import random
import numpy as np
import nibabel as nib
import cv2
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.data_loader.dataset import TractographyDataset


    
def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def registration_collate_fn_stack_mode(data_dicts):
    """Collate function for registration in stack mode.
    Args:
        data_dicts (List[Dict])
    Returns:
        collated_dict (Dict)
    """
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if key not in collated_dict:
                collated_dict[key] = []
                
            # Handle different data types appropriately
            if isinstance(value, (str, list)):
                collated_dict[key].append(value)
            else:
                # Convert numerical data to tensor
                try:
                    value = torch.from_numpy(np.asarray(value)).to(torch.float)
                    collated_dict[key].append(value)
                except TypeError as e:
                    print(f"Error converting key {key} with value type {type(value)}")
                    raise e

    # Stack tensors, leave other types as lists
    for key, value in collated_dict.items():
        if isinstance(value[0], torch.Tensor):
            collated_dict[key] = torch.stack(value, dim=0)
        # else leave as list for non-tensor data

    return collated_dict

def get_dataloader(cfg, train=True):
    """
    Create a PyTorch DataLoader for tractography data
    
    Args:
        bundle (str): Bundle name (e.g., 'AF_L')
        subjects (list): List of subject IDs
        split (str): Data split ('trainset' etc.)
        batch_size (int): Batch size
        seq_length (int): Length of point subsequence to extract
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = TractographyDataset(cfg=cfg, train=train)
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=sampler,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader

def train_data_loader(cfg):
    """
    This function is to create a training dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_dataloader(cfg=cfgs, train=True)


def evaluation_data_loader(cfg):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_dataloader(cfg=cfgs, train=False)




# if __name__ == "__main__":
#     dataset = TractographyDataset(bundle='AF_L', subjects=['sub-1030'])
#     print(f"Dataset size: {len(dataset)}")
    
#     # Test extracting a single sample
#     sample = dataset[0]
#     print(f"Sample points : {sample['points']}")
#     print(f"Sample condition shape: {sample['condition'].shape}")
    
#     # Test with DataLoader
#     dataloader = get_dataloader(bundle='AF_L', subjects=['sub-1030'], batch_size=4)
#     for batch in dataloader:
#         print(f"Batch points shape: {batch['points'].shape}")
#         print(f"Batch condition shape: {batch['condition'].shape}")
#         break