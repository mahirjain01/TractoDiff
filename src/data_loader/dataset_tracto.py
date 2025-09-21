import os
import copy
import pickle
import random
import numpy as np
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

def get_dataloader(cfg, train=True, logger = None):
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
    dataset = TractographyDataset(cfg=cfg, train=train, logger=logger)
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    
    if not train:
        dataset.set_stats(train_dataset.mean, train_dataset.std) 
    
    shuffle = False
    if(train):
        shuffle = True
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader

def get_dataloaders(cfg, logger=None):
    """
    Creates and returns the training and evaluation dataloaders.

    This function ensures that the evaluation dataset is normalized using
    statistics computed from the training dataset.

    Args:
        cfg (object): The configuration object for the dataset.
        logger (object, optional): A logger instance.

    Returns:
        tuple: A tuple containing (training_dataloader, evaluation_dataloader).
    """
    train_dataset = TractographyDataset(cfg=cfg, train=True, logger=logger)

    eval_dataset = TractographyDataset(cfg=cfg, train=False, logger=logger)
    
    eval_dataset.set_stats(train_dataset.mean, train_dataset.std)

    train_sampler = DistributedSampler(train_dataset) if cfg.distributed else None
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if cfg.distributed else None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=registration_collate_fn_stack_mode,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False, 
        sampler=eval_sampler,
        collate_fn=registration_collate_fn_stack_mode,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, eval_dataloader

def train_data_loader(cfg, logger = None):
    """
    This function is to create a training dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_dataloader(cfg=cfgs, train=True, logger=logger)


def evaluation_data_loader(cfg, logger = None):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_dataloader(cfg=cfgs, train=False, logger=logger)


# if __name__ == "__main__":
#     dataset = TractographyDataset(bundle='AF_L', subjects=['sub-1030'])
#     print(f"Dataset size: {len(dataset)}")
    
#     sample = dataset[0]
#     print(f"Sample points : {sample['points']}")
#     print(f"Sample condition shape: {sample['condition'].shape}")
    
#     # Test with DataLoader
#     dataloader = get_dataloader(bundle='AF_L', subjects=['sub-1030'], batch_size=4)
#     for batch in dataloader:
#         print(f"Batch points shape: {batch['points'].shape}")
#         print(f"Batch condition shape: {batch['condition'].shape}")
#         break