import argparse
from src.utils.configs import TrainingConfig
from src.infer import Inference
from src.utils.configs import TrainingConfig, GeneratorType, DiffusionModelType, CRNNType
from src.utils.arguments import get_configuration

import os
import cv2
import numpy as np


def verify_visualization(output_dir):
    # Load first image
    img_path = os.path.join(output_dir, 'local_map_trajectory_0.png')
    img = cv2.imread(img_path)
    
    yellow_mask = np.all(img == [0, 255, 255], axis=-1)  # BGR format
    cyan_mask = np.all(img == [255, 255, 0], axis=-1)    # BGR format
    
    has_yellow = np.any(yellow_mask)
    has_cyan = np.any(cyan_mask)
    
    print("Visualization check:")
    print(f"Found ground truth (yellow): {has_yellow}")
    print(f"Found prediction (cyan): {has_cyan}")
    
    return has_yellow and has_cyan

def main():

    cfgs = get_configuration()

    cfgs.loss.output_dir = "/tracto/TractoDiff/inference_results"
    cfgs.output_dir = "/tracto/TractoDiff/inference_results"
    
    inferencer = Inference(cfgs)
    
    # Run inference
    try:
        inferencer.run()
        print("Inference completed successfully!")
        # verify_visualization(args.output_dir)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
    finally:
        inferencer.cleanup()

if __name__ == '__main__':
    main()