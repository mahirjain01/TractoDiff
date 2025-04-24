import torch
from torch import nn

from src.models.perception import Perception, LidarImageModel
from src.models.diffusion import Diffusion

from src.utils.configs import DataDict, GeneratorType


class HNav(nn.Module):
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device

        self.generator_type = config.generator_type
        self.generator = Diffusion(self.config.diffusion)
        self.generator = self.generator.to(device)

    def forward(self, input_dict, sample=False):
        # Ensure we're on the correct device
        curr_device = next(self.parameters()).device
        if str(curr_device) != str(self.device):
            self.to(self.device)

        # output = {DataDict.path: input_dict[DataDict.path],
        #               DataDict.heuristic: input_dict[DataDict.heuristic],
        #               DataDict.local_map: input_dict[DataDict.local_map]}

        # # /////// shape for condition vector /////////////
        # observation = self.perception(lidar=input_dict[DataDict.lidar], vel=input_dict[DataDict.vel],
        #                                 target=input_dict[DataDict.target])
        # print("The observation shape is: ", observation.shape)
        
        if sample:
            return self.sample(input_dict=input_dict)
        else:
            output = {DataDict.points: input_dict[DataDict.points],
                      DataDict.subject_id: input_dict[DataDict.subject_id],
                      DataDict.bundle: input_dict[DataDict.bundle]}

            observation = input_dict[DataDict.condition]
            # print("The observation shape is: ", observation.shape)
            generator_output = self.generator(observation=observation, gt_path=input_dict[DataDict.points],
                                              traversable_step=input_dict[DataDict.traversable_step])
            output.update(generator_output)
            return output

    def sample(self, input_dict):
        # Ensure we're on the correct device

        curr_device = next(self.parameters()).device
        if str(curr_device) != str(self.device):
            self.to(self.device)
            
        output = {}
        if DataDict.points in input_dict.keys():
            output.update({DataDict.points: input_dict[DataDict.points]})
        if DataDict.subject_id in input_dict.keys():
            output.update({DataDict.subject_id: input_dict[DataDict.subject_id]})
        if DataDict.bundle in input_dict.keys():
            output.update({DataDict.bundle: input_dict[DataDict.bundle]})
        
        observation = input_dict[DataDict.condition]
        generator_output = self.generator.sample(observation=observation)
        output.update(generator_output)
        return output


def get_model(config, device):
    model = HNav(config=config, device=device)
    return model.to(device)
