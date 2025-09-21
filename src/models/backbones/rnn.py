import torch
from torch import nn

from src.models.backbones.unet import SinusoidalPosEmb
from src.utils.functions import get_device
from src.utils.configs import RNNType, CRNNType

class RNNDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, steps=1, rnn_type=RNNType.gru, output_threshold=None,
                 activation_func=nn.Softsign):
        super(RNNDecoder, self).__init__()
        self.steps = steps
        self.rnn_type = rnn_type
        self.output_threshold = output_threshold

        if activation_func is None:
            self.in_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2))
        else:
            self.in_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation_func(),
                                       nn.Linear(hidden_dim, hidden_dim), activation_func())
        if self.rnn_type == RNNType.lstm:
            self.rnn = nn.LSTMCell(input_size=in_dim, hidden_size=hidden_dim, bias=True)
        elif self.rnn_type == RNNType.gru:
            self.rnn = nn.GRUCell(input_size=in_dim, hidden_size=hidden_dim, bias=True)
        else:
            raise Exception("the rnn type is not defined")
        if activation_func is None:
            self.out_fc = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh(),
                                        nn.Linear(256, 64), nn.Tanh(),
                                        nn.Linear(64, out_dim), nn.Tanh())
        else:
            self.out_fc = nn.Sequential(nn.Linear(hidden_dim, 256), activation_func(),
                                        nn.Linear(256, 64), activation_func(),
                                        nn.Linear(64, out_dim), activation_func())

    def step_lstm(self, x, h, c):
        h_1, c_1 = self.rnn(x, (h, c))
        output = self.out_fc(c_1)
        return h_1, c_1, output

    def step_gru(self, x, h):
        h_1 = self.rnn(x, h)
        output = self.out_fc(h_1)
        return h_1, output

    def forward(self, x_pre, z):
        B, C = x_pre.size()
        x = torch.concat((x_pre, z), dim=-1)
        c = torch.zeros((B, 2), dtype=torch.float).to(get_device(x.device))
        h = self.in_fc(z)
        outputs = []
        for i in range(self.steps):
            if self.rnn_type == RNNType.lstm:
                h, c, output = self.step_lstm(x=x, c=c, h=h)
            elif self.rnn_type == RNNType.gru:
                h, output = self.step_gru(x=x, h=h)
            else:
                raise Exception("the rnn type is not defined")
            outputs.append(output)
        output_tensor = torch.stack(outputs, dim=0).to(get_device(x.device))  # N x B x 2
        if self.output_threshold is not None:
            output_tensor = torch.clip(output_tensor, min=-self.output_threshold, max=self.output_threshold)
        return torch.transpose(output_tensor, dim0=0, dim1=1)  # B x N x 2

class CRNN(nn.Module):
    def __init__(self, cfg):
        super(CRNN, self).__init__()
        self.waypoints = cfg.waypoint_num

    def forward(self, noisy_trajectory, time_step, global_cond, local_cond=None):
        pass

class RNNDiffusion(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, diffusion_step_embed_dim=256, steps=1, rnn_type=RNNType.gru,
                 output_threshold=1.0, activation_func=nn.Softsign): 
        
        # in_dim = 48 (16 * 3) ; out_dim = 3 ; hidden_dim = 512 => Basically an output by a GRU unit

        super(RNNDiffusion, self).__init__()
        self.steps = steps
        self.rnn_type = rnn_type
        self.output_threshold = output_threshold  

        self.position_embedding = nn.Linear(3, hidden_dim) # convert coords to 512

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4), 
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + hidden_dim # 128 + 256 = 384
        
        if activation_func is None:
            self.in_fc = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.LeakyReLU(0.7),
                                       nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.7))
        else:
            self.in_fc = nn.Sequential(nn.Linear(cond_dim, hidden_dim), activation_func(),
                                       nn.Linear(hidden_dim, hidden_dim), activation_func())

        self.waypoints = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.7),
                                       nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.7))

        self.hidden_dim = hidden_dim
        if self.rnn_type == RNNType.lstm:
            self.rnn = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        elif self.rnn_type == RNNType.gru:
            self.rnn = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        else:
            raise Exception("the rnn type is not defined")
        
        # self.out_fc = nn.Sequential(nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.7),
        #                             nn.Linear(256, 64), nn.LeakyReLU(0.7),
        #                             nn.Linear(64, out_dim))
        
        if self.output_threshold is None:
            self.out_fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LeakyReLU(0.2),
                                        nn.Linear(128, 32), nn.LeakyReLU(0.2),
                                        nn.Linear(32, out_dim))
        else:
            self.out_fc = nn.Sequential(nn.Linear(hidden_dim, 128), activation_func(),
                                        nn.Linear(128, 32), activation_func(),
                                        nn.Linear(32, out_dim))

    def step_lstm(self, x, h, c):
        h_1, c_1 = self.rnn(x, (h, c))
        output = self.out_fc(c_1)  
        return h_1, c_1, output

    def step_gru(self, x, h):
        h_1 = self.rnn(x, h)
        output = self.out_fc(h_1) # shape = [2*B, 3]
        return h_1, output 

    def forward(self, noisy_trajectory, time_step, global_cond, local_cond=None):
        global_feature = self.diffusion_step_encoder(time_step) # [2B x 128]
        condition = torch.concat((global_feature, global_cond), dim=-1) #[2B x 384]  
        h = self.in_fc(condition)  # [2B X 256] => Activation function matters here : think about it!!!!!

        B, N, C = noisy_trajectory.shape # [2B X 16 X 3]
        x_pre = self.waypoints(noisy_trajectory.view(B, -1)) # => self.waypoints([2B x 48]) => [2B x 256]

        # initial_position = noisy_trajectory[:, 0, :] # => First coordinate of all streamline in the batch : [2B, 3]
        # position_embedding = self.position_embedding(initial_position) # [2B, 3] => [2B, 512]
        # h = h + position_embedding  #[2B x 512] 

        c = torch.zeros((B, self.hidden_dim), dtype=torch.float).to(get_device(x_pre.device))

        outputs = []
        for i in range(self.steps): # 16 steps => 16 GRU units to generate 16 3-d coordinates
            if self.rnn_type == RNNType.lstm:
                h, c, output = self.step_lstm(x=x_pre, c=c, h=h)
            elif self.rnn_type == RNNType.gru:
                h, output = self.step_gru(x=x_pre, h=h)
            else:
                raise Exception("the rnn type is not defined")
            outputs.append(output)

        output_tensor = torch.stack(outputs, dim=0).to(get_device(x_pre.device))  # 16 x B x 3
        if self.output_threshold is not None:
            output_tensor = torch.clip(output_tensor, min=-self.output_threshold, max=self.output_threshold)

        return torch.transpose(output_tensor, dim0=0, dim1=1)  # B x 16 x 3
