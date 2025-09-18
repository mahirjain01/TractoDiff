import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from src.models.backbones.rnn import RNNDiffusion
from src.models.backbones.unet import ConditionalUnet1D
from src.utils.configs import DataDict, DiffusionModelType


class Diffusion(nn.Module):
    def __init__(self, cfg, activation_func=nn.Softsign):
        super(Diffusion, self).__init__()
        self.model_type = cfg.model_type
        # self.diffusion_type = cfg.diffusion_type
        self.noise_scheduler = DDPMScheduler(beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                                             prediction_type="sample", num_train_timesteps=cfg.num_train_timesteps,
                                             clip_sample_range=cfg.clip_sample_range, clip_sample=cfg.clip_sample,
                                             beta_schedule=cfg.beta_schedule)
       
        # Initialize scheduler timesteps to None, will be set properly in sample()
        self.noise_scheduler.timesteps = None
        self.time_steps = cfg.num_train_timesteps
        self.use_traversability = cfg.use_traversability
        self.estimate_traversability = cfg.estimate_traversability
        self.traversable_steps = cfg.traversable_steps

        # /////////////////////////////////////////////////// CHANGED FOR TRACTO ///////////////////////////////////////////////////
        self.zd = 512 
        self.waypoint_dim = 3
        self.diffusion_step_embed_dim = 256
        cfg.perception_in = 346
        # /////////////////////////////////////////////////// CHANGED FOR TRACTO ///////////////////////////////////////////////////
        
        self.waypoints_num = cfg.waypoints_num
       
        if activation_func is None:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), nn.LeakyReLU(0.1),
                                         nn.Linear(1024, 2048), nn.LeakyReLU(0.2),
                                         nn.Linear(2048, 512), nn.LeakyReLU(0.2),
                                         nn.Linear(512, self.zd), nn.LeakyReLU(0.2))
        else:
            self.encoder = nn.Sequential(nn.Linear(cfg.perception_in, 1024), activation_func(),
                                         nn.Linear(1024, 2048), activation_func(),
                                         nn.Linear(2048, 512), activation_func(),
                                         nn.Linear(512, self.zd), activation_func())
            
        self.trajectory_condition = nn.Linear(self.zd, self.zd)

        if self.model_type == DiffusionModelType.crnn:
            rnn_threshold = cfg.rnn_output_threshold
            self.diff_model = RNNDiffusion(in_dim=self.waypoint_dim * self.waypoints_num, out_dim=self.waypoint_dim,
                                           hidden_dim=self.zd, diffusion_step_embed_dim=self.diffusion_step_embed_dim,
                                           steps=self.waypoints_num, rnn_type=cfg.rnn_type,
                                           output_threshold=rnn_threshold, activation_func=nn.Softsign)
        elif self.model_type == DiffusionModelType.unet:
            self.diff_model = ConditionalUnet1D(input_dim=self.waypoint_dim, global_cond_dim=self.zd,
                                                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                                                down_dims=cfg.down_dims, kernel_size=cfg.kernel_size,
                                                cond_predict_scale=cfg.cond_predict_scale, n_groups=cfg.n_groups)
        else:
            raise Exception("the diffusion model type is not defined")

    def _ensure_scheduler_on_device(self, device):
        """Ensure all scheduler tensors are on the same device."""
        for attr_name in dir(self.noise_scheduler):
            # Skip internal attributes and methods
            if attr_name.startswith('_') or callable(getattr(self.noise_scheduler, attr_name)):
                continue
            
            attr = getattr(self.noise_scheduler, attr_name, None)
            # Check if it's a tensor and move to device if needed
            if isinstance(attr, torch.Tensor):
                setattr(self.noise_scheduler, attr_name, attr.to(device))
            
        # Explicitly handle common important tensor attributes
        if hasattr(self.noise_scheduler, 'timesteps') and self.noise_scheduler.timesteps is not None:
            self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(device)
        if hasattr(self.noise_scheduler, 'alphas_cumprod'):
            self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        if hasattr(self.noise_scheduler, 'final_alpha_cumprod'):
            self.noise_scheduler.final_alpha_cumprod = self.noise_scheduler.final_alpha_cumprod.to(device)
        if hasattr(self.noise_scheduler, 'betas'):
            self.noise_scheduler.betas = self.noise_scheduler.betas.to(device)
        if hasattr(self.noise_scheduler, 'alphas'):
            self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(device)

    def add_trajectory_noise(self, trajectory):
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return noise

    def add_time_step_noise(self, trajectory, traversable_steps=None):
        if traversable_steps is None:
            # Will always go here = 1000
            time_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            time_steps = traversable_steps
        time_step = torch.randint(0, time_steps, (trajectory.shape[0],), device=trajectory.device).long()
        return time_step # shape = [B x 1] in range of [0, 1000)

    def add_trajectory_step_noise(self, trajectory, traversable_step=None):

        # print("use_traversability inside add_trajectory_step_noise:", self.use_traversability)

        device = trajectory.device
        # Ensure scheduler is on the right device
        self._ensure_scheduler_on_device(device)
        
        noise = self.add_trajectory_noise(trajectory=trajectory) # [b x 16 x 3]
        time_step = self.add_time_step_noise(trajectory=trajectory) #  [B x 1]
        noisy_trajectory = self.noise_scheduler.add_noise(original_samples=trajectory, noise=noise, timesteps=time_step)
        
        if self.use_traversability:
            ########################### Doubles everything : Since first half for loss, second half for traversibility : check loss3d.py forward pass #######################################
            t_trajectories = trajectory.clone()
            t_noise = self.add_trajectory_noise(trajectory=t_trajectories)
            if traversable_step is None:
                traversable_step = self.traversable_steps
            t_time_step = self.add_time_step_noise(trajectory=t_trajectories, traversable_steps=traversable_step)
            t_noisy_trajectory = self.noise_scheduler.add_noise(original_samples=t_trajectories, noise=t_noise,
                                                                timesteps=t_time_step)
            noise = torch.concat((noise, t_noise))
            time_step = torch.concat((time_step, t_time_step))
            noisy_trajectory = torch.concat((noisy_trajectory, t_noisy_trajectory), dim=0)
        
        return noisy_trajectory, noise, time_step # Basicually the batch size doubles for computing trav loss

    def forward(self, observation, gt_path=None, traversable_step=None):
        h = self.encoder(observation)  # B x 512
        h_condition = self.trajectory_condition(h) # B x 512        

        # print("The h_condition shape is: ", h_condition.shape)
        # print("The h shape is: ", h.shape)

        # print("The h_condition shape is: ", h_condition.shape)
        output = {}

        noisy_trajectory, noise, time_step = self.add_trajectory_step_noise(trajectory=gt_path, traversable_step=traversable_step)

        if self.use_traversability:
            print("Turning on trab")
            h_condition = torch.concat((h_condition, h_condition), dim=0)   # new shape = [2*B x 512]
        pred = self.diff_model(noisy_trajectory, time_step, local_cond=None, global_cond=h_condition)
        # print("The pred shape is: ", pred.shape)
        output.update({
            DataDict.prediction: pred,
            DataDict.noise: noise,
            DataDict.time_steps: time_step
        })
        return output

    @torch.no_grad()
    def sample(self, observation):
        h = self.encoder(observation)  # B x 512
        h_condition = self.trajectory_condition(h)

        B, C = h_condition.shape
        trajectory = torch.randn(size=(h_condition.shape[0], self.waypoints_num, self.waypoint_dim),
                                 dtype=h_condition.dtype, device=h_condition.device, generator=None)
        all_trajectories = []
        scheduler = self.noise_scheduler
        
        # Ensure scheduler tensors are on the correct device
        self._ensure_scheduler_on_device(h_condition.device)
        
        scheduler.set_timesteps(self.time_steps)
        # Make sure timesteps are on the same device as the model
        scheduler.timesteps = scheduler.timesteps.to(h_condition.device)
        
        for t in scheduler.timesteps:
            if (t < self.time_steps):
                break
            t = t.to(h_condition.device)
            model_output = self.diff_model(trajectory, t.unsqueeze(0).repeat(B, ), local_cond=None,
                                           global_cond=h_condition)
            trajectory = scheduler.step(model_output, t, trajectory, generator=None).prev_sample.contiguous()
            
        output = {
            DataDict.prediction: trajectory,
            DataDict.all_trajectories: all_trajectories,
        }

        return output 