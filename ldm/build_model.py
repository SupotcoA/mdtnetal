import torch
from networks import LatentDiffusion
from utils import print_num_params


def build_model(data_config,
                unet_config,
                ae_config,
                diffusion_config,
                train_config):
    model = LatentDiffusion(data_config,
                            unet_config,
                            ae_config,
                            diffusion_config)
    print_num_params(model.ae, "AE", data_config['log_path'])
    print_num_params(model.unet, "Unet", data_config['log_path'])
    model.cuda()
    optim = torch.optim.Adam(list(model.unet.parameters()) + \
                             list(model.class_embed.parameters()) + \
                             list(model.condition_embed.parameters()) + \
                             list(model.time_embed.parameters()),
                             lr=train_config['base_learning_rate'])
    return model.eval(), optim
