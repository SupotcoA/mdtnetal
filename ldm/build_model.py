import torch
from networks import LatentDiffusion2 as LatentDiffusion
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
    print_num_params(model.ae, "AE", train_config['log_path'])
    print_num_params(model.unet, "Unet", train_config['log_path'])
    if train_config['pretrained']:
        sd=torch.load(train_config['pretrained'], map_location=torch.device('cpu'))
        keys = list(sd.keys())
        for k in keys:
            for ik in ['ae.', 'sampler.']:
                if k.startswith(ik):
                    del sd[k]
        model.load_state_dict(sd, strict=False)
    if torch.cuda.is_available():
        model.cuda()
        print("running on cuda")
    else:
        print("running on cpu!")
    optim = torch.optim.Adam(list(model.unet.parameters()) + \
                             list(model.class_embed.parameters()) + \
                             list(model.condition_embed.parameters()) + \
                             list(model.time_embed.parameters()),
                             lr=train_config['base_learning_rate'])
    if train_config['use_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim,
                                                                  T_max=train_config['train_steps'],
                                                                  eta_min=train_config['min_learning_rate'])
    else:
        lr_scheduler = None
    return model.eval(), optim, lr_scheduler
