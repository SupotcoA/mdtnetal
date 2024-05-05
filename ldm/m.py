# ! pip install --upgrade diffusers[torch]
#
import shutil
try:
    shutil.rmtree('/kaggle/working/mdl')
except:
    pass

# ! git clone https://github.com/SupotcoA/mdl.git

import torch
import os
from build_model import build_model
from train import train

assert __name__ == "__main__"

torch.manual_seed(42)

config = {'ver': 'ldm_afhq_0506_v01',
          'description': 'first_test',
          'outcome_root': '/kaggle/working',
          }
config['outcome_path'] = os.path.join(config['outcome_root'], config['ver'])
config['log_path'] = os.path.join(config['outcome_path'], 'log.txt')

if not os.path.exists(config['outcome_path']):
    os.makedirs(config['outcome_path'])

unet_config = {'in_channels': 4,
               'n_channels': 64,
               'channels_mult': (1, 2, 2, 4),
               'num_res_blocks': 2,
               'c_dim': 256}

ae_config = {'model_type': "stabilityai/sd-vae-ft-ema",
             'latent_size': 32,
             'latent_dim': 4,
             }

assert ae_config['latent_dim'] == unet_config['in_channels']

diffusion_config = {'max_train_steps': 1000,
                    'sample_steps': 1000,
                    }

train_config = {'train_steps': 10000,
                'log_path': config['log_path'],
                'log_every_n_steps': 1000,
                'eval_every_n_steps': 1000,
                'outcome_root': config['outcome_path'],
                'batch_size': 16,
                'base_learning_rate': 2e-4
                }

data_config = {'afhq_root': '/kaggle/input/afhq-512',
               'image_size': 256,
               'batch_size': train_config['batch_size'],
               'x_path': ...,
               'cls_path': ...,
               'split': 0.9,
               'n_classes': 3,
               }

with open(config['log_path'], 'w') as f:
    for cf in [config, unet_config, ae_config,
               diffusion_config, data_config,
               train_config]:
        f.write(str(cf) + '\n')

model, optim = build_model(data_config,
                           unet_config,
                           ae_config,
                           diffusion_config,
                           train_config)

if not os.path.exists(data_config['x_path']):
    from data import build_dataset_img
    build_dataset_img(model, data_config)
else:
    from data import build_cached_dataset
    train_dataset, test_dataset = build_cached_dataset(data_config)

    train(model,
          optim,
          train_config,
          train_dataset,
          test_dataset)

shutil.make_archive(os.path.join(config['outcome_path']),
                    "zip", os.path.join(config['outcome_path']))
