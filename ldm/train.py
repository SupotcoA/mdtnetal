import torch
from utils import Logger, vis_imgs, check_ae


def train(model,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          test_dataset):
    logger = Logger(init_val=0,
                    log_path=train_config['log_path'],
                    log_every_n_steps=train_config['log_every_n_steps'])
    for [x0, cls] in train_dataset:
        check_ae(model, x0.to(model.device), train_config['outcome_root'])
        imgs = model.validate_generation(x0.to(model.device))
        vis_imgs(imgs, "rev_dif_check", "rev_dif_check",
                 train_config['outcome_root'], use_plt=True)
        break
    for [x0, cls] in train_dataset:
        x0, cls = x0.to(model.device), cls.to(model.device)
        loss = model.train_step(x0, cls)
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.update(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps']==0:
            test(model,
                 train_config,
                 test_dataset)
            noised_images,rec_images = model.sim_training(x0, cls, batch_size=9)
            logger.start_generation()
            vis_imgs(noised_images, logger.step, "noised",
                     train_config['outcome_root'])
            vis_imgs(rec_images, logger.step, "rec",
                     train_config['outcome_root'])
            step_s = 400
            noised_images, rec_images = model.midway_generation(x0, cls, batch_size=9,
                                                                step_s=step_s)
            vis_imgs(noised_images, logger.step, f"noised{step_s}",
                     train_config['outcome_root'])
            vis_imgs(rec_images, logger.step, f"rec{step_s}",
                     train_config['outcome_root'])
            for cls in [0, 1, 2]:
                imgs = model.condional_generation(cls=cls, batch_size=9)
                vis_imgs(imgs, logger.step, cls, train_config['outcome_root'])
            logger.end_generation()
            model.train()
        if logger.step % train_config['train_steps']==0:
            break


@torch.no_grad()
def test(model,
         train_config,
         test_dataset):
    model.eval()
    acc_loss = 0
    step = 0
    for [x0, cls] in test_dataset:
        step+=1
        loss = model.train_step(x0.to(model.device), cls.to(model.device))
        acc_loss += loss.cpu().item()
    info = f"Test step\n" \
           + f"loss:{acc_loss / step:.4f}\n"
    print(info)
    with open(train_config['log_path'], 'a') as f:
        f.write(info)
