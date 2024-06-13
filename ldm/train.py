import torch
from utils import Logger, vis_imgs, check_ae
import os


def train(model,
          optim,
          lr_scheduler,
          train_config,
          data_config,
          train_dataset,
          test_dataset):
    def conditional_generation(guidance_scales=[1, ]):
        import matplotlib.pyplot as plt
        exp_noise, pred_noise = model.validate_condional_generation(0)
        plt.plot(exp_noise)
        plt.plot(pred_noise)
        plt.show()
        for cls, stop_t in zip([0, 1, 2, 4, 5], [200, 200, 200, 150, 200]):  ### 3 = fa

            for guidance_scale in guidance_scales:
                # imgs = model.condional_generation(cls=cls,
                #                                   batch_size=9,
                #                                   guidance_scale=guidance_scale)
                # vis_imgs(imgs,
                #          logger.step,
                #          f"g{guidance_scale}_{data_config['dataset_names'][cls]}",
                #          train_config['outcome_root'])
                if cls != 0:
                    bs = 4
                    imgs = model.seq_condional_generation(cls=cls,
                                                          n_steps=10,
                                                          guidance_scale=guidance_scale,
                                                          batch_size=bs)
                    vis_imgs(imgs,
                             logger.step,
                             f"seq_g{guidance_scale}_{data_config['dataset_names'][cls]}",
                             train_config['outcome_root'],
                             seq=bs)
                # imgs = model.halfway_condional_generation(cls=cls,
                #                                           batch_size=9,
                #                                           stop_t=stop_t,
                #                                           guidance_scale=guidance_scale)
                # vis_imgs(imgs,
                #          logger.step,
                #          f"half_g{guidance_scale}_{data_config['dataset_names'][cls]}",
                #          train_config['outcome_root'])

    logger = Logger(init_val=0,
                    log_path=train_config['log_path'],
                    log_every_n_steps=train_config['log_every_n_steps'])
    if train_config['train_steps'] == 0:
        model.eval()
        print("Start inference")
        conditional_generation(guidance_scales=[1, 2, 3, 4, 5, 6])
        return

    for [x0, cls] in train_dataset:
        model.eval()
        check_ae(model, x0.to(model.device), train_config['outcome_root'])
        # imgs = model.validate_generation(x0.to(model.device))
        # vis_imgs(imgs, "rev_dif_check", "rev_dif_check",
        #          train_config['outcome_root'], use_plt=True)
        break

    for [x0, cls] in train_dataset:
        model.train()
        x0, cls = x0.to(model.device), cls.to(model.device)
        loss = model.train_step(x0, cls)
        optim.zero_grad()
        loss.backward()
        optim.step()
        logger.update(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            model.eval()
            test(model,
                 train_config,
                 test_dataset)
            noised_images, rec_images = model.sim_training(x0, cls, batch_size=9)
            logger.start_generation()
            vis_imgs(noised_images, logger.step, "noised",
                     train_config['outcome_root'])
            vis_imgs(rec_images, logger.step, "rec",
                     train_config['outcome_root'])
            step_s = 500
            noised_images, rec_images = model.midway_generation(x0, cls, batch_size=9,
                                                                step_s=step_s)
            vis_imgs(noised_images, logger.step, f"noised{step_s}",
                     train_config['outcome_root'])
            vis_imgs(rec_images, logger.step, f"rec{step_s}",
                     train_config['outcome_root'])
            conditional_generation()
            logger.end_generation()
        if logger.step % train_config['train_steps'] == 0:
            conditional_generation(guidance_scales=[2, 3, 4, 5, 6])
            if train_config['save']:
                state_dict = {name: param for name, param in model.cpu().state_dict().items()
                              if not (name.startswith('ae.') or name.startswith('sampler.'))}
                torch.save(state_dict,
                           os.path.join(train_config['outcome_root'], f"ldm{logger.step}.pth"))
            break


@torch.no_grad()
def test(model,
         train_config,
         test_dataset):
    model.eval()
    acc_loss = 0
    step = 0
    for [x0, cls] in test_dataset:
        step += 1
        loss = model.train_step(x0.to(model.device), cls.to(model.device))
        acc_loss += loss.cpu().item()
    info = f"Test step\n" \
           + f"loss:{acc_loss / step:.4f}\n"
    print(info)
    with open(train_config['log_path'], 'a') as f:
        f.write(info)
