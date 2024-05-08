import numpy as np
import cv2
import os
import torch
import time
import matplotlib.pyplot as plt


@torch.no_grad()
def print_num_params(model, name, log_path):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    with open(log_path, 'a') as f:
        f.write(f"{name} parameters: {num_params}\n")
    print(f"{name} parameters: {num_params}")


@torch.no_grad()
def tensor2bgr(tensor):
    imgs = torch.clip(torch.permute(tensor, [0, 2, 3, 1]).cpu().add(1).mul(127.5), 0, 255)
    return imgs.numpy().astype(np.uint8)[:, :, :, ::-1]


@torch.no_grad()
def vis_imgs(imgs, step, cls, root, use_plt=False):
    imgs = tensor2bgr(imgs)
    if imgs.shape[0] > 9:
        imgs = imgs[:9]
    elif imgs.shape[0] < 9:
        raise ValueError(f"{imgs.shape}")
    h, w, c = imgs.shape[1:]
    base = np.zeros((h * 3, w * 3, c), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            base[i * h:i * h + h, j * w:j * w + w, :] = imgs[i * 3 + j]
    fp = os.path.join(root, f"s{step}_{cls}.png")
    cv2.imwrite(fp, base)
    if use_plt:
        plt.imshow(base[:, :, ::-1])
        plt.show()


class Logger:
    def __init__(self,
                 init_val=0,
                 log_path=None,
                 log_every_n_steps=None):
        self.val = init_val
        self.step = 0
        self.log_path = log_path
        self.log_every_n_steps = log_every_n_steps
        self.time = 0
        self.eval_time = 0

    def update(self, val):
        if self.val == 0:
            self.time = time.time()
        self.val += val
        self.step += 1
        if self.step % self.log_every_n_steps == 0:
            self.log()
            self.val = 0

    def log(self):
        dt = time.time() - self.time
        info = f"Train step {self.step}\n" \
               + f"loss:{self.val / self.log_every_n_steps:.4f}\n" \
               + f"time: {dt:.1f} \n"
        print(info)
        with open(self.log_path, 'a') as f:
            f.write(info)

    def start_generation(self):
        self.eval_time = time.time()

    def end_generation(self):
        dt = time.time() - self.eval_time
        info = f"generation time: {dt:.2f}\n"
        print(info)
        with open(self.log_path, 'a') as f:
            f.write(info)


@torch.no_grad()
def check_ae(model, x, root):
    if x.shape[0] < 9:
        return
    imgs = model.decode(x)[:9]
    vis_imgs(imgs, "ae_check", "ae_check", root, use_plt=True)


