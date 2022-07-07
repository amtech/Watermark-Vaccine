from WDNet import generator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp
import os
import time
from torchvision import datasets, transforms
import torch.nn.functional as F
from generate_wm import generate_watermark, generate_watermark_ori
import math
import pytorch_ssim

G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('../WDNet_G.pkl'), map_location='cpu'))
G.cuda()

transform_norm = transforms.Compose([transforms.ToTensor()])

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

epsilon = (8 / 255.) / std
start_epsilon = (8 / 255.) / std
step_alpha = (2 / 255.) / std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


from numpy.lib.stride_tricks import as_strided as ast


def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


import tensorflow as tf
import numpy as np


def bit_depth_reduce(xs, x_min, x_max, step_num, alpha=1e6):
    ''' Run bit depth reduce on xs.
    :param xs: A batch of images to apply bit depth reduction.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param step_num: Step number for bit depth reduction.
    :param alpha: Alpha for bit depth reduction.
    :return: Bit depth reduced xs.
    '''
    # due to tf.custom_gradient's limitation, we need a wrapper here
    @tf.custom_gradient
    def bit_depth_reduce_op(xs_tf):
        steps = x_min + np.arange(1, step_num, dtype=np.float32) / (step_num / (x_max - x_min))
        steps = steps.reshape([1, 1, 1, 1, step_num-1])
        tf_steps = tf.constant(steps, dtype=tf.float32)

        inputs = tf.expand_dims(xs_tf, 4)
        quantized_inputs = x_min + tf.reduce_sum(tf.sigmoid(alpha * (inputs - tf_steps)), axis=4)
        quantized_inputs = quantized_inputs / ((step_num-1) / (x_max - x_min))

        def bit_depth_reduce_grad(d_output):
            return d_output

        return quantized_inputs, bit_depth_reduce_grad

    return bit_depth_reduce_op(xs)


ans_ssim = 0.0
ans_psnr = 0.0
rmse_all = 0.0
rmse_in = 0.0

r_ans_ssim = 0.0
r_ans_psnr = 0.0
r_rmse_all = 0.0
r_rmse_in = 0.0

a1_ans_ssim = 0.0
a1_ans_psnr = 0.0
a1_rmse_all = 0.0
a1_rmse_in = 0.0

a2_ans_ssim = 0.0
a2_ans_psnr = 0.0
a2_rmse_all = 0.0
a2_rmse_in = 0.0

t = 0

np.random.seed(160)
for index in range(20):
    t += 1
    logo_index = np.random.randint(1, 161)
    imageJ_path = '../demo_src/Watermark_free_image/%s.jpg' % (t)
    logo_path = '../demo_logo/wm/train_color/%s.png' % (logo_index)
    img_J = Image.open(imageJ_path)
    img_source = transform_norm(img_J)
    img_source = torch.unsqueeze(img_source.cuda(), 0)
    mask_black = torch.zeros((1, 256, 256)).cuda()
    mask_black = torch.unsqueeze(mask_black, 0)

    # 初始化扰动
    delta1 = torch.zeros_like(img_source).cuda()
    delta2 = torch.zeros_like(img_source).cuda()
    random_noise = torch.zeros_like(img_source).cuda()
    seed = np.random.randint(0, 1000)

    # 干净
    wm = clamp(img_source, lower_limit, upper_limit)
    wm, mask = generate_watermark_ori(img_source, logo_path, seed)
    wm = torch.unsqueeze(wm.cuda(), 0)
    clean_pred, clean_mask, alpha, w, _ = G(wm)

    # random noise
    for i in range(len(epsilon)):
        random_noise[:, i, :, :].uniform_(-start_epsilon[i][0][0].item(), start_epsilon[i][0][0].item())
    wm_r = clamp(img_source + random_noise, lower_limit, upper_limit)
    wm_r, _ = generate_watermark_ori(wm_r, logo_path, seed)
    wm_r = torch.unsqueeze(wm_r.cuda(), 0)
    random_pred, clean_mask, alpha, w, _ = G(wm_r)

    # adv1
    delta1.requires_grad = True
    for i in range(20):
        start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = G(img_source + delta1)
        loss = F.mse_loss(img_source.data, start_pred_target.float()) + F.mse_loss(mask_black.data, start_mask.float())
        loss.backward()
        grad = delta1.grad.detach()
        d = delta1
        d = clamp(d + step_alpha * torch.sign(grad), -epsilon, epsilon)
        delta1.data = d
        delta1.grad.zero_()
    adv1 = clamp(img_source + delta1, lower_limit, upper_limit)

    '''compress'''
    adv1_np = adv1.detach().cpu().numpy()
    adv1_np = adv1_np.transpose((0, 2, 3, 1))
    adv1_tf = tf.constant(adv1_np)
    adv1_tf = bit_depth_reduce(adv1_tf,0, 1, step_num=6)
    adv1_np = np.array(adv1_tf).transpose((0, 3, 1, 2))
    adv1 = torch.tensor(adv1_np)

    adv1, _ = generate_watermark_ori(adv1, logo_path, seed)
    adv1 = torch.unsqueeze(adv1.cuda(), 0)
    adv1_pred, adv1_mask, alpha, w, _ = G(adv1)

    # adv2
    delta2.requires_grad = True
    for i in range(20):
        start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = G(img_source + delta2)
        loss = 2*F.mse_loss(img_source.data, start_pred_target.float()) + F.mse_loss(mask_black.data, start_mask.float())
        loss.backward()
        grad = -delta2.grad.detach()
        d = delta2
        d = clamp(d + step_alpha * torch.sign(grad), -epsilon, epsilon)
        delta2.data = d
        delta2.grad.zero_()
    adv2 = clamp(img_source + delta2, lower_limit, upper_limit)

    '''compress'''
    adv2_np = adv2.detach().cpu().numpy()
    adv2_np = adv2_np.transpose((0, 2, 3, 1))
    adv2_tf = tf.constant(adv2_np)
    adv2_tf = bit_depth_reduce(adv2_tf,0, 1, step_num=6)
    adv2_np = np.array(adv2_tf).transpose((0, 3, 1, 2))
    adv2 = torch.tensor(adv2_np)

    adv2, _ = generate_watermark_ori(adv2, logo_path, seed)
    adv2 = torch.unsqueeze(adv2.cuda(), 0)
    adv2_pred, adv2_mask, alpha, w, _ = G(adv2)

    print('Step %d' % (t))
    # 原始指标

    '''compress'''
    img_source_np = img_source.detach().cpu().numpy()
    img_source_np = img_source_np.transpose((0, 2, 3, 1))
    img_source_tf = tf.constant(img_source_np)
    img_source_tf = bit_depth_reduce(img_source_tf,0, 1, step_num=6)
    img_source_np = np.array(img_source_tf).transpose((0, 3, 1, 2))
    img_source_compress = torch.tensor(img_source_np)

    img_source = torch.squeeze(img_source_compress)
    clean_pred = torch.squeeze(clean_pred)
    img_source = transforms.ToPILImage()(img_source.detach().cpu()).convert('RGB')
    clean_pred = transforms.ToPILImage()(clean_pred.detach().cpu()).convert('RGB')
    mask = np.asarray(mask) / 255.0
    g_img = np.array(clean_pred)
    real_img = np.array(img_source)
    ans_psnr += psnr(g_img, real_img)
    mse_all = mse(g_img, real_img)
    mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    rmse_all += np.sqrt(mse_all)
    rmse_in += np.sqrt(mse_in)
    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    print('clean image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
    ans_psnr / t, ans_ssim / t, rmse_in / t, rmse_all / t))

    # random 指标
    random_pred = torch.squeeze(random_pred)
    random_pred = transforms.ToPILImage()(random_pred.detach().cpu()).convert('RGB')
    g_img = np.array(random_pred)
    r_ans_psnr += psnr(g_img, real_img)
    r_mse_all = mse(g_img, real_img)
    r_mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    r_rmse_all += np.sqrt(r_mse_all)
    r_rmse_in += np.sqrt(r_mse_in)
    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    r_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    print('random image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
    r_ans_psnr / t, r_ans_ssim / t, r_rmse_in / t, r_rmse_all / t))

    # adv1 指标
    adv1_pred = torch.squeeze(adv1_pred)
    adv1_pred = transforms.ToPILImage()(adv1_pred.detach().cpu()).convert('RGB')
    g_img = np.array(adv1_pred)
    a1_ans_psnr += psnr(g_img, real_img)
    a1_mse_all = mse(g_img, real_img)
    a1_mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    a1_rmse_all += np.sqrt(a1_mse_all)
    a1_rmse_in += np.sqrt(a1_mse_in)
    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    a1_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    print('Max-loss image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
        a1_ans_psnr / t, a1_ans_ssim / t, a1_rmse_in / t, a1_rmse_all / t))

    # adv2 指标
    adv2_pred = torch.squeeze(adv2_pred)
    adv2_pred = transforms.ToPILImage()(adv2_pred.detach().cpu()).convert('RGB')
    g_img = np.array(adv2_pred)
    a2_ans_psnr += psnr(g_img, real_img)
    a2_mse_all = mse(g_img, real_img)
    a2_mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    a2_rmse_all += np.sqrt(a2_mse_all)
    a2_rmse_in += np.sqrt(a2_mse_in)
    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    a2_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    print('Min-loss image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
        a2_ans_psnr / t, a2_ans_ssim / t, a2_rmse_in / t, a2_rmse_all / t))

    print('=' * 100)