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


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)


import tensorflow as tf


def jpeg_compress(xs, x_min, x_max, quality=95):
    ''' Run jpeg compress on xs.
    :param xs: A batch of images to compress.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param quality: Jpeg compress quality.
    :return: Compressed images tensor with same numerical scale to the input image.
    '''

    # due to tf.custom_gradient's limitation, we need a wrapper here
    @tf.custom_gradient
    def jpeg_compress_op(xs_tf):
        # batch_size x width x height x channel
        imgs = tf.cast((xs_tf - x_min) / ((x_max - x_min) / 255.0), tf.uint8)
        imgs_jpeg = tf.map_fn(lambda img: tf.image.decode_jpeg(tf.image.encode_jpeg(img, quality=quality)), imgs)
        imgs_jpeg.set_shape(xs_tf.shape)

        def jpeg_compress_grad(d_output):
            return d_output

        return tf.cast(imgs_jpeg, xs_tf.dtype) / (255.0 / (x_max - x_min)) + x_min, jpeg_compress_grad

    return jpeg_compress_op(xs)


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
    adv1_tf = jpeg_compress(adv1_tf, 0, 1)
    adv1_np = np.array(adv1_tf).transpose((0, 3, 1, 2))
    adv1 = torch.tensor(adv1_np)

    adv1, _ = generate_watermark_ori(adv1, logo_path, seed)
    adv1 = torch.unsqueeze(adv1.cuda(), 0)
    adv1_pred, adv1_mask, alpha, w, _ = G(adv1)

    # adv2
    delta2.requires_grad = True
    for i in range(20):
        start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = G(img_source + delta2)
        loss = 2 * F.mse_loss(img_source.data, start_pred_target.float()) + F.mse_loss(mask_black.data,
                                                                                       start_mask.float())
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
    adv2_tf = jpeg_compress(adv2_tf, 0, 1)
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
    img_source_tf = jpeg_compress(img_source_tf, 0, 1)
    img_source_np = np.array(img_source_tf).transpose((0, 3, 1, 2))
    img_source_compress = torch.tensor(img_source_np)


    img_p = torch.squeeze(wm)
    adv1 = torch.squeeze(adv1)
    adv2 = torch.squeeze(adv2)

    clean_p = torch.squeeze(clean_pred)
    clean_mask_p = torch.squeeze(clean_mask)
    adv1_p = torch.squeeze(adv1_pred)
    adv1_mask_p = torch.squeeze(adv1_mask)
    adv2_p = torch.squeeze(adv2_pred)
    adv2_mask_p = torch.squeeze(adv2_mask)

    #
    img_p = transforms.ToPILImage()(img_p.detach().cpu()).convert('RGB')
    img_p.save('./figure1/img_source.jpg')
    clean_p = transforms.ToPILImage()(clean_p.detach().cpu()).convert('RGB')
    clean_p.save('./figure1/clean_pred.jpg')
    clean_mask_p = transforms.ToPILImage()(clean_mask_p.detach().cpu()).convert('RGB')
    clean_mask_p.save('./figure1/clean_mask.jpg')
    adv1_p = transforms.ToPILImage()(adv1_p.detach().cpu()).convert('RGB')
    adv1_p.save('./figure1/adv1_pred.jpg')
    adv1_mask_p = transforms.ToPILImage()(adv1_mask_p.detach().cpu()).convert('RGB')
    adv1_mask_p.save('./figure1/adv1_mask.jpg')
    adv2_p = transforms.ToPILImage()(adv2_p.detach().cpu()).convert('RGB')
    adv2_p.save('./figure1/adv2_pred.jpg')
    adv2_mask_p = transforms.ToPILImage()(adv2_mask_p.detach().cpu()).convert('RGB')
    adv2_mask_p.save('./figure1/adv2_mask.jpg')
    adv1 = transforms.ToPILImage()(adv1.detach().cpu()).convert('RGB')
    adv1.save('./figure1/adv1.jpg')
    adv2 = transforms.ToPILImage()(adv2.detach().cpu()).convert('RGB')
    adv2.save('./figure1/adv2.jpg')

    print('=' * 100)