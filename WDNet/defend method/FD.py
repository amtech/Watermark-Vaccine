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


import numpy as np
from numpy import pi
import math
import tensorflow as tf
from scipy.fftpack import dct, idct, rfft, irfft
from PIL import Image

T = np.array([
    [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
    [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
    [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
    [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
    [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
    [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
    [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
    [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
])

""
Jpeg_def_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 36, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

""
num = 8
q_table = np.ones((num, num)) * 30
q_table[0:4, 0:4] = 25
print(q_table)


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def rfft2(block):
    return rfft(rfft(block.T).T)


def irfft2(block):
    return irfft(irfft(block.T).T)


def FD_jpeg_encode(input_matrix):
    output = []
    input_matrix = (input_matrix + 1) / 2. * 255
    n = input_matrix.shape[0]
    input_matrix = np.array([np.array(Image.fromarray(np.uint8(input_matrix[i])).resize((304, 304))) for i in range(n)])

    h = input_matrix.shape[1]
    w = input_matrix.shape[2]
    c = input_matrix.shape[3]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    output3 = np.zeros((n, 3, h, w))
    vertical_blocks_num = h / num
    n_block = np.split(input_matrix, n, axis=0)
    for i in range(0, n):
        c_block = np.split(n_block[i], c, axis=3)
        j = 0
        for ch_block in c_block:
            vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=1)
            k = 0
            for block_ver in vertical_blocks:
                hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=2)
                m = 0
                for block in hor_blocks:
                    block = np.reshape(block, (num, num))
                    block = dct2(block)
                    # quantization
                    table_quantized = np.matrix.round(np.divide(block, q_table))
                    table_quantized = np.squeeze(np.asarray(table_quantized))
                    # de-quantization
                    table_unquantized = table_quantized * q_table
                    IDCT_table = idct2(table_unquantized)
                    if m == 0:
                        output = IDCT_table
                    else:
                        output = np.concatenate((output, IDCT_table), axis=1)
                    m = m + 1
                if k == 0:
                    output1 = output
                else:
                    output1 = np.concatenate((output1, output), axis=0)
                k = k + 1
            output2[j] = output1
            j = j + 1
        output3[i] = output2

    output3 = np.transpose(output3, (0, 2, 1, 3))
    output3 = np.transpose(output3, (0, 1, 3, 2))
    output3 = np.array([np.array(Image.fromarray(np.uint8(output3[i])).resize((299, 299))) for i in range(n)])
    output3 = output3 / 255
    output3 = np.clip(np.float32(output3), 0.0, 1.0)
    output3 = output3 * 2. - 1.
    return output3


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
    adv1_tf = FD_jpeg_encode(adv1_np)
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
    adv2_tf = FD_jpeg_encode(adv2_np)
    adv2_np = np.array(adv2_tf).transpose((0, 3, 1, 2))
    adv2 = torch.tensor(adv2_np)

    adv2, mask = generate_watermark_ori(adv2, logo_path, seed)
    adv2 = torch.unsqueeze(adv2.cuda(), 0)
    adv2_pred, adv2_mask, alpha, w, _ = G(adv2)

    print('Step %d' % (t))
    # 原始指标

    '''compress'''

    img_source_np = img_source.detach().cpu().numpy()
    img_source_np = img_source_np.transpose((0, 2, 3, 1))
    img_source_tf = FD_jpeg_encode(img_source_np)
    img_source_np = np.array(img_source_tf).transpose((0, 3, 1, 2))
    img_source_com = torch.tensor(img_source_np)

    img_source = torch.squeeze(img_source_com)

    clean_pred = torch.squeeze(clean_pred)
    img_source = transforms.ToPILImage()(img_source.detach().cpu()).convert('RGB')
    clean_pred = transforms.ToPILImage()(clean_pred.detach().cpu()).convert('RGB')
    mask = np.asarray(mask) / 255.0
    real_img = np.array(img_source)
    print(real_img.shape)

    # adv1 指标
    adv1_pred = torch.squeeze(adv1_pred)
    adv1_pred = transforms.ToPILImage()(adv1_pred.detach().cpu()).convert('RGB')
    g_img = np.array(adv1_pred)
    print(g_img.shape)
    print(real_img.shape)
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