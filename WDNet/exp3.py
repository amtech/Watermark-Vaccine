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
from generate_wm import generate_watermark,generate_watermark_ori
import pytorch_ssim
import math
print(torch.cuda.is_available())
G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl'),map_location='cpu'))
G.cuda()

i = 0
all_time = 0.0
imageJ_path='./demo_src/Watermark_free_image/449.jpg'
logo_path = './dataset/CLWD/watermark_logo/train_color/106.png'


transform_norm = transforms.Compose([transforms.ToTensor()])
img_J = Image.open(imageJ_path)
img_source = transform_norm(img_J)
img_source = torch.unsqueeze(img_source.cuda(), 0)
st = time.time()
std = (1.0, 1.0, 1.0)
std = torch.tensor(std).view(3,1,1).cuda()

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

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

# epsilon = (8/ 255.) / std
# start_epsilon = (8 / 255.) / std
step_alpha = ( 2/ 255.) / std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

mask_black = torch.zeros((1,256,256)).cuda()
mask_black = torch.unsqueeze(mask_black,0)
seed = 160



r_rmse_all_history = []
a1_rmse_all_history = []

r_rmse_in_history = []
a2_rmse_in_history = []



for limit in range(17):

    epsilon = (limit/255.)/std
    start_epsilon = (limit / 255.) / std
    delta1 = torch.zeros_like(img_source).cuda()
    delta2 = torch.zeros_like(img_source).cuda()
    random_noise = torch.zeros_like(img_source).cuda()
    for i in range(len(epsilon)):
        random_noise[:, i, :, :].uniform_(-start_epsilon[i][0][0].item(), start_epsilon[i][0][0].item())
    delta1.requires_grad = True
    delta2.requires_grad = True

    '''攻击1'''
    for i in range(50):
        start_pred_target,  start_mask,  start_alpha,  start_w,  start_I_watermark = G(img_source+delta1)
        loss = F.mse_loss(img_source.data, start_pred_target.float())
        loss.backward()
        grad = delta1.grad.detach()
        d=delta1
        d = clamp(d+step_alpha * torch.sign(grad), -epsilon, epsilon)
        delta1.data = d
        delta1.grad.zero_()

    '''攻击2'''
    for i in range(50):
        start_pred_target,  start_mask,  start_alpha,  start_w,  start_I_watermark = G(img_source+delta2)
        loss = 2*F.mse_loss(img_source.data, start_pred_target.float())+ F.mse_loss(mask_black.data, start_mask.float())
        loss.backward()
        grad = -delta2.grad.detach()
        d=delta2
        d = clamp(d+step_alpha * torch.sign(grad), -epsilon, epsilon)
        delta2.data = d
        delta2.grad.zero_()


    '''加水印'''
    wm = clamp(img_source+random_noise, lower_limit , upper_limit )
    wm,mask = generate_watermark_ori(img_source,logo_path,seed)
    wm = torch.unsqueeze(wm.cuda(), 0)
    clean_pred, clean_mask, alpha, w, _ = G(wm)

    adv1 = clamp(img_source+delta1, lower_limit , upper_limit )
    adv1,_ = generate_watermark_ori(adv1,logo_path,seed)
    adv1 = torch.unsqueeze(adv1.cuda(), 0)
    adv1_pred, adv1_mask, alpha, w, _ = G(adv1)

    adv2 = clamp(img_source+delta2, lower_limit , upper_limit )
    adv2,_ = generate_watermark_ori(adv2,logo_path,seed)
    adv2 = torch.unsqueeze(adv2.cuda(), 0)
    adv2_pred, adv2_mask, alpha, w, _ = G(adv2)

    # random noise
    wm_r = clamp(img_source + random_noise, lower_limit, upper_limit)
    wm_r, _ = generate_watermark_ori(wm_r, logo_path, seed)
    wm_r = torch.unsqueeze(wm_r.cuda(), 0)
    random_pred, clean_mask, alpha, w, _ = G(wm_r)

    ##保存图片
    img_p = torch.squeeze(wm)
    adv1_s = torch.squeeze(adv1)
    adv2_s = torch.squeeze(adv2)

    clean_p=torch.squeeze(clean_pred)
    clean_mask_p = torch.squeeze(clean_mask)
    adv1_p=torch.squeeze(adv1_pred)
    adv1_mask_p = torch.squeeze(adv1_mask)
    adv2_p=torch.squeeze(adv2_pred)
    adv2_mask_p = torch.squeeze(adv2_mask)


    img_p = transforms.ToPILImage()(img_p.detach().cpu()).convert('RGB')
    img_p.save('./figure3/img_source_%d.jpg'%limit)
    clean_p = transforms.ToPILImage()(clean_p.detach().cpu()).convert('RGB')
    clean_p.save('./figure3/clean_pred_%d.jpg'%limit)
    clean_mask_p = transforms.ToPILImage()(clean_mask_p.detach().cpu()).convert('RGB')
    clean_mask_p.save('./figure3/clean_mask_%d.jpg'%limit)
    adv1_p = transforms.ToPILImage()(adv1_p.detach().cpu()).convert('RGB')
    adv1_p.save('./figure3/adv1_pred_%d.jpg'%limit)
    adv1_mask_p = transforms.ToPILImage()(adv1_mask_p.detach().cpu()).convert('RGB')
    adv1_mask_p.save('./figure3/adv1_mask_%d.jpg'%limit)
    adv2_p = transforms.ToPILImage()(adv2_p.detach().cpu()).convert('RGB')
    adv2_p.save('./figure3/adv2_pred_%d.jpg'%limit)
    adv2_mask_p = transforms.ToPILImage()(adv2_mask_p.detach().cpu()).convert('RGB')
    adv2_mask_p.save('./figure3/adv2_mask_%d.jpg'%limit)
    adv1_s = transforms.ToPILImage()(adv1_s.detach().cpu()).convert('RGB')
    adv1_s.save('./figure3/adv1.jpg_%d.jpg'%limit)
    adv2_s = transforms.ToPILImage()(adv2_s.detach().cpu()).convert('RGB')
    adv2_s.save('./figure3/adv2.jpg_%d.jpg'%limit)

    ans_ssim_h = 0.0
    ans_psnr_h = 0.0
    ans_ssim_w = 0.0
    ans_psnr_w = 0.0
    rmse_all_h = 0.0
    rmse_in_h = 0.0
    rmse_all_w = 0.0
    rmse_in_w = 0.0

    r_ans_ssim_h = 0.0
    r_ans_psnr_h = 0.0
    r_ans_ssim_w = 0.0
    r_ans_psnr_w = 0.0
    r_rmse_all_h = 0.0
    r_rmse_in_h = 0.0
    r_rmse_all_w = 0.0
    r_rmse_in_w = 0.0

    a1_ans_ssim = 0.0
    a1_ans_psnr = 0.0
    a1_rmse_all = 0.0
    a1_rmse_in = 0.0

    a2_ans_ssim = 0.0
    a2_ans_psnr = 0.0
    a2_rmse_all = 0.0
    a2_rmse_in = 0.0

    print('%d/255 step'%limit)
    img_eval = transforms.ToPILImage()(img_source[0].detach().cpu()).convert('RGB')
    real_img = np.array(img_eval)

    ##原始指标
    img_source_tick = img_source
    img_source_tensor = torch.squeeze(img_source)
    clean_pred =torch.squeeze(clean_pred)
    wm_clean = torch.squeeze(wm)

    img_source_tensor = transforms.ToPILImage()(img_source_tensor.detach().cpu()).convert('RGB')
    clean_pred = transforms.ToPILImage()(clean_pred.detach().cpu()).convert('RGB')
    wm_clean = transforms.ToPILImage()(wm_clean.detach().cpu()).convert('RGB')

    mask = np.asarray(mask) / 255.0
    g_img = np.array(clean_pred)
    real_img = np.array(img_source_tensor)
    wm_clean = np.array(wm_clean)

    ans_psnr_h += psnr(g_img, real_img)
    ans_psnr_w += psnr(g_img, wm_clean)

    mse_all_h = mse(g_img, real_img)
    mse_in_h = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
                np.sum(mask) + 1e-6)
    rmse_all_h += np.sqrt(mse_all_h)
    rmse_in_h += np.sqrt(mse_in_h)

    mse_all_w = mse(g_img, wm_clean)
    mse_in_w = mse(g_img * mask, wm_clean * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    rmse_all_w += np.sqrt(mse_all_w)
    rmse_in_w += np.sqrt(mse_in_w)

    real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    wm_clean_tensor = torch.from_numpy(wm_clean).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    w_img_tensor = wm_clean_tensor.cuda()
    ans_ssim_h += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    ans_ssim_w += pytorch_ssim.ssim(w_img_tensor, real_img_tensor)


    print('clean image:psnr_h: %.4f, ssim_h: %.4f, rmse_all_h: %.4f,rmse_in_h: %.4f' % (ans_psnr_h , ans_ssim_h , rmse_all_h ,rmse_in_h ,))
    print('clean image:psnr_w: %.4f, ssim_w: %.4f, rmse_all_w: %.4f,rmse_in_w: %.4f' % (ans_psnr_w , ans_ssim_w ,  rmse_all_w ,rmse_in_w ,))

    #random 指标
    real_random = clamp(img_source_tick+random_noise, lower_limit, upper_limit)
    real_random = torch.squeeze(real_random)
    random_pred =torch.squeeze(random_pred)
    wm_random = torch.squeeze(wm_r)

    real_random = transforms.ToPILImage()(real_random.detach().cpu()).convert('RGB')
    random_pred = transforms.ToPILImage()(random_pred.detach().cpu()).convert('RGB')
    wm_random = transforms.ToPILImage()(wm_random.detach().cpu()).convert('RGB')
    real_random = np.array(real_random)
    g_img = np.array(random_pred)
    wm_random = np.array(wm_random)

    r_ans_psnr_h += psnr(g_img, real_random)
    r_ans_psnr_w += psnr(g_img, wm_random)

    r_mse_all_h = mse(g_img, real_random)
    r_mse_in_h = mse(g_img * mask, real_random * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
                np.sum(mask) + 1e-6)
    r_rmse_all_h += np.sqrt(r_mse_all_h)
    r_rmse_in_h += np.sqrt(r_mse_in_h)

    r_mse_all_w = mse(g_img, wm_random)
    r_mse_in_w = mse(g_img * mask, wm_random * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    r_rmse_all_w += np.sqrt(r_mse_all_w)
    r_rmse_in_w += np.sqrt(r_mse_in_w)


    real_img_tensor = torch.from_numpy(real_random).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    wm_random_tensor = torch.from_numpy(wm_random).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    wm_random_tensor = wm_random_tensor.cuda()
    r_ans_ssim_h += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    r_ans_ssim_w += pytorch_ssim.ssim(g_img_tensor, wm_random_tensor)

    print('random image:psnr_h: %.4f, ssim_h: %.4f, rmse_all_h: %.4f, rmse_in_w: %.4f' % (r_ans_psnr_h , r_ans_ssim_h , r_rmse_all_h ,r_rmse_in_h ))
    print('random image:psnr_w: %.4f, ssim_w: %.4f, rmse_all_h: %.4f, rmse_in_w: %.4f' % (r_ans_psnr_w , r_ans_ssim_w , r_rmse_all_w ,r_rmse_in_w ))

    # adv1 指标
    real_adv1 = clamp(img_source_tick + delta1, lower_limit, upper_limit)

    real_adv1 = torch.squeeze(real_adv1)
    adv1_pred = torch.squeeze(adv1_pred)
    adv1_wm = torch.squeeze(adv1)

    real_adv1 = transforms.ToPILImage()(real_adv1.detach().cpu()).convert('RGB')
    adv1_pred = transforms.ToPILImage()(adv1_pred.detach().cpu()).convert('RGB')
    adv1_wm = transforms.ToPILImage()(adv1_wm.detach().cpu()).convert('RGB')

    real_adv1 = np.array(real_adv1)
    g_img = np.array(adv1_pred)
    adv1_wm = np.array(adv1_wm)

    a1_ans_psnr += psnr(g_img, real_adv1)
    a1_mse_all = mse(g_img, real_adv1)
    a1_mse_in = mse(g_img * mask, real_adv1 * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    a1_rmse_all += np.sqrt(a1_mse_all)
    a1_rmse_in += np.sqrt(a1_mse_in)
    real_img_tensor = torch.from_numpy(real_adv1).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    a1_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
    print('Max-loss image:psnr_h: %.4f, ssim_h: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (a1_ans_psnr , a1_ans_ssim , a1_rmse_all ,a1_rmse_in ))

    # adv2 指标
    real_adv2 = clamp(img_source_tick + delta2, lower_limit, upper_limit)

    real_adv2 = torch.squeeze(real_adv2)
    adv2_pred = torch.squeeze(adv2_pred)
    adv2_wm = torch.squeeze(adv2)
    real_adv2 = transforms.ToPILImage()(real_adv2.detach().cpu()).convert('RGB')
    adv2_pred = transforms.ToPILImage()(adv2_pred.detach().cpu()).convert('RGB')
    adv2_wm = transforms.ToPILImage()(adv2_wm.detach().cpu()).convert('RGB')
    real_adv2 = np.array(real_adv2)
    g_img = np.array(adv2_pred)
    adv2_wm = np.array(adv2_wm)

    a2_ans_psnr += psnr(g_img, adv2_wm)
    a2_mse_all = mse(g_img, adv2_wm)
    a2_mse_in = mse(g_img * mask, adv2_wm * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
            np.sum(mask) + 1e-6)
    a2_rmse_all += np.sqrt(a2_mse_all)
    a2_rmse_in += np.sqrt(a2_mse_in)
    real_img_tensor = torch.from_numpy(real_adv2).float().unsqueeze(0) / 255.0
    g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
    adv2_wm_tensor = torch.from_numpy(adv2_wm).float().unsqueeze(0) / 255.0
    real_img_tensor = real_img_tensor.cuda()
    g_img_tensor = g_img_tensor.cuda()
    adv2_wm_tensor = adv2_wm_tensor.cuda()
    a2_ans_ssim += pytorch_ssim.ssim(g_img_tensor, adv2_wm_tensor)
    print('Min-loss image:psnr_w: %.4f, ssim_w: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (
        a2_ans_psnr , a2_ans_ssim ,  a2_rmse_all ,  a2_rmse_in ))

    r_rmse_all_history.append(r_rmse_all_h)
    a1_rmse_all_history.append(a1_rmse_all)

    r_rmse_in_history.append(r_rmse_in_w)
    a2_rmse_in_history.append(a2_rmse_in)

    print('=' * 100)





print('*'*100)
print(r_rmse_in_history)
print(a2_rmse_in_history)

print('*'*100)
print(r_rmse_all_history)
print(a1_rmse_all_history)

