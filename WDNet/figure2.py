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
from generate_wm import generate_watermark,generate_watermark_ori,generate_watermark_loc2
import pytorch_ssim
import math

print(torch.cuda.is_available())
G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G10.pkl'),map_location='cpu'))
G.cuda()

i = 0
all_time = 0.0
imageJ_path='./demo_src/Watermark_free_image/446.jpg'
logo_path = './dataset/CLWD/watermark_logo/test_color/7.png'
seed = 170

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

epsilon = (8/ 255.) / std
start_epsilon = (8 / 255.) / std
step_alpha = ( 2/ 255.) / std



mask_black = torch.zeros(1,256,256).cuda()
mask_black = torch.unsqueeze(mask_black,0)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
delta1 = torch.zeros_like(img_source).cuda()
delta2 = torch.zeros_like(img_source).cuda()
random_noise = torch.zeros_like(img_source).cuda()
for i in range(len(epsilon)):
    random_noise[:, i, :, :].uniform_(-start_epsilon[i][0][0].item(), start_epsilon[i][0][0].item())

delta1.requires_grad = True
delta2.requires_grad = True

ans_ssim=0.0
ans_psnr=0.0
rmse_all=0.0
rmse_in=0.0


a1_ans_ssim=0.0
a1_ans_psnr=0.0
a1_rmse_all=0.0
a1_rmse_in=0.0

a2_ans_ssim=0.0
a2_ans_psnr=0.0
a2_rmse_all=0.0
a2_rmse_in=0.0

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

#


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
wm = clamp(img_source, lower_limit , upper_limit )
img1 = wm
wm,mask = generate_watermark_loc2(img_source,logo_path,seed)
wm = torch.unsqueeze(wm.cuda(), 0)
clean_pred, clean_mask, alpha, w, _ = G(wm)


wm_r = clamp(img_source+random_noise, lower_limit, upper_limit)
wm_r,_ = generate_watermark_loc2(wm_r, logo_path, seed)
wm_r = torch.unsqueeze(wm_r.cuda(), 0)
random_pred, random_mask, alpha, w, _ = G(wm_r)


adv1 = clamp(img_source+delta1, lower_limit , upper_limit )
img2 = adv1
adv1,_ = generate_watermark_loc2(adv1,logo_path,seed)
adv1 = torch.unsqueeze(adv1.cuda(), 0)
adv1_pred, adv1_mask, alpha, w, _ = G(adv1)

adv2 = clamp(img_source+delta2, lower_limit , upper_limit )
img3 = adv2
adv2,_ = generate_watermark_loc2(adv2,logo_path,seed)
adv2 = torch.unsqueeze(adv2.cuda(), 0)
adv2_pred, adv2_mask, alpha, w, _ = G(adv2)



##保存图片
img_p = torch.squeeze(wm)
adv1 = torch.squeeze(adv1)
adv2 = torch.squeeze(adv2)
random = torch.squeeze(wm_r)
img1 = torch.squeeze(img1)
img2 = torch.squeeze(img2)
img3 = torch.squeeze(img3)



clean_p=torch.squeeze(clean_pred)
clean_mask_p = torch.squeeze(clean_mask)

adv1_p=torch.squeeze(adv1_pred)
adv1_mask_p = torch.squeeze(adv1_mask)

random_p=torch.squeeze(random_pred)
random_mask_p = torch.squeeze(random_mask)

adv2_p=torch.squeeze(adv2_pred)
adv2_mask_p = torch.squeeze(adv2_mask)



img_p = transforms.ToPILImage()(img_p.detach().cpu()).convert('RGB')
img_p.save('./figure1/img_source.jpg')
clean_p = transforms.ToPILImage()(clean_p.detach().cpu()).convert('RGB')
clean_p.save('./figure1/clean_pred.jpg')
clean_mask_p = transforms.ToPILImage()(clean_mask_p.detach().cpu()).convert('L')
clean_mask_p.save('./figure1/clean_mask.jpg')
adv1_p = transforms.ToPILImage()(adv1_p.detach().cpu()).convert('RGB')
adv1_p.save('./figure1/adv1_pred.jpg')
adv1_mask_p = transforms.ToPILImage()(adv1_mask_p.detach().cpu()).convert('L')
adv1_mask_p.save('./figure1/adv1_mask.jpg')
adv2_p = transforms.ToPILImage()(adv2_p.detach().cpu()).convert('RGB')
adv2_p.save('./figure1/adv2_pred.jpg')
adv2_mask_p = transforms.ToPILImage()(adv2_mask_p.detach().cpu()).convert('L')
adv2_mask_p.save('./figure1/adv2_mask.jpg')
adv1 = transforms.ToPILImage()(adv1.detach().cpu()).convert('RGB')
adv1.save('./figure1/adv1.jpg')
adv2 = transforms.ToPILImage()(adv2.detach().cpu()).convert('RGB')
adv2.save('./figure1/adv2.jpg')
img1 = transforms.ToPILImage()(img1.detach().cpu()).convert('RGB')
img1.save('./figure1/img1.jpg')
img2 = transforms.ToPILImage()(img2.detach().cpu()).convert('RGB')
img2.save('./figure1/img2.jpg')
img3 = transforms.ToPILImage()(img3.detach().cpu()).convert('RGB')
img3.save('./figure1/img3.jpg')


random = transforms.ToPILImage()(random.detach().cpu()).convert('RGB')
random .save('./figure1/random.jpg')
random_pred = transforms.ToPILImage()(random_p.detach().cpu()).convert('RGB')
random_pred.save('./figure1/random_pred.jpg')
random_mask_p = transforms.ToPILImage()(random_mask_p.detach().cpu()).convert('L')
random_mask_p.save('./figure1/random_mask.jpg')

#
# # 原始指标
# img_source = torch.squeeze(img_source)
# clean_pred = torch.squeeze(clean_pred)
# img_source = transforms.ToPILImage()(img_source.detach().cpu()).convert('RGB')
# clean_pred = transforms.ToPILImage()(clean_pred.detach().cpu()).convert('RGB')
# mask = np.asarray(mask) / 255.0
# g_img = np.array(clean_pred)
# real_img = np.array(img_source)
# ans_psnr += psnr(g_img, real_img)
# mse_all = mse(g_img, real_img)
# mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
#         np.sum(mask) + 1e-6)
# rmse_all += np.sqrt(mse_all)
# rmse_in += np.sqrt(mse_in)
# real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
# g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
# real_img_tensor = real_img_tensor.cuda()
# g_img_tensor = g_img_tensor.cuda()
# ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
# print('clean image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
# ans_psnr, ans_ssim , rmse_in , rmse_all))
#
# adv1_pred = torch.squeeze(adv1_pred)
# adv1_pred = transforms.ToPILImage()(adv1_pred.detach().cpu()).convert('RGB')
# g_img = np.array(adv1_pred)
# a1_ans_psnr += psnr(g_img, real_img)
# a1_mse_all = mse(g_img, real_img)
# a1_mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
#         np.sum(mask) + 1e-6)
# a1_rmse_all += np.sqrt(a1_mse_all)
# a1_rmse_in += np.sqrt(a1_mse_in)
# real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
# g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
# real_img_tensor = real_img_tensor.cuda()
# g_img_tensor = g_img_tensor.cuda()
# a1_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
# print('Max-loss image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
# a1_ans_psnr , a1_ans_ssim , a1_rmse_in , a1_rmse_all ))
#
# # adv2 指标
# adv2_pred = torch.squeeze(adv2_pred)
# adv2_pred = transforms.ToPILImage()(adv2_pred.detach().cpu()).convert('RGB')
# g_img = np.array(adv2_pred)
# a2_ans_psnr += psnr(g_img, real_img)
# a2_mse_all = mse(g_img, real_img)
# a2_mse_in = mse(g_img * mask, real_img * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
#         np.sum(mask) + 1e-6)
# a2_rmse_all += np.sqrt(a2_mse_all)
# a2_rmse_in += np.sqrt(a2_mse_in)
# real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
# g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
# real_img_tensor = real_img_tensor.cuda()
# g_img_tensor = g_img_tensor.cuda()
# a2_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
# print('Min-loss image:psnr: %.4f, ssim: %.4f, rmse_in: %.4f, rmse_all: %.4f' % (
#     a2_ans_psnr , a2_ans_ssim , a2_rmse_in , a2_rmse_all ))
#
#
# torch.save(delta1,'delta1_wdnet.pt')
# torch.save(delta2,'delta2_wdnet.pt')