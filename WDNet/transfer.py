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
from generate_wm import generate_watermark,generate_watermark_ori,generate_watermark_loc
import pytorch_ssim
import math

print(torch.cuda.is_available())
G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl'),map_location='cpu'))
G.cuda()

i = 0
all_time = 0.0
imageJ_path='./demo_src/Watermark_free_image/5.jpg'
logo_path = './dataset/CLWD/watermark_logo/train_color/16.png'


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

t = 1

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

seed = 160
delta1.requires_grad = True
delta2.requires_grad = True

ans_ssim=0.0
ans_psnr=0.0
rmse_all=0.0
rmse_in=0.0

r_ans_ssim=0.0
r_ans_psnr=0.0
r_rmse_all=0.0
r_rmse_in=0.0

a1_ans_ssim=0.0
a1_ans_psnr=0.0
a1_rmse_all=0.0
a1_rmse_in=0.0

a2_ans_ssim=0.0
a2_ans_psnr=0.0
a2_rmse_all=0.0
a2_rmse_in=0.0


delta_a1= torch.load('./delta1_split.pt')
delta_b1= torch.load('./delta1_bmvc.pt')/255
delta_c1= torch.load('./delta1_wdnet.pt')
delta1 = (delta_a1+delta_b1+delta_c1)
delta1 = clamp(delta1, -epsilon, epsilon)

delta_a2= torch.load('./delta2_split.pt')
delta_b2= torch.load('./delta2_bmvc.pt')/255
delta_c2= torch.load('./delta2_wdnet.pt')
delta2 = (delta_a2+delta_b2+delta_c2)
delta2 = clamp(delta2, -epsilon, epsilon)



'''加水印'''
wm = clamp(img_source+random_noise, lower_limit , upper_limit )
wm,mask = generate_watermark_loc(img_source,logo_path,seed)
wm = torch.unsqueeze(wm.cuda(), 0)
clean_pred, clean_mask, alpha, w, _ = G(wm)

for i in range(len(epsilon)):
    random_noise[:, i, :, :].uniform_(-start_epsilon[i][0][0].item(), start_epsilon[i][0][0].item())
wm_r = clamp(img_source + random_noise, lower_limit, upper_limit)
wm_r, _ = generate_watermark_loc(wm_r, logo_path, seed)
wm_r = torch.unsqueeze(wm_r.cuda(), 0)
random_pred, clean_mask, alpha, w, _ = G(wm_r)

adv1 = clamp(img_source+delta1, lower_limit , upper_limit )
adv1,_ = generate_watermark_loc(adv1,logo_path,seed)
adv1 = torch.unsqueeze(adv1.cuda(), 0)
adv1_pred, adv1_mask, alpha, w, _ = G(adv1)

adv2 = clamp(img_source+delta2, lower_limit , upper_limit )
adv2,_ = generate_watermark_loc(adv2,logo_path,seed)
adv2 = torch.unsqueeze(adv2.cuda(), 0)
adv2_pred, adv2_mask, alpha, w, _ = G(adv2)



##保存图片
img_p = torch.squeeze(wm)
adv1 = torch.squeeze(adv1)
adv2 = torch.squeeze(adv2)

clean_p=torch.squeeze(clean_pred)
clean_mask_p = torch.squeeze(clean_mask)
adv1_p=torch.squeeze(adv1_pred)
adv1_mask_p = torch.squeeze(adv1_mask)
adv2_p=torch.squeeze(adv2_pred)
adv2_mask_p = torch.squeeze(adv2_mask)


# 原始指标
img_source_tick = img_source
img_source = torch.squeeze(img_source)
clean_pred = torch.squeeze(clean_pred)
wm_clean = torch.squeeze(wm)

img_source = transforms.ToPILImage()(img_source.detach().cpu()).convert('RGB')
clean_pred = transforms.ToPILImage()(clean_pred.detach().cpu()).convert('RGB')
wm_clean = transforms.ToPILImage()(wm_clean.detach().cpu()).convert('RGB')

mask = np.asarray(mask) / 255.0
g_img = np.array(clean_pred)
real_img = np.array(img_source)
wm_clean = np.array(wm_clean)

ans_psnr += psnr(g_img, real_img)
mse_all = mse(g_img, real_img)
mse_in = mse(g_img * mask, wm_clean * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
        np.sum(mask) + 1e-6)
rmse_all += np.sqrt(mse_all)
rmse_in += np.sqrt(mse_in)
real_img_tensor = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
real_img_tensor = real_img_tensor.cuda()
g_img_tensor = g_img_tensor.cuda()
ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
print('clean image:psnr: %.4f, ssim: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (
ans_psnr / t, ans_ssim / t, rmse_all / t, rmse_in / t))

# random 指标
real_random = clamp(img_source_tick + random_noise, lower_limit, upper_limit)
real_random = torch.squeeze(real_random)
random_pred = torch.squeeze(random_pred)
wm_random = torch.squeeze(wm_r)

real_random = transforms.ToPILImage()(real_random.detach().cpu()).convert('RGB')
random_pred = transforms.ToPILImage()(random_pred.detach().cpu()).convert('RGB')
wm_random = transforms.ToPILImage()(wm_random.detach().cpu()).convert('RGB')
real_random = np.array(real_random)
g_img = np.array(random_pred)
wm_random = np.array(wm_random)

r_ans_psnr += psnr(g_img, real_random)
r_mse_all = mse(g_img, real_random)
r_mse_in = mse(g_img * mask, wm_random * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
        np.sum(mask) + 1e-6)
r_rmse_all += np.sqrt(r_mse_all)
r_rmse_in += np.sqrt(r_mse_in)
real_img_tensor = torch.from_numpy(real_random).float().unsqueeze(0) / 255.0
g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
real_img_tensor = real_img_tensor.cuda()
g_img_tensor = g_img_tensor.cuda()
r_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
print('random image:psnr: %.4f, ssim: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (
r_ans_psnr / t, r_ans_ssim / t, r_rmse_all / t, r_rmse_in / t))

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
a1_mse_in = mse(g_img * mask, adv1_wm * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
        np.sum(mask) + 1e-6)
a1_rmse_all += np.sqrt(a1_mse_all)
a1_rmse_in += np.sqrt(a1_mse_in)
real_img_tensor = torch.from_numpy(real_adv1).float().unsqueeze(0) / 255.0
g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
real_img_tensor = real_img_tensor.cuda()
g_img_tensor = g_img_tensor.cuda()
a1_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
print('Max-loss image:psnr: %.4f, ssim: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (
    a1_ans_psnr / t, a1_ans_ssim / t, a1_rmse_all / t, a1_rmse_in / t))

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

a2_ans_psnr += psnr(g_img, real_adv2)
a2_mse_all = mse(g_img, real_adv2)
a2_mse_in = mse(g_img * mask, adv2_wm * mask) * mask.shape[0] * mask.shape[1] * mask.shape[2] / (
        np.sum(mask) + 1e-6)
a2_rmse_all += np.sqrt(a2_mse_all)
a2_rmse_in += np.sqrt(a2_mse_in)
real_img_tensor = torch.from_numpy(real_adv2).float().unsqueeze(0) / 255.0
g_img_tensor = torch.from_numpy(g_img).float().unsqueeze(0) / 255.0
real_img_tensor = real_img_tensor.cuda()
g_img_tensor = g_img_tensor.cuda()
a2_ans_ssim += pytorch_ssim.ssim(g_img_tensor, real_img_tensor)
print('Min-loss image:psnr: %.4f, ssim: %.4f, rmse_all: %.4f, rmse_in: %.4f' % (
    a2_ans_psnr / t, a2_ans_ssim / t, a2_rmse_all / t, a2_rmse_in / t))


#visualization
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