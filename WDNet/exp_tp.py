#同一位置，不同logo
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
from generate_wm import generate_watermark,generate_watermark_tp


print(torch.cuda.is_available())
G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl'),map_location='cpu'))
G.cuda()

i = 0
all_time = 0.0
imageJ_path='./demo_src/Watermark_free_image/10.jpg'
logo_path1 = './demo_logo/wm/test_color/15.png'



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



epsilon = (8/ 255.) / std
start_epsilon = (8 / 255.) / std
step_alpha = ( 2/ 255.) / std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
delta1 = torch.zeros_like(img_source).cuda()
delta2 = torch.zeros_like(img_source).cuda()
random_noise = torch.zeros_like(img_source).cuda()

for i in range(len(epsilon)):
    random_noise[:, i, :, :].uniform_(-start_epsilon[i][0][0].item(), start_epsilon[i][0][0].item())


delta1.requires_grad = True
delta2.requires_grad = True




'''攻击1'''
for i in range(20):
    start_pred_target,  start_mask,  start_alpha,  start_w,  start_I_watermark = G(img_source+delta1)
    loss = F.mse_loss(img_source.data, start_pred_target.float())
    loss.backward()
    grad = -delta1.grad.detach()
    d=delta1
    d = clamp(d+step_alpha * torch.sign(grad), -epsilon, epsilon)
    delta1.data = d
    delta1.grad.zero_()

#


'''攻击2'''
for i in range(20):
    start_pred_target,  start_mask,  start_alpha,  start_w,  start_I_watermark = G(img_source+delta2)
    loss = F.mse_loss(img_source.data, start_pred_target.float())
    loss.backward()
    grad = delta2.grad.detach()
    d=delta2
    d = clamp(d+step_alpha * torch.sign(grad), -epsilon, epsilon)
    delta2.data = d
    delta2.grad.zero_()



seeds1 = np.random.randint(0,1000)
seeds2 = np.random.randint(0,1000)
seeds3 = np.random.randint(0,1000)

'''加水印'''
#logo1
wm1,mask1 = generate_watermark_tp(img_source,logo_path1,seeds1)
wm1 = torch.unsqueeze(wm1.cuda(), 0)
clean_pred1, clean_mask1, alpha, w, _ = G(wm1)

adv11 = clamp(img_source+delta1, lower_limit , upper_limit)
adv11,_ = generate_watermark_tp(adv11,logo_path1,seeds1)
adv11 = torch.unsqueeze(adv11.cuda(), 0)
adv1_pred1, adv1_mask1, alpha, w, _ = G(adv11)

adv21 = clamp(img_source+delta2, lower_limit , upper_limit )
adv21,_ = generate_watermark_tp(adv21,logo_path1,seeds1)
adv21 = torch.unsqueeze(adv21.cuda(), 0)
adv2_pred1, adv2_mask1, alpha, w, _ = G(adv21)



#logo2

wm2,mask2 = generate_watermark_tp(img_source,logo_path1,seeds2)
wm2 = torch.unsqueeze(wm2.cuda(), 0)
clean_pred2, clean_mask2, _, _, _ = G(wm2)

adv12 = clamp(img_source+delta1, lower_limit , upper_limit )
adv12,_ = generate_watermark_tp(adv12,logo_path1,seeds2)
adv12 = torch.unsqueeze(adv12.cuda(), 0)
adv1_pred2, adv1_mask2 ,_, _, _ = G(adv12)

adv22 = clamp(img_source+delta2, lower_limit , upper_limit )
adv22,_ = generate_watermark_tp(adv22,logo_path1,seeds2)
adv22 = torch.unsqueeze(adv22.cuda(), 0)
adv2_pred2, adv2_mask2 ,_, _, _ = G(adv22)



#logo3


wm3,mask3 = generate_watermark_tp(img_source,logo_path1,seeds3)
wm3 = torch.unsqueeze(wm3.cuda(), 0)
clean_pred3, clean_mask3, _, _, _ = G(wm3)

adv13 = clamp(img_source+delta1, lower_limit , upper_limit )
adv13,_ = generate_watermark_tp(adv13,logo_path1,seeds3)
adv13 = torch.unsqueeze(adv13.cuda(), 0)
adv1_pred3, adv1_mask3, _, _, _ = G(adv13)

adv23 = clamp(img_source+delta2, lower_limit , upper_limit )
adv23,_ = generate_watermark_tp(adv23,logo_path1,seeds3)
adv23 = torch.unsqueeze(adv23.cuda(), 0)
adv2_pred3, adv2_mask3, _, _, _ = G(adv23)




##保存图片
img_p1 = torch.squeeze(wm1)
adv11 = torch.squeeze(adv11)
adv21 = torch.squeeze(adv21)
img_p2 = torch.squeeze(wm2)
adv12 = torch.squeeze(adv12)
adv22 = torch.squeeze(adv22)
img_p3 = torch.squeeze(wm3)
adv13 = torch.squeeze(adv13)
adv23 = torch.squeeze(adv23)



clean_p1=torch.squeeze(clean_pred1)
clean_mask_p1 = torch.squeeze(clean_mask1)
adv1_p1=torch.squeeze(adv1_pred1)
adv1_mask_p1 = torch.squeeze(adv1_mask1)
adv2_p1=torch.squeeze(adv2_pred1)
adv2_mask_p1 = torch.squeeze(adv2_mask1)

clean_p2=torch.squeeze(clean_pred2)
clean_mask_p2 = torch.squeeze(clean_mask2)
adv1_p2=torch.squeeze(adv1_pred2)
adv1_mask_p2 = torch.squeeze(adv1_mask2)
adv2_p2=torch.squeeze(adv2_pred2)
adv2_mask_p2 = torch.squeeze(adv2_mask2)

clean_p3=torch.squeeze(clean_pred3)
clean_mask_p3 = torch.squeeze(clean_mask3)
adv1_p3=torch.squeeze(adv1_pred3)
adv1_mask_p3 = torch.squeeze(adv1_mask3)
adv2_p3=torch.squeeze(adv2_pred3)
adv2_mask_p3 = torch.squeeze(adv2_mask3)



img_p1 = transforms.ToPILImage()(img_p1.detach().cpu()).convert('RGB')
img_p1.save('./figure2/transparency/img_source1.jpg')
img_p2 = transforms.ToPILImage()(img_p2.detach().cpu()).convert('RGB')
img_p2.save('./figure2/transparency/img_source2.jpg')
img_p3 = transforms.ToPILImage()(img_p3.detach().cpu()).convert('RGB')
img_p3.save('./figure2/transparency/img_source3.jpg')


clean_p1 = transforms.ToPILImage()(clean_p1.detach().cpu()).convert('RGB')
clean_p1.save('./figure2/transparency/clean_pred1.jpg')
clean_p2 = transforms.ToPILImage()(clean_p2.detach().cpu()).convert('RGB')
clean_p2.save('./figure2/transparency/clean_pred2.jpg')
clean_p3 = transforms.ToPILImage()(clean_p3.detach().cpu()).convert('RGB')
clean_p3.save('./figure2/transparency/clean_pred3.jpg')

# adv11 = transforms.ToPILImage()(adv11.detach().cpu()).convert('RGB')
# adv11.save('./figure2/location/adv1_1.jpg')
# adv12 = transforms.ToPILImage()(adv12.detach().cpu()).convert('RGB')
# adv12.save('./figure2/location/adv1_2.jpg')
# adv13 = transforms.ToPILImage()(adv13.detach().cpu()).convert('RGB')
# adv13.save('./figure2/location/adv1_3.jpg')
#
# adv21 = transforms.ToPILImage()(adv21.detach().cpu()).convert('RGB')
# adv21.save('./figure2/location/adv2_1.jpg')
# adv22 = transforms.ToPILImage()(adv22.detach().cpu()).convert('RGB')
# adv22.save('./figure2/location/adv2_2.jpg')
# adv23 = transforms.ToPILImage()(adv23.detach().cpu()).convert('RGB')
# adv23.save('./figure2/location/adv2_3.jpg')

adv1_p1 = transforms.ToPILImage()(adv1_p1.detach().cpu()).convert('RGB')
adv1_p1.save('./figure2/transparency/adv1_p1.jpg')
adv1_p2 = transforms.ToPILImage()(adv1_p2.detach().cpu()).convert('RGB')
adv1_p2.save('./figure2/transparency/adv1_p2.jpg')
adv1_p3 = transforms.ToPILImage()(adv1_p3.detach().cpu()).convert('RGB')
adv1_p3.save('./figure2/transparency/adv1_p3.jpg')

adv1_mask_p1 = transforms.ToPILImage()(adv1_mask_p1.detach().cpu()).convert('RGB')
adv1_mask_p1.save('./figure2/transparency/adv1_mask_p1.jpg')
adv1_mask_p2 = transforms.ToPILImage()(adv1_mask_p2.detach().cpu()).convert('RGB')
adv1_mask_p2.save('./figure2/transparency/adv1_mask_p2.jpg')
adv1_mask_p3 = transforms.ToPILImage()(adv1_mask_p3.detach().cpu()).convert('RGB')
adv1_mask_p3.save('./figure2/transparency/adv1_mask_p3.jpg')

adv2_p1 = transforms.ToPILImage()(adv2_p1.detach().cpu()).convert('RGB')
adv2_p1.save('./figure2/transparency/adv2_p1.jpg')
adv2_p2 = transforms.ToPILImage()(adv2_p2.detach().cpu()).convert('RGB')
adv2_p2.save('./figure2/transparency/adv2_p2.jpg')
adv2_p3 = transforms.ToPILImage()(adv2_p3.detach().cpu()).convert('RGB')
adv2_p3.save('./figure2/transparency/adv2_p3.jpg')


adv2_mask_p1 = transforms.ToPILImage()(adv2_mask_p1.detach().cpu()).convert('RGB')
adv2_mask_p1.save('./figure2/transparency/adv2_mask_p1.jpg')
adv2_mask_p2 = transforms.ToPILImage()(adv2_mask_p2.detach().cpu()).convert('RGB')
adv2_mask_p2.save('./figure2/transparency/adv2_mask_p2.jpg')
adv2_mask_p3 = transforms.ToPILImage()(adv2_mask_p3.detach().cpu()).convert('RGB')
adv2_mask_p3.save('./figure2/transparency/adv2_mask_p3.jpg')

