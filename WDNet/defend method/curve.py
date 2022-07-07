import matplotlib.pyplot as plt
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
# pre-datas
x = np.arange(100, 0, -10)  # x坐标

bmap = brewer2mpl.get_map('Set3','qualitative',10)
colors = bmap.mpl_colors


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
rmse1= [3.09 ,3.08,3.08 ,3.10 ,3.11 ,3.11,3.08,3.09,3.09,3.14 ]
rmse2=[4.87 ,	5.49 ,	5.55 ,	5.59 ,	5.92 ,	7.26 ,	6.17 ,	5.99 ,	6.03 ,	6.39]
rmse3=[7.54 ,	6.74 ,	6.10 ,	6.00 ,	6.47 ,	7.40 ,	6.81 ,	6.73 ,	6.89 ,	7.08]

rmsew1=[54.58 ,	54.37 ,	54.54 ,	54.26 ,	54.18 ,	54.39 ,	54.65 ,	54.54 ,	54.61 ,	53.59]
rmsew2=[54.78 ,	54.49 ,	54.27 ,	54.30 ,	54.40 ,	54.30 ,	54.62 ,	54.62 ,	54.46 ,	53.30]
rmsew3=[35.28 ,	41.50 ,	49.95 ,	51.56 ,	52.37 ,	50.59 ,	53.88 ,	54.35 ,	54.49 ,	53.50]

ax1.plot(x, rmse1, lw=1, c='r', marker='s', ms=4,label='RMSE$^h$(Clean)')  # 绘制y1
ax1.plot(x, rmse2, lw=1, c='g', marker='o',  ms=4,label='RMSE$^h$(RN)')  # 绘制y2
ax1.plot(x, rmse3, lw=1, c='b', marker='D',  ms=4,label='RMSE$^h$(DWV)')  # 绘制y2
ax2.plot(x, rmsew1, lw=2,linestyle='--', c='r', marker='s', ms=4,label='RMSE$^w_w$(Clean)')  # 绘制y1
ax2.plot(x, rmsew2, lw=1,linestyle='--',c='g', marker='o',  ms=4,label='RMSE$^w_w$(RN)')  # 绘制y2
ax2.plot(x, rmsew3, lw=1, linestyle='--',c='b', marker='D',  ms=4,label='RMSE$^w_w$(HWV)')  # 绘制y2

# plt-style
# # plt.rcParams['front.sans-serif'] = ['SimHei']
# plt.rcParams['axes.prop_cycle'] = colors

plt.xticks(x)  # x轴的刻度
plt.gca().invert_xaxis()
ax1.set_ylim(2, 14)
ax2.set_ylim(30, 66)
ax1.set_xlabel('JPEG compression',fontsize='xx-large')  # x轴标注
ax1.set_ylabel('RMSE$^h$',fontsize='xx-large')  # y轴标注
ax2.set_ylabel('RMSE$^w_w$',fontsize='xx-large')  # y轴标注
fig.legend(loc=1,ncol=2,fontsize='large',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.subplots_adjust(left=0.1,right=0.88)
# plt.savefig('./figure3a.png')  # 保存图片
plt.show()




fig, ax1 = plt.subplots()

x = np.arange(0 ,2.25, 0.25)  # x坐标
print(x)
rmse1=[3.09,	3.13,	3.15,	3.08,	3.06,	3.06,	3.06,	3.06,	3.06]
rmse2=[3.24,	3.24,	3.75,	4.55,	4.84,	4.96,	5.02,	5.06,	5.08]
rmse3=[8.25,	8.40,	7.08,	4.91,	4.97,	5.11,	5.18,	5.22,	5.25]

rmsew1=[54.34,	54.35,	54.45,	54.36,	54.35,	54.32,	54.23,	54.21,	54.20]
rmsew2=[54.55,	54.45,	54.45,	54.45,	53.30,	54.29,	54.35,	54.18,	54.31]
rmsew3=[32.32,	28.88,	35.12,	49.33,	53.11,	53.34,	53.49,	53.91,	53.99]

ax2 = ax1.twinx()
ax1.plot(x, rmse1, lw=1, c='r', marker='s', ms=4,label='RMSE$^h$(Clean)')  # 绘制y1
ax1.plot(x, rmse2, lw=1, c='g', marker='o',  ms=4,label='RMSE$^h$(RN)')  # 绘制y2
ax1.plot(x, rmse3, lw=1, c='b', marker='D',  ms=4,label='RMSE$^h$(DWV)')  # 绘制y2
ax2.plot(x, rmsew1, lw=2,linestyle='--', c='r', marker='s', ms=4,label='RMSE$^w_w$(Clean)')  # 绘制y1
ax2.plot(x, rmsew2, lw=1,linestyle='--',c='g', marker='o',  ms=4,label='RMSE$^w_w$(RN)')  # 绘制y2
ax2.plot(x, rmsew3, lw=1, linestyle='--',c='b', marker='D',  ms=4,label='RMSE$^w_w$(HWV)')  # 绘制y2



# plt-style
plt.xticks(x)  # x轴的刻度
ax1.set_ylim(2, 14)
ax2.set_ylim(26, 68)
ax1.set_xlabel('Radius of Gaussian Blur',fontsize='xx-large')  # x轴标注
ax1.set_ylabel('RMSE$^h$',fontsize='xx-large')  # y轴标注
ax2.set_ylabel('RMSE$^w_w$',fontsize='xx-large')  # y轴标注


fig.legend(loc=1,ncol=2,fontsize='large',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
# plt.savefig('./figure3a.png')  # 保存图片]
plt.subplots_adjust(left=0.1,right=0.9)
plt.show()
