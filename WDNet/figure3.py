import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
# pre-datas
x = np.arange(0, 14, 1)  # x坐标
import matplotlib.ticker as ticker



figure, ax = plt.subplots()
rmse1=[2.217595476104247, 2.227418969047331, 2.248060732924996, 2.273524751332581, 2.3181095304681776, 2.3602371493224292, 2.377129921441287, 2.4213572615282795, 2.4891202062258198, 2.5418334239182387, 2.6166559239387692, 2.6588461960172727, 2.6975741567898015, 2.7452156757612527]
rmse2=[2.217595476104247, 2.2444513923025498, 3.0895219161826506, 5.7954801816140975, 5.563546871584503, 7.369464068101673, 8.793258257765459, 8.094980035937493, 9.006183619257886, 8.925907753479912, 9.17280936910853, 8.886321820347925, 9.236164089678754, 9.166022630666227]


plt.plot(x, rmse1, lw=2, c='red', marker='s', ms=5,label='Random')  # 绘制y1
plt.plot(x, rmse2, lw=2, c='b', marker='o',  ms=5,label='DWV')  # 绘制y2

# plt-style
plt.xticks(x)  # x轴的刻度
# plt.xlim(0.5, 10.5)  # x轴坐标范围
# plt.ylim(-500, 5800)  # y轴坐标范围
plt.xlabel('Perturbation Range $\epsilon$*255',fontsize='xx-large')  # x轴标注
plt.ylabel('RMSE$^h}$',fontsize='xx-large')  # y轴标注
plt.legend(loc='upper left',fontsize='x-large')  # 图例
tick_spacing = 4
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.savefig('./figure3d.png')  # 保存图片


figure, ax = plt.subplots()

rmse_w1=[65.2678660177683, 65.28989438233525, 65.30249099258876, 65.33068229952232, 65.34881482112857, 65.426199221795, 65.50661019370276, 65.49715495268012, 65.61526879694767, 65.79060392182708, 65.76526363651394, 65.70483400406845, 65.81357375184028, 66.07748366683421]
rmse_w2=[65.2678660177683, 65.49745954659453, 65.17794435871639, 63.57999127571612, 64.16846023474754, 45.5726815310014, 19.813510204259824, 42.061851310967896, 8.807927553134366, 0.03198563876295989, 23.086791642859065, 59.12701153674838, 41.327689076856394, 41.03498962747494]


plt.plot(x, rmse_w1, lw=2, c='red', marker='s', ms=5,label='Random')  # 绘制y1
plt.plot(x, rmse_w2, lw=2, c='b', marker='o',  ms=5,label='HWV')  # 绘制y2
# plt-style
plt.xticks(x)  # x轴的刻度
# plt.xlim(0.5, 10.5)  # x轴坐标范围
# plt.ylim(-500, 5800)  # y轴坐标范围
plt.xlabel('Perturbation Range $\epsilon$*255',fontsize='xx-large')  # x轴标注
plt.ylabel('RMSE$^w_w$',fontsize='xx-large')  # y轴标注
plt.legend(loc='upper left',fontsize='x-large')  # 图例
tick_spacing = 4
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.savefig('./figure3c.png')  # 保存图片


