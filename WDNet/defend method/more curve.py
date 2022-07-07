import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
# pre-datas
x = np.arange(100, -20, -20)  # x坐标

print(x)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
rmse1= [3.0759,	4.0799,	5.1359,	6.2959,	7.0018	,9.2604]
rmse2 = [7.8238	,5.5491,	5.9334	,6.714	,7.1665,	9.2602]
rmse3 =[4.8522	,4.5196	,5.4411	,6.4668	,7.0634	,9.2636]

rmsew1= [15.6904	,15.6414,	16.0831	,16.5293,	17.4798	,27.8649]
rmsew2 = [20.5512	,15.8357	,16.0838	,16.8354	,17.4166	,27.7743]
rmsew3 =[28.9819,	20.2409	,17.7023,	16.8522,	17.549,	27.9876]

ax1.plot(x, rmse1, lw=1, c='red', marker='s', ms=5,label='RMSE Clean')  # 绘制y1
ax1.plot(x, rmse2, lw=1, c='g', marker='o',  ms=5,label='RMSE DWV')  # 绘制y2
ax1.plot(x, rmse3, lw=1, c='b', marker='^',  ms=5,label='RMSE HWV')  # 绘制y2
ax2.plot(x, rmsew1, lw=1,linestyle='-.', c='red', marker='s', ms=5,label='RMSE$_w$ Clean')  # 绘制y1
ax2.plot(x, rmsew2, lw=1,linestyle='-.',c='g', marker='o',  ms=5,label='RMSE$_w$ DWV')  # 绘制y2
ax2.plot(x, rmsew3, lw=1, linestyle='-.',c='b', marker='^',  ms=5,label='RMSE$_w$ HWV')  # 绘制y2

# plt-style
plt.xticks(x)  # x轴的刻度
plt.gca().invert_xaxis()
ax1.set_ylim(0, 13)
ax2.set_ylim(14, 36)
ax1.set_xlabel('JPEG compression',fontsize='xx-large')  # x轴标注
ax1.set_ylabel('RMSE',fontsize='xx-large')  # y轴标注
ax2.set_ylabel('RMSE$_w$',fontsize='xx-large')  # y轴标注
fig.legend(loc=1,ncol=2,fontsize='large',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.subplots_adjust(left=0.1,right=0.88)
# plt.savefig('./figure3a.png')  # 保存图片
plt.show()




fig, ax1 = plt.subplots()

x = np.arange(0 ,2.25, 0.25)  # x坐标
print(x)
rmse1= [3.0759,	3.1448	,4.6735	,5.7198	,5.9891	,6.0982	,6.1518	,6.1791	,6.1988]
rmse2 = [7.8238	,8.5262,	7.8063	,6.3113	,6.3513,	6.4237	,6.4584,	6.4817	,6.4956]
rmse3 =[4.8522	,4.3278	,5.1355	,5.88	,6.1332,	6.2226,	6.2628,	6.3001,	6.3166]

rmsew1= [15.6904	,15.6907	,16.2314	,16.7978	,17.0671,	17.2541	,17.3488,	17.4066,	17.445]
rmsew2 = [20.5512,	26.0553	,20.4911	,17.389	,17.5057,	17.5248,	17.6337,	17.6246	,17.6578]
rmsew3 =[28.9819	,31.3009	,27.2887	,20.8037	,18.81	,17.9435	,18.0665	,17.994	,18.0564]

ax2 = ax1.twinx()
ax1.plot(x, rmse1, lw=1, c='red', marker='s', ms=5,label='RMSE Clean')  # 绘制y1
ax1.plot(x, rmse2, lw=1, c='g', marker='o',  ms=5,label='RMSE DWV')  # 绘制y2
ax1.plot(x, rmse3, lw=1, c='b', marker='^',  ms=5,label='RMSE HWV')  # 绘制y2
ax2.plot(x, rmsew1, lw=1,linestyle='-.', c='red', marker='s', ms=5,label='RMSE$_w$ Clean')  # 绘制y1
ax2.plot(x, rmsew2, lw=1,linestyle='-.',c='g', marker='o',  ms=5,label='RMSE$_w$ DWV')  # 绘制y2
ax2.plot(x, rmsew3, lw=1, linestyle='-.',c='b', marker='^',  ms=5,label='RMSE$_w$ HWV')  # 绘制y2

# plt-style
plt.xticks(x)  # x轴的刻度
ax1.set_xlabel('Radius of Gaussian Blur',fontsize='xx-large')  # x轴标注
ax1.set_ylabel('RMSE',fontsize='xx-large')  # y轴标注
ax2.set_ylabel('RMSE$_w$',fontsize='xx-large')  # y轴标注


fig.legend(loc=1,ncol=2,fontsize='large',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
# plt.savefig('./figure3a.png')  # 保存图片]
plt.subplots_adjust(left=0.1,right=0.9)
plt.show()
