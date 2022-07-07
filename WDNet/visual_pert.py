import numpy as np
import cv2



delta = np.load('./delta.npy')
delta = delta*10
delta = delta - delta.min()
scale = 128/delta.mean()

delta = np.clip(delta*scale,0,255)
delta = delta[0].transpose(1,2,0)
cv2.imwrite('delta.jpg',delta)










