import numpy as np
import cv2
import matplotlib.pyplot as plt

height, width = 64, 64

img = cv2.imread("akahara_0001.jpg")
img = cv2.resize(img, (width, height)).astype(np.float32)
img = img.transpose(2,0,1)
in_c = img.shape[0]

np.random.seed(0)

k_channel = 4
k_size = 3
kernels = np.random.normal(0, 0.01, [k_channel, k_size, k_size])

out = np.zeros((k_channel, height, width), dtype=np.float32)

pad = np.floor(k_size / 2).astype(np.int)
pad_img = np.zeros((in_c, pad * 2 + height, pad * 2 + width), np.float32)
pad_img[..., pad: pad+height, pad: pad+width] = img

for y in range(height):
    for x in range(width):
        for ki in range(k_channel):
            out[ki, y, x] = np.sum(pad_img[..., y:y+3, x:x+3] * kernels[ki])

for i in range(k_channel):
    plt.subplot(1,k_channel,i+1)
    plt.imshow(out[i], cmap='gray')

plt.show()
