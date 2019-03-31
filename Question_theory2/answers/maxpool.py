import numpy as np
import cv2
import matplotlib.pyplot as plt

height, width = 64, 64

img = cv2.imread("akahara_0001.jpg")
img = cv2.resize(img, (width, height)).astype(np.float32)

np.random.seed(0)

in_c = img.shape[-1]
k_size = 2
stride = 2
pad = 0

pad_img = np.zeros((pad * 2 + height, pad * 2 + width, in_c), np.float32)
pad_img[pad: pad+height, pad: pad+width] = img

out_height = (height + pad) // stride
out_width = (width + pad) // stride

out = np.zeros((out_height, out_width, in_c), dtype=np.float32)

for y in range(out_height):
    for x in range(out_width):
        for c in range(in_c):
            out[y, x, c] = np.max(pad_img[y * stride: y * stride + k_size,
                                           x * stride: x * stride + k_size, c])

for i in range(in_c):
    plt.subplot(1,in_c,i+1)
    plt.imshow(out[..., i], cmap='gray')

plt.show()
