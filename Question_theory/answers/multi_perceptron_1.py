import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

lr = 0.1

# perceptron
w1 = np.random.normal(0, 1, [2, 2])
b1 = np.random.normal(0, 1, [2])
wout = np.random.normal(0, 1, [2, 1])
bout = np.random.normal(0, 1, [1])
                             
print("weight1 >>\n", w1)
print("bias1 >>\n", b1)
print("weight_out >>\m", wout)
print("bias_out >>\n", bout)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z1 = xs


# test
for i in range(4):
    z2 = sigmoid(np.dot(z1[i], w1) + b1)
    out = sigmoid(np.dot(z2, wout) + bout)
    print("in >>", xs[i], ", out >>", out)
    
