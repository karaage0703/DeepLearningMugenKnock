import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([0, 1, 1, 1], dtype=np.float32)

lr = 0.1

# perceptron
w = np.random.normal(0., 1, [2])
b = np.random.normal(0., 1, [1])
print("weight >>", w)
print("bias >>", b)

# add bias
z1 = xs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# train
for ite in range(5000):
    ite += 1
    # feed forward
    ys = sigmoid(np.dot(z1, w) + b)

    #print("iteration:", ite, "y >>", ys)

    En = -(ts - ys) * ys * (1 - ys)
    grad_w = np.dot(z1.T, En)
    grad_b = np.dot(np.ones([En.shape[0]]), En)
    w -= lr * grad_w
    b -= lr * grad_b
    
    
print("training finished!")
print("weight >>", w)
print("bias >>", b)

# test
for i in range(4):
    ys = sigmoid(np.dot(z1[i], w) + b)[0]
    print("in >>", xs[i], ", out >>", ys) 
    
