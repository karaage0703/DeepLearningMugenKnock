import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([0, 0, 0, 1], dtype=np.float32)

lr = 0.1

# perceptron
w = np.random.normal(0., 1, (3))
print("weight >>", w)

# add bias
z1 = np.hstack([xs, [[1] for _ in range(4)]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# train
for ite in range(5000):
    ite += 1
    # feed forward
    ys = sigmoid(np.dot(z1, w))

    #print("iteration:", ite, "y >>", ys)

    En = -(ts - ys) * ys * (1 - ys)
    grad_w = np.dot(z1.T, En)
    w -= lr * grad_w
    
    
print("training finished!")
print("weight >>", w)

# test

for i in range(4):
    ys = sigmoid(np.dot(z1[i], w))
    print("in >>", xs[i], ", out >>", ys) 
    
