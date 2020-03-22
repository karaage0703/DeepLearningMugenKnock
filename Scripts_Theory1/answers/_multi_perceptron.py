import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[1], [-1], [-1], [1]], dtype=np.float32)

lr = 0.1

z1 = np.hstack([xs, [[1] for _ in range(4)]])

# perceptron
w1 = np.random.normal(0, 1, [3, 2])
w2 = np.random.normal(0, 1, [3, 1])
print("weight1 >>\n", w1)
print("weight2 >>\n", w2)

# add bias
#_x = np.hstack([x, [[1] for _ in range(4)]])

# train
ite = 0
for _ in range(5):
    ite += 1

    # feed forward
    z2 = np.dot(z1, w1)
    _z2 = np.hstack((z2, [[1] for _ in range(4)]))
    ys = np.dot(_z2, w2)

    print("iteration:", ite, "y >>\n", ys)

    # back propagation
    #if len(np.where((ys * ts) < 0)[0]) < 1:
    #    break

    _ts = ts.copy()
    _ts[ys * ts >= 0] = 0
    En = w2
    print("En", En)
    grad_w2 = np.dot(_z2.T, En)
    w2 -= lr * grad_w2

    #grad_w1 = np.dot(z1.T, np.dot(grad_w2, w2.T) * w1)
    
    grad_w1 = np.dot(En, w2.T)
    grad_w1 *= _z2
    grad_w1 = np.dot(xs.T, grad_w1)
    #w1 -= lr * grad_w1.T

    
print("training finished!")
print("weight1 >>\n", w1)
print("weight2 >>\n", w2)

# test
for i in range(4):
    z2 = np.dot(z1[i], w1)
    z2 = np.hstack((z2, [1]))
    y = np.dot(z2, w2)
    print("in >>", xs[i], ", out >>", y) 
    
