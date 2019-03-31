import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

lr = 0.1

# perceptron
w1 = np.random.normal(0, 1, [2, 2])
b1 = np.random.normal(0, 1, [2])
w2 = np.random.normal(0, 1, [2, 2])
b2 = np.random.normal(0, 1, [2])
wout = np.random.normal(0, 1, [2, 1])
bout = np.random.normal(0, 1, [1])
                             
print("weight1 >>\n", w1)
print("bias1 >>\n", b1)
print("weight_out >>\m", wout)
print("bias_out >>\n", bout)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z1 = xs

for ite in range(10000):
    ite += 1

    # feed forward
    z2 = sigmoid(np.dot(z1, w1) + b1)
    z3 = sigmoid(np.dot(z2, w2) + b2)
    out = sigmoid(np.dot(z3, wout) + bout)

    # back propagate
    En = (out - ts) * out * (1 - out)
    grad_wout = np.dot(z3.T, En)
    grad_bout = np.dot(np.ones([En.shape[0]]), En)
    wout -= lr * grad_wout
    bout -= lr * grad_bout
    
    # backpropagation inter layer
    grad_u2 = np.dot(En, wout.T) * z3 * (1 - z3)
    grad_w2 = np.dot(z2.T, grad_u2)
    grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2
        
    grad_u1 = np.dot(grad_u2, w2.T) * z2 * (1 - z2)
    grad_w1 = np.dot(z1.T, grad_u1)
    grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1


print("weight1 >>\n", w1)
print("bias1 >>\n", b1)
print("weight_out >>\n", wout)
print("bias_out >>\n", bout)

# test
for i in range(4):
    z2 = sigmoid(np.dot(z1[i], w1) + b1)
    z3 = sigmoid(np.dot(z2, w2) + b2)
    out = sigmoid(np.dot(z3, wout) + bout)
    print("in >>", xs[i], ", out >>", out)
    
