import numpy as np

np.random.seed(0)

class NN:
    def __init__(self, ind=2, w=64, outd=1, lr=0.1):
        self.w2 = np.random.randn(ind, w)
        self.b2 = np.random.randn(w)
        self.wout = np.random.randn(w, outd)
        self.bout = np.random.randn(outd)
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = self.sigmoid(np.dot(self.z1, self.w2) + self.b2)
        self.out = self.sigmoid(np.dot(self.z2, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        out_d = 2*(self.out - t) * self.out * (1 - self.out)
        out_dW = np.dot(self.z2.T, out_d)
        out_dB = np.dot(np.ones([1, out_d.shape[0]]), out_d)
        self.wout -= self.lr * out_dW
        self.bout -= self.lr * out_dB[0]

        # backpropagation inter layer
        w2_d = np.dot(out_d, self.wout.T) * self.z2 * (1 - self.z2)
        w2_dW = np.dot(self.z1.T, w2_d)
        w2_dB = np.dot(np.ones([1, w2_d.shape[0]]), w2_d)
        self.w2 -= self.lr * w2_dW
        self.b2 -= self.lr * w2_dB[0]

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
train_t = np.array([[1], [0], [0], [1]], dtype=np.float32)

nn = NN(ind=train_x.shape[1])

# train
for i in range(1000):
    nn.forward(train_x)
    nn.train(train_x, train_t)

# test
for j in range(4):
    x = train_x[j]
    t = train_t[j]
    print("in:", x, "pred:", nn.forward(x))
