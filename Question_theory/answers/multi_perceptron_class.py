import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[0], [1], [1], [0]], dtype=np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FullyConnectedLayer():
    def __init__(self, in_n, out_n, use_bias=True, activation=None):
        self.w = np.random.normal(0, 1, [in_n, out_n])
        if use_bias:
            self.b = np.random.normal(0, 1, [out_n])
        else:
            self.b = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None

    def set_lr(self, lr=0.1):
        self.lr = lr

    def forward(self, feature_in):
        self.x_in = feature_in
        x = np.dot(feature_in, self.w)
        
        if self.b is not None:
            x += self.b
            
        if self.activation is not None:
            x = self.activation(x)
        self.x_out = x
        
        return x

    
    def backward(self, w_pro, grad_pro):
        grad = np.dot(grad_pro, w_pro.T)
        if self.activation is sigmoid:
            grad *= (self.x_out * (1 - self.x_out))
        grad_w = np.dot(self.x_in.T, grad)
        self.w -= self.lr * grad_w

        if self.b is not None:
            grad_b = np.dot(np.ones([grad.shape[0]]), grad)
            self.b -= self.lr * grad_b

        return grad

    
class Model():
    def __init__(self, *args, lr=0.1):
        self.layers = args
        for l in self.layers:
            l.set_lr(lr=lr)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        
        return x

    def backward(self, t):
        En = (self.output - t) * self.output * (1 - self.output)
        grad_pro = En
        w_pro = np.eye(En.shape[-1])
        
        for i, layer in enumerate(self.layers[::-1]):
            grad_pro = layer.backward(w_pro=w_pro, grad_pro=grad_pro)
            w_pro = layer.w
            


model = Model(FullyConnectedLayer(in_n=2, out_n=64, activation=sigmoid),
              FullyConnectedLayer(in_n=64, out_n=32, activation=sigmoid),
              FullyConnectedLayer(in_n=32, out_n=1, activation=sigmoid), lr=0.1)


for ite in range(10000):
    ite += 1

    model.forward(xs)
    model.backward(ts)


# test
for i in range(4):
    out = model.forward(xs[i])
    print("in >>", xs[i], ", out >>", out)
    
