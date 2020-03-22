import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)

# perceptron
w = np.random.normal(0., 1, (3))
print("weight >>", w)

# add bias
_xs = np.hstack([xs, [[1] for _ in range(4)]])

for i in range(4):
    ys = np.dot(w, _xs[i])
    print("in >>", _xs[i], "y >>", ys) 
