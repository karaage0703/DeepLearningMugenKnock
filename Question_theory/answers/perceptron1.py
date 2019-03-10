import numpy as np

np.random.seed(0)

x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
t = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)

# perceptron
w = np.random.normal(0., 1, (3))
print("weight >>", w)

# add bias
_x = np.hstack([x, [[1] for _ in range(4)]])

for i in range(4):
    y = np.dot(w, _x[i])
    print("in >>", _x[i], "y >>", y) 
