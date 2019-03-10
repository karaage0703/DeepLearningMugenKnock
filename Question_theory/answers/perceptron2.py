import numpy as np

np.random.seed(0)

x = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
t = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

lr = 0.1

# perceptron
w = np.random.normal(0., 1, (3))

# add bias
_x = np.hstack([x, [[1] for _ in range(4)]])

# train
ite = 0
while True:
    ite += 1
    y = np.array(list(map(lambda x: np.dot(w, x), _x)))

    En = np.array([0 for _ in range(3)], dtype=np.float32)

    for i in range(4):
        if y[i] * t[i] < 0:
            En += t[i] * _x[i]

    print("iteration:", ite, "y >>", y)
    if np.any(En != 0):
        w += lr * En
    else:
        break
print("training finished!")

# test
y = np.array(list(map(lambda x: np.dot(w, x), _x)))

for i in range(4):
    y = np.dot(w, _x[i])
    print("in >>", _x[i], ", out >>", y) 
    
