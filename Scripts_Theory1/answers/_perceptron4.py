import numpy as np

np.random.seed(0)

xs = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
ts = np.array(((0), (0), (0), (1)), dtype=np.float32)

lrs = [0.1, 0.01]
linestyles = ['solid', 'dashed']
plts = []

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

for _i in range(len(lrs)):
    lr = lrs[_i]
    
    # perceptron
    np.random.seed(0)
    w = np.random.normal(0., 1, (3))
    print("weight >>", w)

    # add bias
    z1 = np.hstack([xs, [[1] for _ in range(4)]])

    # train
    ite = 1
    w1 = [w[0]]
    w2 = [w[1]]
    w3 = [w[2]]

    for _ in range(1000):
        # feed forward
        ys = sigmoid(np.dot(z1, w))
        #ys = sigmoid(np.array(list(map(lambda x: np.dot(w, x), z1))))

        print("iteration:", ite, "y >>", ys)

        En = -2 * (ys - ts) * ys * (1 - ys)
        grad_w = np.dot(z1.T, En)
        w += lr * grad_w

        w1.append(w[0])
        w2.append(w[1])
        w3.append(w[2])

        ite += 1

    print("training finished!")
    print("weight >>", w)

    inds = list(range(ite))
    import matplotlib.pyplot as plt
    linestyle = linestyles[_i]
    plts.append(plt.plot(inds, w1, markeredgewidth=0, linestyle=linestyle)[0])
    plts.append(plt.plot(inds, w2, markeredgewidth=0, linestyle=linestyle)[0])
    plts.append(plt.plot(inds, w3, markeredgewidth=0, linestyle=linestyle)[0])

plt.legend(plts, ["w1:lr=0.1","w2:lr=0.1","w3:lr=0.1","w1:lr=0.01","w2:lr=0.01","w3:lr=0.01"], loc=1)
plt.savefig("answer_perceptron3.png")
plt.show()

# test
#ys = np.array(list(map(lambda x: np.dot(w, x), _xs)))

for i in range(4):
    ys = sigmoid(np.dot(w, np.hstack([xs[i], [1]])))
    print("in >>", xs[i], ", out >>", ys) 
    
