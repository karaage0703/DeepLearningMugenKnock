import numpy as np

np.random.seed(0)

xs = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
ts = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

lrs = [0.1, 0.01]
linestyles = ['solid', 'dashed']
plts = []

for _i in range(len(lrs)):
    lr = lrs[_i]
    
    # perceptron
    np.random.seed(0)
    w = np.random.normal(0., 1, (3))
    print("weight >>", w)

    # add bias
    _xs = np.hstack([xs, [[1] for _ in range(4)]])

    # train
    ite = 0
    w1 = [w[0]]
    w2 = [w[1]]
    w3 = [w[2]]

    while True:
        ite += 1
        # feed forward
        ys = np.dot(_xs, w)

        print("iteration:", ite, "y >>", ys)

        # update parameters
        if len(np.where(ys * ts < 0)[0]) < 1:
            break

        _ys = ys.copy()
        _ts = ts.copy()
        _ys[ys * ts >= 0] = 0
        _ts[ys * ts >= 0] = 0
        En = np.dot(_ts, _xs)
        w += lr * En

        w1.append(w[0])
        w2.append(w[1])
        w3.append(w[2])

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
ys = np.array(list(map(lambda x: np.dot(w, x), _xs)))

for i in range(4):
    ys = np.dot(w, _xs[i])
    print("in >>", _xs[i], ", out >>", ys) 
    
