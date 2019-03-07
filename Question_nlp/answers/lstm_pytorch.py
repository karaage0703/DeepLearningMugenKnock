import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import os

GPU = False
torch.manual_seed(0)
n_gram = 10

_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。@#"
chars = [c for c in _chars]
num_classes = len(chars)

class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        base = 128
        self.h1 = torch.nn.LSTM(num_classes, base, batch_first=True)
        self.fc_out = torch.nn.Linear(base, num_classes)
        
    def forward(self, x):
        x, hn = self.h1(x)
        x = x[:, -1]
        x = self.fc_out(x)
        return x
    
def data_load():
    fname = 'sandwitchman.txt'
    xs = []
    ts = []
    txt = ''
    for _ in range(n_gram):
        txt += '@'
    onehots = []
    
    with open(fname, 'r') as f:
        for l in f.readlines():
            txt += l.strip() + '#'
        txt = txt[:-1] + '@'

        for c in txt:
            onehot = [0 for _ in range(num_classes)]
            onehot[chars.index(c)] = 1
            onehots.append(onehot)
        
        for i in range(len(txt) - n_gram - 1):
            xs.append(onehots[i:i+n_gram])
            ts.append(chars.index(txt[i+n_gram]))

    xs = np.array(xs)
    ts = np.array(ts)
    
    return xs, ts


# train
def train():
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Mynet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    xs, ts = data_load()
    print(xs.shape)

    # training
    mb = 128
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(500):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)
        y = F.log_softmax(y, dim=1)

        loss = torch.nn.CrossEntropyLoss()(y, t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Mynet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    def decode(x):
        return chars[x.argmax()]
    
    gens = ''

    for _ in range(n_gram):
        gens += '@'

    pred = 0
    count = 0
        
    while pred != '@' and count < 1000:
        in_txt = gens[-n_gram:]
        x = []
        for _in in in_txt:
            _x = [0 for _ in range(num_classes)]
            _x[chars.index(_in)] = 1
            x.append(_x)
        
        x = np.expand_dims(np.array(x), axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        pred = model(x)
        pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]

        # sample random from probability distribution
        ind = np.random.choice(num_classes, 1, p=pred)
        
        pred = chars[ind[0]]
        gens += pred
        count += 1

    # pose process
    gens = gens.replace('#', os.linesep).replace('@', '')
        
    print('--\ngenerated')
    print(gens)
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
