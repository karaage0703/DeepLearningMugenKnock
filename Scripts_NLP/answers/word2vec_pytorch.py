
import numpy as np
import argparse
from glob import glob
from copy import copy
import random
import pickle

# network
import torch
import torch.nn.functional as F
torch.manual_seed(0)

# GPU config
GPU = False
device = torch.device("cuda" if GPU else "cpu")

hidden_dim = 128
mb = 32
opt = "Adam" # SGD, Adam
C = 3  # word2vec window size satisfying C >= 1
x_length = 1 + C * 2  # training label length
TopN = 10 # display N similar word in test

# lr, iteration
train_factors = [[0.01, 1000]] 

import MeCab
mecab = MeCab.Tagger("-Owakati")


class Word2Vec(torch.nn.Module):
    def __init__(self, input_size, dim=512):
        super(Word2Vec, self).__init__()

        self.embed = torch.nn.Linear(input_size, dim)
        self.outs = []
        for _ in range(C * 2):
            self.outs.append(torch.nn.Linear(dim, input_size))
        self.out = torch.nn.Linear(dim, input_size)

    def forward(self, input):
        embed = self.embed(input)

        xs = []
        for i in range(C * 2):
            x = self.outs[i](embed)
            x = F.softmax(x, dim=1)
            xs.append(x)
        #x = self.out(embed)
        #x = F.softmax(x, dim=1)
        return xs

    def get_vec(self, input):
        return self.embed(input)


    
def data_load():
    sentences = []
    # get vocabrary
    _chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッー、。「」1234567890!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.@#"
    voca = ["<BRANK>"] + [c for c in _chars]

    # each file
    for file_path in glob("./sandwitchman_*.txt"):
        print("read:", file_path)
        with open(file_path, 'r') as f:
            # get line in 1 file
            lines = [x.strip() for x in f.read().strip().split("\n")]
        
            # get vocabrary from mecab parsed
            for line in lines:
                voca = list(set(voca) | set(mecab.parse(line).strip().split(" ")))

            # add sentences
            sentences += lines

    # vocabrary sort
    voca.sort()

    # display sentence number
    print("sentence pairs num:", len(sentences))
    
    sentence_index = []

    # each sentence
    for s in sentences:
        # mecab parse
        s_parse = mecab.parse(s).strip().split(" ")

        # add brank label first and end
        _s = ["<BRANK>"] * C + s_parse + ["<BRANK>"] * C

        # make training pairs
        for i in range(C, len(s_parse) + C):
            s_index = [voca.index(x) for x in _s[i-C : i+C+1]]
            sentence_index += [s_index]

    return voca, sentence_index


# train
def train():
    # data load
    voca, sentence_index = data_load()
    voca_num = len(voca)

    # write vocabrary lists
    pickle.dump(voca, open("vocabrary_word2vec.bn", "wb"))

    print("vocabrary num:", voca_num)
    print("e.x.", voca[:5])
    
    # model
    model = Word2Vec(voca_num, dim=hidden_dim).to(device)

    # minibatch index
    mbi = 0

    data_num = len(sentence_index)
    train_ind = np.arange(data_num)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    # loss function
    loss_fn = torch.nn.NLLLoss()
    
    # each learning rate and iteration
    for lr, ite in train_factors:
        print("lr", lr, " ite", ite)

        # optimizer
        if opt == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif opt == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise Exception("invalid optimizer:", opt)
        
        # each iteration
        for ite in range(ite):
            # get minibatch index
            if mbi + mb > data_num:
                mb_ind = copy(train_ind[mbi:])
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(data_num-mbi))]))
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            # get minibatch
            X_inds = [sentence_index[i] for i in mb_ind]

            loss = 0
            accuracy = 0.
            total_len = 0

            # each data of minibatch
            for mb_index in range(mb):
                # 1 data of minibatch
                Xs = np.array(X_inds[mb_index]).reshape([-1, 1])

                input_X = np.zeros([1, voca_num])
                input_X[:, Xs[C]] = 1
                input_X = torch.tensor(input_X, dtype=torch.float).to(device)
                
                # reset graph
                optimizer.zero_grad()
            
                # data length
                total_len += x_length

                # forward network
                ys = model(input_X)

                # target label index
                t_inds = [_i for _i in range(x_length) if _i != C]

                # each target label
                for i, y in zip(t_inds, ys):
                    # target label
                    t = torch.tensor(Xs[i], dtype=torch.long).to(device)

                    # get loss
                    loss += loss_fn(torch.log(y), t)

                    # count accuracy
                    if y.argmax() == t:
                        accuracy += 1

                """
                # each target label
                for i in range(x_length):
                    # forward network
                    y = model(input_X)

                    # target label
                    t = torch.tensor(Xs[i], dtype=torch.long).to(device)
                    #t = torch.tensor(Xs[i], dtype=torch.long).to(device).view(-1, voca_num)

                    # get loss
                    loss += loss_fn(torch.log(y), t)

                    # count accuracy
                    if y.argmax() == t:
                        accuracy += 1
                """

            # loss backward
            loss.backward()

            # update weight
            optimizer.step()
            
            # get loss
            loss = loss.item() / total_len
            accuracy = accuracy / total_len

            if (ite + 1) % 10 == 0:
                print("iter :", ite+1, ",loss >>:", loss, "accuracy:", accuracy)

    torch.save(model.state_dict(), 'word2vec.pt')
    

# test
def test(first_sentence="サンドウィッチマン"):
    # get vocabrary
    voca = pickle.load(open("vocabrary_word2vec.bn", "rb"))
    voca_num = len(voca)

    print("vocabrary num:", voca_num)
    
    # load trained model
    model = Word2Vec(voca_num, dim=hidden_dim).to(device)
    model.load_state_dict(torch.load('word2vec.pt'))

    xs = []

    # if word not found in vocabrary
    if first_sentence not in voca:
        raise Exception("not found word:", first_sentence)

    # get vector features of vocabrary
    mb = 32

    # feature vectors library
    features = np.ndarray([0, hidden_dim])

    with torch.no_grad():
        for i in range(0, voca_num, mb):
            # get minibatch
            _mb = min(mb, voca_num - i)

            # one hot vector
            input_X = torch.zeros([_mb, voca_num], dtype=torch.float).to(device)
            input_X[np.arange(_mb), np.arange(i, min(i + mb, voca_num))] = 1

            # get vector feature
            vecs = model.get_vec(input_X)
            vecs = vecs.detach().cpu().numpy()

            # add feature vectors
            features = np.vstack([features, vecs])

    print(features.shape)

    # make one hot input X
    input_X = torch.zeros([1, voca_num], dtype=torch.float).to(device)
    input_X[:, voca.index(first_sentence)] = 1

    # get target feature vector
    vec = model.get_vec(input_X)
    vec = vec.detach().cpu().numpy()[0]

    # get similarity
    #similarity_scores = np.sum(np.abs(features - vec) ** 2, axis=1)

    # get cosine similarity
    Norm_A = np.linalg.norm(features, axis=1)
    Norm_B = np.linalg.norm(vec)

    similarity_scores = np.dot(features, vec) / Norm_A / Norm_B

    # get min index,,   Skip first because it is target input word
    min_inds = similarity_scores.argsort()[::-1]

    print("Target:", first_sentence)

    # print
    for i in range(TopN):
        ind = min_inds[i]
        print("top{}: {} ({:.4f})".format(i + 1, voca[ind], similarity_scores[ind]))


    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--input', dest='input', default="サンドウィッチマン", type=str)
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test(args.input)

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
