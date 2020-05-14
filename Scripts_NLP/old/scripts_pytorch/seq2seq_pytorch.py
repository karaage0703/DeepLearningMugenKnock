
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
MAX_LENGTH = 100
teacher_forcing_ratio = 0.5
mb = 1
Attention = False
opt = "SGD" # SGD, Adam

# lr, iteration
train_factors = [[0.001, 100000]] 

next_word_mode = "prob" # prob, argmax

import MeCab
mecab = MeCab.Tagger("-Owakati")


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size):
        super(EncoderRNN, self).__init__()

        self.embedding = torch.nn.Embedding(input_size, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
        #self.gru2 = torch.nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        #output, 2 = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, hidden_dim, device=device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, output_size):
        super(DecoderRNN, self).__init__()

        self.embedding = torch.nn.Embedding(output_size, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
        #self.gru2 = torch.nn.GRU(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output, hidden = self.gru(output, hidden2)
        output = F.softmax(self.out(output[0]), dim=1)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, hidden_dim, device=device)


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self,  output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size, hidden_dim)
        self.attn = torch.nn.Linear(hidden_dim * 2, self.max_length)
        self.attn_combine = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Query
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Query + Key
        QK = torch.cat((embedded[0], hidden[0]), 1)

        # Query + Key -> Attention mask
        attn_weights = F.softmax(self.attn(QK), dim=1)

        # Attention mask x Value -> Attention
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Query + Attention
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        # GRU 
        output, hidden = self.gru(output, hidden)

        # Output (Class predict)
        output = F.softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, hidden_dim, device=device)
    

    
def data_load():
    sentence_pairs = []

    _chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッー、。「」1234567890!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.@#"

    voca = ["<BOS>", "<EOS>", "<FINISH>", "<UNKNOWN>"] + [c for c in _chars]

    for file_path in glob("./sandwitchman_*.txt"):
        print("read:", file_path)
        with open(file_path, 'r') as f:
            lines = [x.strip() for x in f.read().strip().split("\n")]
        
            for line in lines:
                voca = list(set(voca) | set(mecab.parse(line).strip().split(" ")))

            lines_before = lines
            lines_after = lines[1:] + ["<FINISH>"]
            sentence_pairs += [[s1, s2] for (s1, s2) in zip(lines_before, lines_after)]

    voca.sort()

    print("sentence pairs num:", len(sentence_pairs))
    
    sentence_pairs_index = []

    for s1, s2 in sentence_pairs:
        s1_parse = mecab.parse(s1).strip().split(" ")
        if s2 == "<FINISH>":
            s2_parse = ["<BOS>", s2, "<EOS>"]
        else:
            s2_parse = ["<BOS>"] + mecab.parse(s2).strip().split(" ") + ["<EOS>"]
        
        s1_index = [voca.index(x) for x in s1_parse]
        s2_index = [voca.index(x) for x in s2_parse]
        
        sentence_pairs_index += [[s1_index, s2_index]]


    #sentence_pairs_index = np.array(sentence_pairs_index)

    return voca, sentence_pairs_index


# train
def train():
    # data load
    voca, sentence_pairs = data_load()
    voca_num = len(voca)

    pickle.dump(voca, open("vocabrary.bn", "wb"))

    print("vocabrary num:", voca_num)
    print("e.x.", voca[:5])
    
    # model
    encoder = EncoderRNN(voca_num).to(device)
    if Attention:
        decoder = AttnDecoderRNN(voca_num, dropout_p=0.1).to(device)
    else:
        decoder = DecoderRNN(voca_num).to(device)
    

    #encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

    mbi = 0

    data_num = len(sentence_pairs)
    train_ind = np.arange(data_num)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for lr, ite in train_factors:
        print("lr", lr, " ite", ite)

        if opt == "SGD":
            encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=0.9)
            decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr, momentum=0.9)
        elif opt == "Adam":
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
            decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        else:
            raise Exception("invalid optimizer:", opt)
        
        
        for ite in range(ite):
            if mbi + mb > data_num:
                mb_ind = copy(train_ind[mbi:])
                np.random.shuffle(train_ind)
                mb_ind = np.hstack((mb_ind, train_ind[:(mb-(data_num-mbi))]))
            else:
                mb_ind = train_ind[mbi: mbi+mb]
                mbi += mb

            x_pairs = [sentence_pairs[i] for i in mb_ind]

            loss = 0
            accuracy = 0.
            total_len = 0

            for mb_index in range(mb):
                xs = torch.tensor(x_pairs[mb_index][0]).to(device).view(-1, 1)
                ts = torch.tensor(x_pairs[mb_index][1]).to(device).view(-1, 1)
            
                encoder_hidden = encoder.initHidden()
                #encoder_hidden2 = encoder.initHidden()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
            
                xs_length = xs.size()[0]
                ts_length = ts.size()[0]

                total_len += ts_length

                encoder_outputs = torch.zeros(MAX_LENGTH, hidden_dim).to(device)

                for ei in range(xs_length):
                    encoder_output, encoder_hidden = encoder(xs[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_xs = torch.tensor([[voca.index("<BOS>")]]).to(device)
            
                decoder_hidden = encoder_hidden
                
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                if use_teacher_forcing:
                    # Teacher forcing: Feed the target (ground-truth word) as the next input
                    for di in range(ts_length):
                        if Attention:
                            decoder_ys, decoder_hidden, decoder_attention = decoder(decoder_xs, decoder_hidden, encoder_outputs)
                        else:
                            decoder_ys, decoder_hidden = decoder(decoder_xs, decoder_hidden)
        
                        # add loss
                        loss += loss_fn(torch.log(decoder_ys), ts[di])

                        # count accuracy
                        if decoder_ys.argmax() == ts[di]:
                            accuracy += 1.
                            
                        # set next decoder's input (ground-truth label)
                        decoder_xs = ts[di]

                else:
                    # Without teacher forcing: use its own predictions as the next input
                    for di in range(ts_length):
                        if Attention:
                            decoder_ys, decoder_hidden, decoder_attention = decoder(decoder_xs, decoder_hidden, encoder_outputs)
                        else:
                            decoder_ys, decoder_hidden = decoder(decoder_xs, decoder_hidden)

                        # Select top 1 word with highest probability
                        #topv, topi = decoder_ys.topk(1)
                        # choice argmax
                        if next_word_mode == "argmax":
                            topv, topi = decoder_ys.data.topk(1)

                        elif next_word_mode == "prob":
                            topi = torch.multinomial(decoder_ys, 1)
                        
                        # set next input for decoder training
                        decoder_xs = topi.squeeze().detach()

                        # add loss
                        loss += loss_fn(torch.log(decoder_ys), ts[di])

                        # count accuracy
                        if decoder_ys.argmax() == ts[di]:
                            accuracy += 1.

                        if decoder_xs.item() == voca.index("<EOS>"):
                            break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss = loss.item() / ts_length
            accuracy = accuracy / total_len

            if (ite + 1) % 10 == 0:
                print("iter :", ite+1, ",loss >>:", loss, "accuracy:", accuracy)

    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
    

# test
def test(first_sentence="どうもーサンドウィッチマンです"):

    voca = pickle.load(open("vocabrary.bn", "rb"))
    voca_num = len(voca)
    
    # load trained model
    encoder = EncoderRNN(voca_num).to(device)
    if Attention:
        decoder = AttnDecoderRNN(voca_num, dropout_p=0.1).to(device)
    else:
        decoder = DecoderRNN(voca_num).to(device)
    
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))

    with torch.no_grad():
        xs = []
        for x in mecab.parse(first_sentence).strip().split(" "):
            if x in voca:
                xs += [voca.index(x)]
            else:
                xs += [voca.index("<UNKNOWN>")]

        xs = torch.tensor(xs, dtype=torch.long).to(device)

        count = 0

        print("A:", first_sentence)

        while count < 100:
            input_length = xs.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(MAX_LENGTH, hidden_dim).to(device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(xs[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[voca.index("<BOS>")]], device=device)  # SOS

            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

            for di in range(MAX_LENGTH):
                if Attention:
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

                # choice argmax
                if next_word_mode == "argmax":
                    topv, topi = decoder_output.data.topk(1)

                elif next_word_mode == "prob":
                    topi = torch.multinomial(decoder_output, 1)

                if topi.item() == voca.index("<EOS>"):
                    decoded_words.append('<EOS>')
                    break
                elif topi.item() == voca.index("<FINISH>"):
                    break
                else:
                    decoded_words.append(voca[topi.item()])

                decoder_input = topi.squeeze().detach()

            decoded_words = decoded_words[1:-1]

            xs = [voca.index(x) for x in decoded_words]  
            xs = torch.tensor(xs).to(device)

            sentence = "".join(decoded_words)

            if "<FINISH>" in sentence:
                break

            for key in ["<BOS>", "<EOS>", "<FINISH>", "<UNKNOWN>"]:
                sentence = sentence.replace(key, "")
            
            attention = decoder_attentions[:di + 1]

            
            
            if count % 2 == 0:
                print("B:", sentence)
            else:
                print("A:", sentence)

            count += 1
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--input', dest='input', default="ちょっと何言ってるのか分からない", type=str)
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
