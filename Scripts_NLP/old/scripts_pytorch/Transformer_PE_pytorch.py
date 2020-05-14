import numpy as np
import argparse
from glob import glob
from copy import copy
import random
import pickle
import sys

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
Attention = True
opt = "SGD" # SGD, Adam

# lr, iteration
train_factors = [[0.001, 2000]] 

next_word_mode = "prob" # prob, argmax

import MeCab
mecab = MeCab.Tagger("-Owakati")

use_Bidirectional = False # Bi-directional
dropout_p = 0.2 # Dropout ratio
num_layers = 1

Encoder_attention_time = 1  # Transformer technique 3 : Hopping if > 1
Decoder_attention_time = 1  # Transformer technique 3 : Hopping if > 1
use_Source_Target_Attention = True # use source target attention
use_Encoder_Self_Attention = True # self attention of Encoder
use_Decoder_Self_Attention = True # self attention of Decoder
MultiHead_Attention_N = 8 # Multi head attention Transformer technique 1
use_FeedForwardNetwork = True # Transformer technique 4
use_PositionalEncoding = True # Transformer technique 5


# automatically get RNN hidden dimension from above config
RNN_dim = hidden_dim * 2 if use_Bidirectional else hidden_dim
tensor_dim = num_layers * 2 if use_Bidirectional else num_layers


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, max_length=MAX_LENGTH, 
        dropout_p=0.1, num_layers=1,
        attention_time=1,
        use_Source_Target_Attention=False,
        use_Self_Attention=False,
        MultiHead_Attention_N=2,
        use_FFNetwork=False,
        use_PositionalEncoding=False):
    
        super(Encoder, self).__init__()
        self.max_length = max_length

        # Embedding
        self.embedding = torch.nn.Embedding(input_size, hidden_dim)

        # Positional Encoding
        if use_PositionalEncoding:
            self.positionalEncoding = PositionalEncoding()

        # Attention
        self.attentions = []
        if attention_time > 0:
            _attentions = []
            for i in range(attention_time):
                # step2 : Self Attention
                if use_Self_Attention:
                    _attentions.append(Attention(
                        hidden_dim=hidden_dim, 
                        memory_dim=hidden_dim, 
                        dropout_p=dropout_p, 
                        max_length=max_length, 
                        #self_Attention_Decoder=True, 
                        head_N=MultiHead_Attention_N
                        ))

                # Feed Forward Network
                if use_FFNetwork:
                    _attentions.append(FeedForwardNetwork(
                        hidden_dim=hidden_dim, 
                        dropout_p=dropout_p))

            self.attentions = _attentions

        # output GRU
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=use_Bidirectional)


    def forward(self, x, hidden, memory):
        # Embedding
        x = self.embedding(x).view(1, 1, -1)

        # Memory embedding
        memory = self.embedding(memory).permute(1, 0, 2)
        memory = memory.float()

        # Positional Encoding
        if hasattr(self, "positionalEncoding"):#self.use_PositionalEncoding:
            x = self.positionalEncoding(x)
            memory = self.positionalEncoding(memory)

        # Attention
        for layer in self.attentions:
            x = layer(x, memory, memory)

        # output GRU
        x, hidden = self.gru(x, hidden)
        return x, hidden

    def initHidden(self):
        return torch.zeros(tensor_dim, 1, hidden_dim, device=device)


class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, RNN_dim, dropout_p=0.1, num_layers=1,
        attention_time=1,
        max_length=MAX_LENGTH,
        use_Source_Target_Attention=False,
        use_Self_Attention=False,
        MultiHead_Attention_N=2,
        use_FFNetwork=False,
        use_PositionalEncoding=False):

        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Embedding
        self.input_embedding = torch.nn.Embedding(output_dim, hidden_dim)
        self.input_embedding_dropout = torch.nn.Dropout(dropout_p)

        # Positional Encoding
        if use_PositionalEncoding:
            self.positionalEncoding = PositionalEncoding()

        # step1 : Attention
        self.attentions = []
        if attention_time > 0:
            _attentions = []
            for i in range(attention_time):
                # step2 : Self Attention
                if use_Self_Attention:
                    _attentions.append(Attention(
                        hidden_dim=hidden_dim, 
                        memory_dim=hidden_dim, 
                        dropout_p=dropout_p, 
                        max_length=max_length, 
                        self_Attention_Decoder=True,
                        head_N=MultiHead_Attention_N
                        ))
                
                # step1 : Source Target Attention
                if use_Source_Target_Attention:
                    _attentions.append(Attention(
                        hidden_dim=hidden_dim, 
                        memory_dim=RNN_dim, 
                        dropout_p=dropout_p, 
                        max_length=max_length,
                        head_N=MultiHead_Attention_N
                        ))

                # Feed Forward Network
                if use_FFNetwork:
                    _attentions.append(FeedForwardNetwork(
                        hidden_dim=hidden_dim, 
                        dropout_p=dropout_p))
        
            self.attentions = _attentions

        # output GRU
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=use_Bidirectional)
        self.out = torch.nn.Linear(RNN_dim, output_dim)
    

    def forward(self, x, hidden, memory_encoder, memory_decoder):
        # Embedding
        x = self.input_embedding(x)
        x = self.input_embedding_dropout(x)

        # Memory Embedding
        memory_decoder = self.input_embedding(memory_decoder).permute(1, 0, 2)

        # Positional Encoding
        if hasattr(self, "positionalEncoding"):
            x = self.positionalEncoding(x)
            memory_decoder = self.positionalEncoding(memory_decoder)

        # Attention
        for layer in self.attentions:
            x = layer(x, memory_encoder, memory_decoder)

        # output GRU
        x, hidden = self.gru(x, hidden)
        x = self.out(x[0])
        x = F.softmax(x, dim=-1)
        return x, hidden, None



class Attention(torch.nn.Module):
    def __init__(self, hidden_dim, memory_dim, dropout_p=0.1, max_length=MAX_LENGTH, head_N=1, self_Attention_Decoder=False):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.head_N = head_N
        self.self_Attention_Decoder = self_Attention_Decoder

        # Attention Query
        #self.Q_embedding = torch.nn.Embedding(self.output_size, hidden_dim)
        #self.Q_dropout = torch.nn.Dropout(self.dropout_p)
        self.Q_dense = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Q_dense_dropout = torch.nn.Dropout(dropout_p)
        #self.Q_BN = torch.nn.BatchNorm1d(hidden_dim)
        
        # Attention Key
        self.K_dense = torch.nn.Linear(memory_dim, hidden_dim)
        self.K_dense_dropout = torch.nn.Dropout(dropout_p)
        #self.K_BN = torch.nn.BatchNorm1d(hidden_dim)
        
        # Attetion Value
        self.V_dense = torch.nn.Linear(memory_dim, hidden_dim)
        self.V_dense_dropout = torch.nn.Dropout(dropout_p)
        #self.V_BN = torch.nn.BatchNorm1d(hidden_dim)
        
        # Attention mask
        #self.attention = torch.nn.Linear(hidden_dim * 2, self.max_length)
        self.attention_dense = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_dropout = torch.nn.Dropout(dropout_p)
        #self.attention_BN = torch.nn.BatchNorm1d(hidden_dim)


    def forward(self, _input, memory, memory2):
        # get Query
        Q = self.Q_dense(_input.view(1, -1))
        #Q = self.Q_BN(Q)
        Q = self.Q_dense_dropout(Q)
        Q = Q.view(1, 1, -1)
        
        # one head -> Multi head
        Q = Q.view(1, self.hidden_dim // self.head_N, self.head_N)
        Q = Q.permute([2, 0, 1])

        # Transformer technique 1 : scaled dot product
        Q *= Q.size()[-1] ** -0.5


        if self.self_Attention_Decoder:
            memory = memory2

        # memory transforme [mb(=1), length, dim] -> [length, dim]
        if len(memory.size()) > 2:
            memory = memory[0]
        
        # get Key
        K = self.K_dense(memory)
        #K = self.K_BN(K)
        K = self.K_dense_dropout(K)
        K = K.view(1, -1, self.hidden_dim)


        # one head -> Multi head
        K = K.view(-1, self.hidden_dim // self.head_N, self.head_N)
        K = K.permute([2, 1, 0])

        # get Query and Key (= attention logits)
        QK = torch.bmm(Q, K)


        # Transformer technique 2 : masking attention weight
        any_zero = memory.sum(dim=1)
        pad_mask = torch.ones([1, 1, self.max_length]).to(device)
        pad_mask[:, :, torch.nonzero(any_zero)] = 0

        _, _, QK_length = QK.size()
        pad_mask = pad_mask[:, :, :QK_length]


        QK += pad_mask * sys.float_info.min
        
        # get attention weight
        attention_weights = F.softmax(QK, dim=-1)
        
        # get Value
        V = self.V_dense(memory)
        #V = self.V_BN(V)
        V = self.V_dense_dropout(V)
        V = V.view(1, -1, self.hidden_dim)

        # one head -> Multi head
        V = V.view(-1, self.hidden_dim // self.head_N, self.head_N)
        V = V.permute(2, 0, 1)
        
        # Attetion x Value
        attention_feature = torch.bmm(attention_weights, V)

        # Multi head -> one head
        attention_feature = attention_feature.permute(1, 2, 0)
        attention_feature = attention_feature.contiguous().view(1, 1, -1)
        
        # attention + Input
        attention_x = torch.cat([_input, attention_feature], dim=-1)
        
        # apply attention dense
        attention_output = self.attention_dense(attention_x)
        #attention_output = self.attention_BN(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = F.relu(attention_output)

        return attention_output


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, memory_encoder, decoder):
        x = self.module(x)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        mb, sequence_length, dimension = x.size()
        positionalEncodingFeature = np.zeros([mb, sequence_length, dimension], dtype=np.float32)

        position_index = np.arange(sequence_length).repeat(dimension).reshape(-1, dimension)
        dimension_index = np.tile(np.arange(dimension), [sequence_length, 1])

        positionalEncodingFeature[:, :, 0::2] = np.sin(position_index[:, 0::2] / (10000 ** (2 * dimension_index[:, 0::2] / dimension)))
        positionalEncodingFeature[:, :, 1::2] = np.cos(position_index[:, 1::2] / (10000 ** (2 * dimension_index[:, 1::2] / dimension)))

        positionalEncodingFeature = torch.tensor(positionalEncodingFeature).to(device)

        x += positionalEncodingFeature

        return x


    
def data_load():
    sentence_pairs = []

    _chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョッー、。「」1234567890!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.@#"

    voca = ["<BOS>", "<EOS>", "<FINISH>", "<UNKNOWN>"] + [c for c in _chars]

    for file_path in glob("./sandwitchman*.txt"):
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
    encoder = Encoder(
        voca_num, 
        hidden_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        attention_time=Encoder_attention_time,
        use_Source_Target_Attention=use_Source_Target_Attention,
        use_Self_Attention=use_Encoder_Self_Attention,
        MultiHead_Attention_N=MultiHead_Attention_N,
        use_FFNetwork=use_FeedForwardNetwork,
        use_PositionalEncoding=use_PositionalEncoding
        ).to(device) 

    decoder = Decoder(
        hidden_dim,
        voca_num, 
        RNN_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        attention_time=Decoder_attention_time, 
        use_Source_Target_Attention=use_Source_Target_Attention,
        use_Self_Attention=use_Encoder_Self_Attention,
        MultiHead_Attention_N=MultiHead_Attention_N,
        use_FFNetwork=use_FeedForwardNetwork,
        use_PositionalEncoding=use_PositionalEncoding
        ).to(device)

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
                xs_float = torch.tensor(x_pairs[mb_index][0], dtype=torch.float).to(device).view(-1, 1)
                ts = torch.tensor(x_pairs[mb_index][1]).to(device).view(-1, 1)
            
                encoder_hidden = encoder.initHidden()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
            
                xs_length = xs.size()[0]
                ts_length = ts.size()[0]

                total_len += ts_length

                encoder_outputs = torch.zeros(MAX_LENGTH, RNN_dim).to(device)

                for ei in range(xs_length):
                    encoder_output, encoder_hidden = encoder(xs[ei], encoder_hidden, xs)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_xs = torch.tensor([[voca.index("<BOS>")]]).to(device)
            
                decoder_hidden = encoder_hidden
                
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                self_memory = decoder_xs

                if use_teacher_forcing:
                    # Teacher forcing: Feed the target (ground-truth word) as the next input
                    for di in range(ts_length):
                        decoder_ys, decoder_hidden, decoder_attention = decoder(decoder_xs, decoder_hidden, encoder_outputs, self_memory)

        
                        # add loss
                        loss += loss_fn(torch.log(decoder_ys), ts[di])

                        # count accuracy
                        if decoder_ys.argmax() == ts[di]:
                            accuracy += 1.
                            
                        # set next decoder's input (ground-truth label)
                        decoder_xs = ts[di].view(1, -1)
                        #self_memory[di] = decoder_xs.clone().detach()[0]
                        self_memory = torch.cat([self_memory, decoder_xs])

                else:
                    # Without teacher forcing: use its own predictions as the next input
                    for di in range(ts_length):
                        decoder_ys, decoder_hidden, decoder_attention = decoder(decoder_xs, decoder_hidden, encoder_outputs, self_memory)
                        
                        # Select top 1 word with highest probability
                        #topv, topi = decoder_ys.topk(1)
                        # choice argmax
                        if next_word_mode == "argmax":
                            topv, topi = decoder_ys.data.topk(1)

                        elif next_word_mode == "prob":
                            topi = torch.multinomial(decoder_ys, 1)
                        
                        # set next input for decoder training
                        decoder_xs = topi.squeeze().detach().view(1, -1)

                        # add loss
                        loss += loss_fn(torch.log(decoder_ys), ts[di])

                        # count accuracy
                        if decoder_ys.argmax() == ts[di]:
                            accuracy += 1.

                        if decoder_xs.item() == voca.index("<EOS>"):
                            break

                        self_memory = torch.cat([self_memory, decoder_xs])

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
    encoder = Encoder(
        voca_num, 
        hidden_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        attention_time=Encoder_attention_time,
        use_Source_Target_Attention=use_Source_Target_Attention,
        use_Self_Attention=use_Encoder_Self_Attention,
        MultiHead_Attention_N=MultiHead_Attention_N,
        use_FFNetwork=use_FeedForwardNetwork,
        use_PositionalEncoding=use_PositionalEncoding
        ).to(device) 

    decoder = Decoder(
        hidden_dim,
        voca_num, 
        RNN_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        attention_time=Decoder_attention_time, 
        use_Source_Target_Attention=use_Source_Target_Attention,
        use_Self_Attention=use_Encoder_Self_Attention,
        MultiHead_Attention_N=MultiHead_Attention_N,
        use_FFNetwork=use_FeedForwardNetwork,
        use_PositionalEncoding=use_PositionalEncoding
        ).to(device)

    
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))

    with torch.no_grad():
        xs = []
        for x in mecab.parse(first_sentence).strip().split(" "):
            if x in voca:
                xs += [voca.index(x)]
            else:
                xs += [voca.index("<UNKNOWN>")]

        xs = torch.tensor(xs, dtype=torch.long).to(device).view(-1, 1)

        count = 0

        print("A:", first_sentence)

        while count < 100:
            input_length = xs.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(MAX_LENGTH, RNN_dim).to(device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(xs[ei], encoder_hidden, xs)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_x = torch.tensor([[voca.index("<BOS>")]], device=device)  # SOS

            decoder_hidden = encoder_hidden
            decoded_words = []

            self_memory = decoder_x

            for di in range(MAX_LENGTH):
                decoder_ys, decoder_hidden, decoder_attention = decoder(decoder_x, decoder_hidden, encoder_outputs, self_memory)
        
                # choice argmax
                if next_word_mode == "argmax":
                    topv, topi = decoder_ys.data.topk(1)

                elif next_word_mode == "prob":
                    topi = torch.multinomial(decoder_ys, 1)

                if topi.item() == voca.index("<EOS>"):
                    decoded_words.append('<EOS>')
                    break
                elif topi.item() == voca.index("<FINISH>"):
                    break
                else:
                    decoded_words.append(voca[topi.item()])

                decoder_x = topi.squeeze().detach().view(1, -1)

                self_memory = torch.cat([self_memory, decoder_x])

            decoded_words = decoded_words[1:-1]

            xs = [voca.index(x) for x in decoded_words]  
            xs = torch.tensor(xs).to(device).view(-1, 1)

            sentence = "".join(decoded_words)

            if "<FINISH>" in sentence:
                break

            for key in ["<BOS>", "<EOS>", "<FINISH>", "<UNKNOWN>"]:
                sentence = sentence.replace(key, "")
            
            
            
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
