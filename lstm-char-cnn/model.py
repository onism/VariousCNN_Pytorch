import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# CNN

class CharCNN(nn.Module):
    '''
        Conv layer with multiple filters of different widths
    '''
    def __init__(self, embedding_dim, num_filters=16):
        super(CharCNN, self).__init__()
        kernels = [2,3,4,5]
        self.word_dim = sum(kernel * num_filters for kernel in kernels) # concanate all CNN
        self.layers = nn.ModuleList([
            nn.Conv2d(1, num_filters*kernel, (kernel, embedding_dim), stride=(1,1)) for kernel in kernels
        ])
    
    def forward(self, x):
        '''
            Input: batch * temporal * max_length * embedding_dim
            output: batch * temporal * word_dim
        '''
        x = [F.relu(layer(x)).squeeze() for layer in self.layers]
        # batch * (num_filters*kernel) * width * 1
        x = [F.max_pool1d(kernel, kernel.size()[-1]) for kernel in x]
        x = torch.cat(x, dim=1)
        return x

class HighwayNetwork(nn.Module):
    '''
     Eq 8:  z =t \odot g (W_{H}y +b_{H})+(1-t) \odot y
            t = \sigma(W_t y + b_T)
    '''
    def __init__(self, word_dim):
        super(HighwayNetwork, self).__init__()
        self.Wh = nn.Linear(word_dim, word_dim, bias=True)
        self.Wt = nn.Linear(word_dim, word_dim, bias=True)
    
    def forward(self, x):
        transform_gate = torch.sigmoid(self.Wt(x))
        carry_gate = 1 - transform_gate
        return transform_gate * F.relu(self.Wh(x)) + carry_gate * x


class CharCNNRNN(nn.Module):

    def __init__(self, embedding_dim, vocab_size, word_vocab_size, padding_idx=0):
        super(CharCNNRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.CharCNN = CharCNN(embedding_dim)
        self.word_dim = self.CharCNN.word_dim
        self.HighwayNetwork = HighwayNetwork(self.word_dim)
        self.hidden_size = 128
        self.no_rnn_layers = 1
        self.bidirectional = True
        self.rnn = nn.LSTM(self.word_dim, hidden_size=self.hidden_size, num_layers=self.no_rnn_layers, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size*2, word_vocab_size)
        else:
            self.fc = nn.Linear(self.hidden_size, word_vocab_size)
        self.dropput = nn.Dropout(0.2)
        self.out = nn.LogSoftmax(dim=-1)
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros([self.no_rnn_layers*2, batch_size, self.hidden_size]), torch.zeros([self.no_rnn_layers*2, batch_size, self.hidden_size]) )
        else:
            return (torch.zeros([self.no_rnn_layers, batch_size, self.hidden_size]), torch.zeros([self.no_rnn_layers, batch_size, self.hidden_size]) )

    def forward(self, x):
        x = self.embedding(x)
        batch_size, seq_len, max_len, emb_dim = x.size()
        x = x.view(batch_size*seq_len, 1, max_len, emb_dim) # why?
        y = self.CharCNN(x)
        y = y.permute(0,2,1)
        z = self.HighwayNetwork(y)
        hid = self.init_hidden(z.size()[0])
        z, _ = self.rnn(z, hid)
        z = self.fc(z)
        z = self.dropput(z)
        z = self.out(z)
        return z





# test charCNN
# charcnn_model = CharCNN(128)
# x = np.random.randn(128,1,10,128).astype(np.float32)
# x = charcnn_model(torch.from_numpy(x))
# print(x.size())

# highway_model = HighwayNetwork(charcnn_model.word_dim)
# # x change axis between 2,1
# x =  x.permute (0,2,1)
# x = highway_model(x)
# print(x.size())

# test CharCNNRNN
embedding_dim=128 
vocab_size=1000
word_vocab_size = 10
charcnn_rnn_model = CharCNNRNN(embedding_dim, vocab_size, word_vocab_size)
x = np.random.randint(vocab_size, size=(128,1,10)) 
x = charcnn_rnn_model(torch.from_numpy(x))
print(x.size())


        