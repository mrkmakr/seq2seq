#https://qiita.com/kenchin110100/items/b34f5106d5a211f4c004

import json
import glob
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import MeCab
import collections
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import optimizer
from chainer import serializers

file_name = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_name, 'temp_data')
#data_path = os.path.normpath(joined_path)
print(data_path)

def convert(sentence, dictionary):
    return [dictionary[word] if word in dictionary.keys() else -1 for word in sentence]
def process(ids):
    return [end_id] + ids + [null_id] * (th_seq_length - len(ids) - 1)

with open(os.path.join(data_path, 'word2id_dict.pkl'),'rb') as f:
    word2id_dict = pickle.load(f)
with open(os.path.join(data_path, 'id2word_dict.pkl'),'rb') as f:
    id2word_dict = pickle.load(f)
with open(os.path.join(data_path, 'questions.pkl'),'rb') as f:
    questions = pickle.load(f)
with open(os.path.join(data_path, 'answers.pkl'),'rb') as f:
    answers = pickle.load(f)

m = MeCab.Tagger ("-Owakati")

null_id = 0
end_id = 1
th_seq_length = 20

n_words = len(word2id_dict)

class RNN(chainer.Chain):
    def __init__(self, n_words, n_hiddens):
        super(RNN,self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, n_hiddens)
            self.lstm = L.LSTM(n_hiddens, n_hiddens)
            self.fc = L.Linear(n_hiddens, n_words)
            
    def reset_state(self):
        self.lstm.reset_state()

    def get_state(self):
        return self.lstm.c, self.lstm.h
    
    def set_state(self, c, h):
        self.lstm.c = c
        self.lstm.h = h
        
    def __call__(self,x):
        h = self.embed(x)
        h = self.lstm(h)
        h = self.fc(h)
        return h
    

def batch_sampling(x,y,bs):
    r = np.random.permutation(n_seq)[:bs]
    x_batch = x[r]
    y_batch = y[r]
    return x_batch, y_batch

enc = RNN(n_words, 300)
dec = RNN(n_words, 300)

serializers.load_npz(os.path.join(data_path, 'enc_trained'),enc)
serializers.load_npz(os.path.join(data_path, 'dec_trained'),dec)

bs = 32
ite_pre = 100
ite_seq2seq = 30000

def chat(inp, argmax_decoding = False, print_info = True):
    """
    inp : japanese sentence input
    argmax_decoding : if this is True, this bot reply deterministically
    print_info : if this is True, print input and some information
    """
    inp_wakati = m.parse(inp).split()
    inp_id = convert(inp_wakati, word2id_dict)
    #print(inp_id)
    q = np.array(process(inp_id))

    if print_info:
        if max(np.mean(q == questions,axis=1)) == 1:
            print('training data include this question')
        if -1 in inp_id:
            print('input includes unknown word')
    q = q[::-1]
    
    id_preds = []

    enc.reset_state()
    dec.reset_state()
    for t in range(th_seq_length - 1):
        p = enc(q[t:t+1])

    c,h = enc.get_state()
    dec.set_state(c,h)
    ps = []
    for t in range(1, th_seq_length):
        if t == 1:
            p = dec(np.array([end_id]))
        else:
            p = dec(np.array([pred_id]))
            
        p = F.softmax(p).data[0]
        
        if argmax_decoding == True:
            pred_id = np.argmax(p)
        else:
            pred_id = np.random.choice(range(n_words), p = p)
        p = p[pred_id]
        ps.append(p)
        id_preds.append(int(pred_id))
        if pred_id == null_id:
            break
#     p_joint = 1
#     for p in ps:
#         p_joint *= p
#     print(p_joint ** (1/len(ps)))
    out = ''.join(convert(id_preds[:-1],id2word_dict))
    if print_info:
        print('input  : ', inp)
    print(u'output : ', out)
    
    return out

while True:
    inp = input('input : ')
    print(inp)
    out = chat(inp, print_info = False)
    print('-----')
