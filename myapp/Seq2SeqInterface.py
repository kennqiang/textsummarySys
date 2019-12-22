from io import open
import re
import spacy
import os
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import test_final_v2




class seq2seq:
    def __init__(self, weight_path=None):
        self.SOS = 0
        self.EOS = 1
        self.OOV = 2

        if weight_path==None:
            weight_path = r'weight'

        # load spacy
        spacy_path = os.path.join(weight_path, r'en_core_web_lg-2.2.0')
        self.nlp = spacy.load(spacy_path)

        # load word_dict and number_dict
        wd_path = os.path.join(weight_path,'word_dict.pkl' )
        nd_path = os.path.join(weight_path, 'number_dict.pkl')
        with open(wd_path, "rb") as fp:   #Pickling
            self.word_dict = pickle.load(fp)
        with open(nd_path, "rb") as fp:   #Pickling
            self.number_dict = pickle.load(fp)


        # NN
        n_class = 159594  # vocab list
        vocab_size = 159594

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('now using {}'.format(device))
        # Parameter
        n_hidden = 256

        embedding_dim = 300
        num_layers = 1
        batch_size = 1
        step = 0

        # self.model = test_final.Attention()
        model_path = os.path.join(weight_path, 'model.checkpoint')
        self.model = test_final_v2.init(model_path, use_gpu)
        # model.eval()

        
    def pre_process(self, text:str):
        SOS = self.SOS
        EOS = self.EOS
        OOV = self.OOV
        
        text=text.replace('\n',' ')
        text=text.replace('"','" ')    
        print(text)
        doc = self.nlp(text)
        # for token in doc:
        #     print(token)
        # print(doc)

        vec = []
        vec.append(SOS)
        for token in doc:
            word = token.lemma_
            if token.is_oov:
                vec.append(OOV)
            else:
                vec.append(self.word_dict[word])
        vec.append(EOS)

        return vec


    def predict(self, input:str):
        n_class = 159594  # vocab list
        vocab_size = 159594

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('now using {}'.format(device))
        # Parameter
        n_hidden = 256

        embedding_dim = 300
        num_layers = 1
        batch_size = 1
        step = 0

        vec = self.pre_process(input)

        out = self.model(vec, self.number_dict, beam_search=True)
        print(out)
        output = ''
        for i in out:
            if i!='<SOS>' and i!='<EOS>':
                output+=i+' '
        return output


if __name__ == "__main__":
    model = seq2seq()
    print(model.pre_process('hello world'))
    print(model.predict('hello world'))