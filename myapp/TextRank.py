#! python3
# -*- coding: utf-8 -*-
import networkx as nx
from spacy import load
from random import random
from math import log
from _io import TextIOWrapper as file_type
from rouge import Rouge
import pandas as pd
import numpy as np
from os.path import exists, basename

class TextRank():
    '''\
A TextRank class for TextRank algorithm. Aimed to extract abstract from our
cnn corpus. In this corpus, there are several files, each coded a piece of
news. Sentences in this news is seperated by two line-breaks. At the end of
each news, the corpus marked highlights in this news with "@highlight", each
followed by a sentence described the important information on one hand.

usage:
    The easies way to use it is:

        from TextRank import TextRank
        tr = TextRank(input_something)

    You can input a file handle or an existing file path, or the string of text
    you want to extract an abstract(if a string type input does not exist, we
    think it's the text).

    Or, if you have an example of abstract, you can constract the object by:

        tr = TextRank(input_something, example_abstract)

    It can also be a file handle, or an existing file path, or the string of
    your abstract.

    After construction of the TextRank object, you can extract the best 3
    sentences or more sentences in this text:

        tr.best(3) # or more if you want

    You can evaluate your abstract you just extracted by:

        scores = tr.evaluating()

    This will use ROUGE to evaluate your abstract compairing with the origenal
    text. If you gave an abstract example to build the object, this will also
    compaire the extracted abstract and the given abstract.

variants:
    text        Origenal text given
    graph       TextRank graph built
    sents       Sentences in text
    highlights  Abstract example given

    abstract    Abstract you just extracted with function "best", if you have
                not used "best", there's no "abstract"

functions:
    def __init__(self, file_read, abstract=None, encoding='utf8', d=0.85,
                 max_iter=200, max_diff=1e-3, min_weight=0, min_sent_len=3):
        Initiating the TextRank class

    def best(self, topn=3):
        Extracting the top x sentences you want.

    def evaluating(self):
        Evaluating the abstract you just extracted by ROUGE.'''

    nlp = load('en_core_web_lg')

    #def list_cnn_text_highlight(self):
    #    '''Devide text data and abstract example in cnn data set.'''
    #    text = self.text.split('@highlight')
    #    #print(text[1:])
    #    self.sents = list(filter(None, text[0].split('\n')))
    #    highlights = list(filter(None, [t.replace('\n', '') for t in text[1:]]))
    #    self.highlights = [highlight+'.' for highlight in highlights]

    def init_data(self):
        '''Token, Sents, Stopwords, lower, puncutations'''
        sentences = []
        doc = self.nlp(self.text)
        self.sents = [sent.text for sent in doc.sents]
        for sent in doc.sents:
            sentences.append([token.lemma_.lower()\
                              for token in sent if not\
                              (token.is_stop or token.is_punct or\
                               token.is_space or token.is_digit)])
        #print(sentences)
        #print(all(sentences))
        return sentences

    def sentence_similarity(self, sentences, i, j, min_sent_len):
        '''Calculating similarity between sentences.'''
        #print(type(sentences))
        #print(sentences)
        if len(sentences[i]) < min_sent_len or\
           len(sentences[j]) < min_sent_len:
            return 0
        #print(sentences)
        word_both = 0
        for word in sentences[i]:
            if word in sentences[j]:
                word_both+=1
        #print(sentences[i])
        #print(sentences[j])
        #print(word_both, log(len(sentences[i])), log(len(sentences[j])))
        return word_both/(log(len(sentences[i])) + log(len(sentences[j])))
    
    def init_graph(self, sentences, min_weight, min_sent_len):
        '''\
    Initiate a graph for TextRank.
    
    Sentences is a list list. For example,
    there's a text:
    
        I have a pen. I have an apple. ohh! Applepen!
    
    The given sentences should be like:
    
        array[ ['I', 'have', 'a', 'pen', '.'],
               ['I', 'have', 'an', 'apple', '.'],
               ['ohh', '!'], ['Applepen'. '!'] ]
    
    Note, it is a good idea to clean your data.'''
        self.graph = nx.Graph()
        for i in range(len(sentences)):
            self.graph.add_node(i, weight=random(), sent=self.sents[i])
        for i in range(len(sentences)-1):
            for j in range(i+1, len(sentences)):
                w = self.sentence_similarity(sentences, i, j, min_sent_len)
                if w > min_weight:
                    self.graph.add_edge(i, j, weight=w)
                else:
                    if (i, j) in self.graph.edges:
                        self.graph.remove_edge(i, j)
        
        to_be_removed = []
        for node in self.graph:
            if len(self.graph.adj[node]) == 0:
                #self.graph.remove_node(node)
                to_be_removed.append(node)
        self.graph.remove_nodes_from(to_be_removed)
    
    def update_score(self, sentences, i, d):
        '''Calculating TextRank score.'''
        sum_jk = {}
        for j in self.graph.adj[i]:
            sum_jk[j] = sum([self.graph.edges[(j,k)]['weight']
                             for k in self.graph.adj[j]])

        w = (1-d)/sum([len(sentences[node]) for node in self.graph.nodes]) +\
            d * sum([self.graph.nodes[j]['weight'] *\
                     self.graph.edges[(i,j)]['weight'] / sum_jk[j]\
                     for j in self.graph.adj[i]])

        self.graph.add_node(i, weight=w)
        return w

    def __init__(self, file_read, highlights=None, encoding='utf8', d=0.85,
                 max_iter=200, max_diff=1e-3, min_weight=0, min_sent_len=3):
        if isinstance(file_read, str):
            if exists(file_read):
                with open(file_read, 'r', encoding=encoding) as handle:
                    file_read = handle.read()
            else:
                pass
        elif isinstance(file_read, file_type):
            file_read = file_read.read()
            #print('file')
        self.text = file_read
        #self.list_cnn_text_highlight()

        if isinstance(highlights, str):
            if exists(highlights):
                with open(highlights, 'r', encoding=encoding) as handle:
                    highlights = handle.read()
                self.highlights = [sent.text for sent in self.nlp(highlights).sents]
            else:
                self.highlights = [sent.text for sent in self.nlp(highlights).sents]
        elif isinstance(highlights, file_type):
            highlights = highlights.read()
            self.highlights = [sent.text for sent in self.nlp(highlights).sents]

        sentences = self.init_data()
        #print(sentences)

        self.init_graph(sentences, min_weight, min_sent_len)
        #print(len(self.graph))

        for k in range(max_iter):
            tmp_flag = [False] * len(sentences)
            for i in range(len(sentences)):
                if i not in self.graph:
                    tmp_flag[i] = True
                    continue

                old = self.graph.nodes[i]['weight']
                new = self.update_score(sentences, i, d)
                if new-old < max_diff:
                    tmp_flag[i] = True
                else:
                    tmp_flag[i] = False
            if all(tmp_flag):
                break
        #print(len(self.graph))

    def best(self, topn=3):
        '''\
Take the best several sentences to form an abstract. "topn" is the number of
sentences you want to extract, default "topn" is -1, means it will equals to
the length of highlights given by the example.'''
        if topn == -1:
            topn=len(self.highlights)
        sorted_sents = sorted([node for node in self.graph.nodes(data=True)],
                              key = lambda x:x[1]['weight'], reverse=True)
        #print(len(sorted_sents[:]))
        self.abstract = [node[1]['sent'] for node in sorted_sents[:topn]]
        return ' '.join(self.abstract)

    def evaluating(self):
        rouge = Rouge()
        try:
            scores = rouge.get_scores(' '.join(self.abstract),
                                      ' '.join(self.sents))
        except:
            return None

        try:
            scores.append(rouge.get_scores(' '.join(self.abstract),\
                                           ' '.join(self.highlights))[0])
        except:
            pass

        return scores

def add_score_to_csv(scores, filename, csv):
    '''save ROUGE evaluated scores in a csv file'''
    if scores == None:
        df = pd.DataFrame([[np.nan]*18], index=(filename,),\
                          columns = list(range(18)))
    else:
        s = np.empty(18, dtype=np.float64)
        col = np.empty((3, 18), dtype='<U21')
        tmp_list = ['abstract-to-fulltext', 'abstract-to-abstract']
        tmp_count = 0
        for col_1 in range(len(scores)):
            for col_2 in scores[col_1]:
                for col_3 in scores[col_1][col_2]:
                    (col[0, tmp_count], col[1, tmp_count], col[2, tmp_count])\
                            = (tmp_list[col_1], col_2, col_3)
                    #print(type(scores), col_1, col_2, col_3)
                    #print(type(s), type(tmp_count))
                    s[tmp_count] = scores[col_1][col_2][col_3]
                    tmp_count += 1

        #print(col)
        #print(s)
        df = pd.DataFrame(s.reshape(1,18), index=(filename,), columns=pd.MultiIndex.from_arrays(col))

    #print(df)
    if exists(csv):
        df.to_csv(csv, mode='a', header=False)
    else:
        df.to_csv(csv)

def main():
    from glob import glob
    files = glob('./test1000/*.story')
    #files = ['000c835555db62e319854d9f8912061cdca1893e.story',\
    #         '00a2aef1e18d125960da51e167a3d22ed8416c09.story']
    file_num = len(files)
    file_count = 1
    for f in files:
        highlights = f+'sam'
        print('file: %s\n%i of %i\n' % (f, file_count, file_num))
        tr = TextRank(f, highlights)
        #print(tr.best(max(3, len(tr.highlights))))
        out=f.replace('.story', '.TextRank')
        with open(out, 'w', encoding='utf8') as handle:
            #print(len(tr.highlights))
            #print(tr.highlights)
            handle.write(tr.best(max(3, len(tr.highlights))))
        add_score_to_csv(tr.evaluating(), basename(f), './test1000/scores.csv')
        file_count+=1

if __name__ == '__main__':
    main()
