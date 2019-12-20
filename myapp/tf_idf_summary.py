import pickle
import re
import codecs
import math
import operator
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

##拆分方法20191218
#将计算tf方法拆分出来，参数由文件改为文本内容

# 获取停用词列表
def get_stop_words(path):
    stop_words=[]
    for line in open(path,'r'):
        stop_words.append(line.strip())
    return stop_words

#删除停用词
def getridofsw(lis, swlist):  # 去除文章中的停用词
    afterswlis = []
    for i in lis:
        if str(i) in swlist:
            continue
        else:
            afterswlis.append(str(i))
    return afterswlis

# 统计词频，词频字典{词：个数} 和 词集合set(便于后续计算idf)
def count_word(content):
    words_set = set()
    wordstf_dic = {}
    new_word_list = []
    words_list = re.split('\s', content)#空行，空格等 进行分词（尚存在问题：出现词''）
    for word in words_list:
        new_word_list.append(re.sub("[\.\,\"\'\:\(\)\?]", '', word))
    del_word = get_stop_words('myapp/stop_words_eng.txt')# 获取停用词表列表
    afterswlis = getridofsw(new_word_list, del_word)
    words_set = set(afterswlis)
    for word in afterswlis:
        if word in wordstf_dic:
            wordstf_dic[word] = wordstf_dic[word] + 1
        else:
            wordstf_dic[word] = 1
    return wordstf_dic,words_set

def count_tfidf(words_dic_per_file, num_files, all_words_idf):
    words_tfidf = {}
    #num_files = len(files_Array)
    for key, value in words_dic_per_file.items():
        if key != " " and key in all_words_idf.keys():
            words_tfidf[key] = value * math.log(num_files / (all_words_idf[key] + 1)) #idf直接读取
    # 降序排序
    values_list = sorted(words_tfidf.items(), key=lambda item: item[1], reverse=True)
    return values_list

def read_tfidf(path):
    word_tfidf={}
    f=open(path,'r',encoding="utf-8")
    for line in f:
        word=line.split(':')[0]
        tfidf=line.split(':')[1]
        word_tfidf[word]=tfidf
    f.close()
    return word_tfidf

def split_sentences(content):
    #sentence_list = sent_tokenize(content)
    # sent_tokenize 对缩写词不起作用，需要使用punked定义缩写词表
    punkt_param = PunktParameters()
    abbreviation = ['i.e','e.g']
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)
    sentence_list = tokenizer.tokenize(content)
    return sentence_list

#抽取文本摘要
def get_summury(path,content):
    word_tfidf= read_tfidf(path)
    sentence_score={}
    ##利用nltk 进行分句
    sentence_list = split_sentences(content)
    for s in sentence_list:
        sentence_score[s]=0
        word_list = re.split('\s',s)
        new_word_list=[]
        for word in word_list:
            new_word_list.append(re.sub("[\.\,\"\'\:\(\)\?]", '', word))
        del_word = get_stop_words('myapp/stop_words_eng.txt')# 获取停用词表列表
        afterswlis = getridofsw(new_word_list, del_word)
        for w in afterswlis:
            if w is not '' and w in word_tfidf.keys():
                sentence_score[s]+=float(word_tfidf[w])
    result = sorted(sentence_score.items(),reverse=True,key=operator.itemgetter(1))
    return result[0]##返回 摘要+分值

#主函数
def summary1(file_content):
    wordstf_dict_list=[] 

    #1.分词，计算词频，并返回词频字典（包括预处理：去掉标点停用词等），考虑是否直接用 nltk
    result_tuple = count_word(file_content) 
    wordstf_dic = result_tuple[0]
    wordstf_dict_list.append(wordstf_dic)

    # 2.获取测试语料库的idf
    with codecs.open('myapp/all_words_idf.dat', 'rb') as f1:# 之前9万测试数据的结果，直接读取
        all_words_idf = pickle.load(f1)

    # 3.计算tf-idf，降序保存
    num_files = 92580
    for word_dict in wordstf_dict_list:
        words_tfidf = count_tfidf(word_dict, num_files, all_words_idf)
        #存储
        with codecs.open("tf_idf.txt", "a", encoding="utf-8") as f:
            for content in words_tfidf:
                f.write(str(content[0]) + ":" + str(content[1]) + "\r\n")

    #4.统计句子的tf-idf值，获取摘要和分值
    tfidf_path='tf_idf.txt'
    summary = get_summury(tfidf_path,file_content)#返回元组
        #创建新文件夹，用来存放通过tfidf方法获得的抽取式摘要
    summary_path='summary.txt'
    file=codecs.open(summary_path,"a",encoding="utf-8")
        #file.write(summary)
    file.write(''.join('%s,%f' %(summary[0],summary[1])))
    print(summary)
    return summary


    
if __name__ == '__main__':
    data ='''A plethora of bibliometric indicators is available nowadays to gauge research performance. The spectrum of bibliometric based measures is 
very broad, from purely size-dependent indicators (e.g. raw counts of scientific contributions and/or citations) up to size-independent
measures (e.g. citations per paper, publications or citations per researcher), through a number of indicators that effectively combine 
quantitative and qualitative features (e.g. the h-index). In this paper we present a straightforward procedure to evaluate the scientific
contribution of territories and institutions that combines size-dependent and scale-free measures. We have analysed in the paper the 
scientific production of 189 countries in the period 2006–2015. Our approach enables effective global and field-related comparative
analyses of the scientific productions of countries and academic/research institutions. Furthermore, the procedure helps to identifying 
strengths and weaknesses of a given country or institution, by tracking variations of performance ratios across research fields. Moreover,
by using a straightforward wealth-index, we show how research performance measures are highly associated with the wealth of countries and 
territories. Given the simplicity of the methods introduced in this paper and the fact that their results are easily understandable
by non-specialists, we believe they could become a useful tool for the assessment of the research output of countries and institutions.
'''
    main(data)
