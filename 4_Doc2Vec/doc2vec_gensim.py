# coding:utf-8

import sys
import gensim
import sklearn
import numpy as np
import jieba
from numpy import linalg as la
from operator import itemgetter, attrgetter

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
jieba.load_userdict("/opt/gongxf/python3_pj/Robot/original_data/finWordDict.txt")
all_text_path='/DATA/1_DataCache/FinCorpus/all_data_pure.csv'
text_path='/opt/gongxf/python3_pj/Robot/13_Doc2Vec/data/Knowledge_0816.txt'

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def stop_words():
    stop_words = []
    with open('/opt/gongxf/python3_pj/Robot/original_data/stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    return stop_words
#停用词字典列表
stopwords_list=stop_words()

def bulid_cut_file():
    file = open('./data/Knowledge_cutstop_0816.txt', 'w', encoding='utf-8')
    all_doc_list=[]
    i=0
    # print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    with open(text_path, "r", encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                all_doc_list.append((i,line))
                i+=1
                raw_words_cut = list(jieba.cut(line, cut_all=False))
                raw_words=[word for word in raw_words_cut if word not in stopwords_list]
                if len(raw_words)>0:
                    file.write(','.join(raw_words)+'\n')
            line = f.readline()
    file.close()

def get_datasest():
    with open("./data/Knowledge_cutstop_0816.txt", 'r',encoding='utf-8') as cf:
        docs = cf.readlines()
    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(',')
        word_list_len = len(word_list)
        word_list[word_list_len - 1] = word_list[word_list_len - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=400,epochs=50):
    model_dm = Doc2Vec(x_train, min_count=50, window=5, size=size, sample=1e-6, negative=5, dm=0,workers=16)
    #size 文本向量长度
    #dm=1 使用skip-gram模型
    model_dm.train(x_train, total_examples=model_dm.corpus_count,epochs=epochs)
    model_dm.save('model/Knowledge_cbow_816')
    return model_dm

def cos(vector1,vector2):
    inA=np.mat(vector1)
    inB=np.mat(vector2)
    num=float(inA*inB.T)
    denom=la.norm(inA)*la.norm(inB)
    return num/denom

def test():
    Robot_Know_list=[]
    Robot_Know=[]
    for line in open("/opt/gongxf/python3_pj/Robot/13_Doc2Vec/data/Robot_Knowledge.txt"):
        line=line.replace('\n', '')
        Robot_Know.append(line)
        cut_list=list(jieba.cut(line, cut_all=False))
        cut_list=[word for word in cut_list if word not in stopwords_list]
        # print(cut_list)
        Robot_Know_list.append(cut_list)
    # print(Robot_Know)
    model_dm = Doc2Vec.load("model/Knowledge_cbow_816")
    test="零钱宝钱不能全额转出"
    test_text=list(jieba.cut(test, cut_all=False))
    test_text = [word for word in test_text if word not in stopwords_list]
    print(test_text)
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print("inferred_vector_dm:",inferred_vector_dm)
    sims=[]
    for i in range(len(Robot_Know_list)):
        # print("i:",i, model_dm.docvecs[i])
        sim_cos=cos(inferred_vector_dm, model_dm.infer_vector(Robot_Know_list[i]))
        # print("sim_cos",sim_cos)
        # sim_cos=cos(inferred_vector_dm, model_dm.docvecs[i])
        i_sims=(i,sim_cos)
        sims.append(i_sims)
    return sorted(sims,key=itemgetter(1),reverse=True),Robot_Know

if __name__ == '__main__':
    # all_doc_list=bulid_cut_file()
    # x_train = get_datasest()
    # model_dm = train(x_train)
    sims,Robot_Know = test()
    for count, sim in sims[0:5]:
        print("list_doc", Robot_Know[count], sim)

