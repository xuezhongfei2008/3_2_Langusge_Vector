#encoding:utf-8
import pandas as pd
from gensim.models import Word2Vec,FastText
# import jieba
import os
import pickle
import numpy

# jieba.load_userdict("/opt/gongxf/python3_pj/Robot/original_data/finWordDict.txt")
stop_words_path = '/opt/gongxf/python3_pj/Robot/original_data/stop_words.txt'

input_path= '/opt/gongxf/python3_pj/Robot/generate_data/knowledge0720_cut.txt'
# input_path= '/opt/gongxf/python3_pj/Robot/generate_data/test_aa.txt'

model_path='/opt/gongxf/python3_pj/Robot/2_fasttext2vec/knowledge0720_cut'

def stop_words():
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    return stopwords

# Step 1: 生成分词的文本列表
def read_data(input_path=input_path,model_path=model_path):
    """    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中,#读取停用词    """
    if (os.path.exists(model_path + "/dialogue_list.pkl")):
        print("加载存在的数据：分词后文本列表.....")
        # 加载分词后文本列表
        pickle_file = open(model_path + '/dialogue_list.pkl', 'rb')
        raw_word_list = pickle.load(pickle_file)
    else:
        print("第一次运行生产数据：分词后文本列表......")
        # 保存分词后问题列表
        raw_word_list = []
        for line in open(input_path,'r',encoding='utf-8').readlines():
            while '\n' in line:
                line = line.replace('\n', '').split(' ')
                raw_word_list.append(line)
        pickle_file = open(model_path + '/dialogue_list.pkl', 'wb')
        pickle.dump(raw_word_list, pickle_file)
        pickle_file.close()
    return raw_word_list
#训练词向量
def bulid_fasttext2vec(model_path=model_path):
    # if (os.path.exists(model_path + "/word2vec_gensim")):
    if (os.path.exists(model_path + "/skip_dia.model")):
    # if (os.path.exists('/DATA/1_DataCache/FinCorpus/cbow.model')):
        print("加载存在的数据：word2vec 模型.....")
        # 加载分词后文本列表
        model = FastText.load(model_path+'/skip_dia.model')
        # model = Word2Vec.load(model_path + '/cbow.model')
    else:
        print("第一次运行生产数据：FastText_gensim......")
        # print(sentences)
        sentences=read_data(input_path=input_path,model_path=model_path)
        model = FastText(sentences, sg=0,size=300, window=5, min_count=3, negative=5, sample=0.001, hs=0, workers=16)
        model.save(model_path+'/knowledge_fast.model')
    return model




if __name__=='__main__':
    # sentences = read_data(input_path=input_path, model_path=model_path)
    # print(sentences)
    model=bulid_fasttext2vec(model_path)
