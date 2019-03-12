#encoding:utf-8
from gensim.models import Word2Vec
import gensim
import jieba
import os
import pickle
import numpy

jieba.load_userdict("./word2vec_wmd_model/finWordDict.txt")
stop_words_path = './word2vec_wmd_model/stop_words.txt'

input_path= './word2vec_wmd_model/dialogue_all_session.csv'
model_path='./word2vec_wmd_model'

def stop_words():
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    return stopwords

# Step 1: 生成分词的文本列表
def read_data(input_path,model_path):
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
        with open(input_path , "r", encoding='UTF-8') as f:
            line = f.readline()
            while line:
                while '\n' in line:
                    line = line.replace('\n', '')
                if len(line) > 0:  # 如果句子非空
                    raw_words = list(jieba.cut(line, cut_all=False))
                    raw_word_list.append([word for word in raw_words if word not in stop_words()])
                line = f.readline()
        pickle_file = open(model_path + '/dialogue_list.pkl', 'wb')
        pickle.dump(raw_word_list, pickle_file)
        pickle_file.close()
    return raw_word_list

def question_cut(model_path):
    pickle_file = open(model_path+'/question_list.pkl', 'rb')
    question = pickle.load(pickle_file)
    question_list=[]
    for i in range(len(question)):
        raw_words=list(jieba.cut(question[i], cut_all=False))
        question_list.append([word for word in raw_words if word not in stop_words()])
    return question,question_list

#加载训练好的word2vec模型，并计算问题相似度
def word2vec_wmdistance(model_path,text):
    #step3:加载模型
    if True:
        print("加载存在的数据：word2vec 模型.....")
        # simmilar_model = Word2Vec.load(model_path+'/word2vec_gensim')
        simmilar_model = gensim.models.KeyedVectors.load_word2vec_format('/ZC_DATA/gloveVectors.txt')

    else:
        print("第一次运行生产数据：word2vec_gensim......")
        sentences=read_data(input_path,model_path)
        simmilar_model = Word2Vec(sentences, sg=1,size=200, window=5, min_count=80, negative=5, sample=0.001, hs=1, workers=16)
        simmilar_model.save(model_path+'/word2vec_gensim.model')
    #setp2:加载标准问题列表
    question, question_list=question_cut(model_path)
    #step3：计算测试问题与标准问题 相似性
    distinct=[]
    text_list = list(jieba.cut(text, cut_all=False))
    text_list=[word for word in text_list if word not in stop_words()]
    for j in range(len(question)):
        distinct1 = simmilar_model.wmdistance(text_list, question_list[j])
        dist_dict=(question[j],distinct1,j)
        distinct.append(dist_dict)
    distinct_sort=sorted(distinct, key=lambda item: item[1])
    print("输入测试文本：", text)
    if distinct_sort[0][1] ==numpy.inf:
        print("抱歉，我还在学习过程中。。。")
    else:
        for question in distinct_sort[0:5]:
            pass
            print("相似问题：", question)
    return question,distinct

def main(text):
    text="零钱宝钱不能全额转出"
    word2vec_wmdistance(model_path=model_path,text=text)

if __name__=='__main__':
    main(text="")

