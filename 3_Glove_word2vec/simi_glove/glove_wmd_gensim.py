#encoding:utf-8
import pandas as pd
from gensim.models import Word2Vec
import gensim
import jieba
import os
import pickle
import sys
import numpy
sys.path.append('../')
from Ex_keywords.ex_keywords_word2vec import keywords
jieba.load_userdict("/opt/gongxf/python3_pj/Robot/original_data/finWordDict.txt")
stop_words_path = '/opt/gongxf/python3_pj/Robot/original_data/stop_words.txt'

input_path_q = '/opt/gongxf/python3_pj/Robot/generate_data/question_all.csv'

# model_path='/DATA/1_DataCache/FinCorpus'
model_path='/opt/gongxf/python3_pj/Robot/word2vec_wmd/word2vec_wmd_model'

def stop_words():
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    return stopwords

# Step 1: 生成分词的文本列表
def read_data(input_path=input_path_q,model_path=model_path):
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
#训练词向量
def bulid_word2vec(model_path=model_path):
    # if (os.path.exists(model_path + "/word2vec_gensim")):
    # if (os.path.exists(model_path + "/skip_q.model")):
    if True:
        # print("加载存在的数据：word2vec 模型.....")
        # 加载分词后文本列表
        model=Word2Vec.load('/opt/gongxf/python3_pj/Robot/1_word2vec/model_skip_dia/skip_dia.model')
        # model = Word2Vec.load(model_path+'/word2vec_gensim')
        # model=gensim.models.KeyedVectors.load_word2vec_format('/opt/gongxf/python3_pj/Robot/Glove_word2vec/model/gloveVectors.txt')
        # model = gensim.models.KeyedVectors.load_word2vec_format('/opt/gongxf/python3_pj/Robot/Glove_word2vec/data_50_400/dialogue_vectors.txt')
        # model = Word2Vec.load(model_path + '/cbow.model')
    else:
        print("第一次运行生产数据：word2vec_gensim......")
        # print(sentences)
        sentences=read_data(input_path=input_path_q,model_path=model_path)
        model = Word2Vec(sentences, sg=1,size=200, window=5, min_count=80, negative=5, sample=0.001, hs=1, workers=16)
        model.save(model_path+'/skip_d.model')
    return model
#测试词向量效果
def test_word2vec(text,model):
    # pass
    # print("aa",model.most_similar(positive=['任性贷', '任性付'], negative=['苏宁金融']))
    # print("苏宁 VS 苏宁金融",model.similarity('苏宁', '苏宁金融'))
    simi_text=model.most_similar(text)
    print("model.most_similar:", text, "  ----")
    for ii in simi_text:
        print(ii)
    #打印字典词
    # print(len(model.wv.vocab.keys()))
    # for j in model.wv.vocab.keys():
    #     print(j)
    #打印某个词向量
    # print(model['任性付'])  # raw numpy vector of a word

def question_cut():
    pickle_file = open('/opt/gongxf/python3_pj/Robot/word2vec_wmd/word2vec_wmd_model/question_list.pkl', 'rb')
    question = pickle.load(pickle_file)
    # print("question_list", question)
    question_list=[]
    question_keyword=[]
    for i in range(len(question)):
        raw_words=list(jieba.cut(question[i], cut_all=False))
        question_list.append([word for word in raw_words if word not in stop_words()])
        question_keyword.append(keywords(question[i]))
    # print(question_list)
    return question,question_list

def question_cut_n():
    pickle_file = open('/opt/gongxf/python3_pj/Robot/word2vec_wmd/word2vec_wmd_model/question_list.pkl', 'rb')
    question = pickle.load(pickle_file)
    # print("question_list", question)
    question_list=[]
    question_n=[]
    for i in range(len(question)):
        raw_words=list(jieba.cut(question[i], cut_all=False))
        question_list.append([word for word in raw_words if word not in stop_words()])
    # print(question_list)
    return question,question_list

def word2vec_wmdistance(model,text):
    #加载标准问题列表
    question, question_list=question_cut()
    distinct=[]
    text_list = list(jieba.cut(text, cut_all=False))
    text_list=[word for word in text_list if word not in stop_words()]
    # print("text_list:",text_list)
    for j in range(len(question)):
        # print("2qu:",question_list[j])
        distinct1 = model.wmdistance(text_list, question_list[j])
        dist_dict=(question[j],distinct1,j)
        distinct.append(dist_dict)
    # print("distinct",distinct)
    # print(sorted(distinct,key=lambda item:item[1]))
    distinct_sort=sorted(distinct, key=lambda item: item[1])
    print("输入测试文本：", text)
    # 输出
    if distinct_sort[0][1] ==numpy.inf:
        print("抱歉，我还在学习过程中。。。")
    else:
        for question in distinct_sort[0:5]:
            pass
            print("相似问题：", question)
    return question,distinct

# def word2vec_wmdistance_keyword(model,text_list,text):
#     #加载标准问题列表
#     question, question_list,question_keyword=question_cut()
#     distinct=[]
#     # text_list=text
#     # text_list = list(jieba.cut(text, cut_all=False))
#     # text_list=[word for word in text_list if word not in stop_words()]
#     # print("text_list:",text_list)
#     for j in range(len(question)):
#         # print("2qu:",question_list[j])
#         distinct1 = model.wmdistance(text_list, question_keyword[j])
#         dist_dict=(question[j],distinct1,j)
#         distinct.append(dist_dict)
#     # print("distinct",distinct)
#     # print(sorted(distinct,key=lambda item:item[1]))
#     distinct_sort=sorted(distinct, key=lambda item: item[1])
#     print("输入测试文本：", text,"------关键词：",text_list)
#     # 输出
#     if distinct_sort[0][1] ==numpy.inf:
#         print("抱歉，我还在学习过程中。。。")
#     else:
#         for question in distinct_sort[0:5]:
#             pass
#             print("相似问题：", question,"相似问关键词：",list(question_keyword[question[2]].keys()))
#     return question,distinct
#
#
# def word2vec_wmdistance_nword(model,text_n,text):
#     #加载标准问题列表
#     question, question_list,question_n=question_cut_n()
#     distinct=[]
#     for j in range(len(question)):
#         # print("2qu:",question_list[j])
#         distinct1 = model.wmdistance(text_n, question_n[j])
#         dist_dict=(question[j],distinct1,j)
#         distinct.append(dist_dict)
#     # print("distinct",distinct)
#     # print(sorted(distinct,key=lambda item:item[1]))
#     distinct_sort=sorted(distinct, key=lambda item: item[1])
#     print("输入测试文本：", text,"------动名词：",text_n)
#     # 输出
#     if distinct_sort[0][1] ==numpy.inf:
#         print("抱歉，我还在学习过程中。。。")
#     else:
#         for question in distinct_sort[0:5]:
#             pass
#             print("相似问题：", question,"动名词：",question_n[question[2]])
#     return question,distinct

if __name__=='__main__':
    while True:
        text = input("输入测试：")
        model=bulid_word2vec(model_path=model_path)
        test_word2vec(text, model)
    # step 1:读取文件中的内容组成一个列表
    # words = read_data(input_path,stop_words_path,model_path)
    # question_cut()
    # model=bulid_word2vec(model_skip_dpath)
    # text="零钱宝钱不能全额转出"
    # # test_word2vec(model=model,text=text)
    # # word2vec_wmdistance(model=model,text=text)
    # word2vec_wmdistance_nword()
