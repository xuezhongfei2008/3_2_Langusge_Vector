#--encoding:utf-8
import sys
import gensim
import sklearn
import numpy as np
import jieba
from numpy import linalg as la
from operator import itemgetter, attrgetter
from gensim import corpora,models,similarities
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
jieba.load_userdict("finWordDict.txt")

TaggededDocument = gensim.models.doc2vec.TaggedDocument
# 训练样本
raw_documents = [
    # '0现金分期单笔申请最高额度是为多少'
    # '1一个身份证号可以绑定几个任性付账户？',
    # '2为什么要实名认证',
    # '3为什么阿里蚂蚁花呗、京东白条就不用上报征信？',
    # '4什么是合并上报征信？',
    # '5任性付最高贷款额度是多少？',
    # '6任性付有哪些分期还款方式？',
    # '7任性付的短信还款提示如何发送？',
    # '8任性付的额度是否可以循环？',
    # '9办理了自动代扣，什么时间扣款？',
    # '10如何找回任性付app端登录密码？',
    # '11申请任性付失败原因？',
    # '12部分还款可以还款任意金额吗？',
    # '13什么时间自动代扣？',
    # '14任性贷在门店使用不了',
    # '15任性贷还款日是几号',
    '0无偿居间介绍买卖毒品的行为应如何定性',
    '1吸毒男动态持有大量毒品的行为该如何认定',
    '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '5为获报酬帮人购买毒品的行为该如何认定',
    '6毒贩出狱后再次够买毒品途中被抓的行为认定',
    '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '9一方未签字办理的结婚登记是否有效',
    '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '12身份证被别人冒用无法登记结婚怎么办？',
    '13同居后又与他人登记结婚是否构成重婚罪',
    '14未办登记只举办结婚仪式可起诉离婚吗',
    '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]
def doc2bow_model(raw_documents):
    corpora_documents = []
    for item_text in raw_documents:
        item_str = list(jieba.cut(item_text,cut_all=False))
        # print("item_str",item_str)
        corpora_documents.append(item_str)
    # print("corpora_documents:",corpora_documents)
    # 生成字典和向量语料
    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    # print("corpus:",corpus)
    similarity = similarities.Similarity('-Similarity-index', corpus, num_features=400)
    test_data_1 = '登陆密码'
    test_cut_raw_1 = list(jieba.cut(test_data_1,cut_all=False))
    # print("test_cut_raw_1",test_cut_raw_1)
    test_corpus_1 = dictionary.doc2bow(test_cut_raw_1) #生成测试语句的词袋向量
    # print("test_corpus_1",test_corpus_1)
    similarity.num_best = 10
    # for doc in similarity:
    #     print(doc)
    print(similarity[test_corpus_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples

def doc2vec_model(raw_documents):
    # 使用doc2vec来判断
    corpora_documents = []
    for i, item_text in enumerate(raw_documents):
        words_list = list(jieba.cut(item_text,cut_all=False))
        document = TaggededDocument(words=words_list, tags=[i])
        corpora_documents.append(document)
    # print(corpora_documents[:2])
    model = Doc2Vec(corpora_documents,dm_mean=0,vector_size=90,min_count=1)
    # model = 1Doc2Vec(dm_mean=0, dm=1, vector_size=128, min_count=1)
    # model.build_vocab(corpora_documents)
    model.train(corpora_documents, total_examples=model.corpus_count,epochs=1000)
    print('#########', model.vector_size)
    test_data_1 = '你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚'
    test_cut_raw_1 = list(jieba.cut(test_data_1,cut_all=False))
    print("test_cut_raw_1", test_cut_raw_1)
    inferred_vector = model.infer_vector(test_cut_raw_1)
    # print(inferred_vector)
    sims = model.docvecs.most_similar([inferred_vector], topn=10)
    print(sims)

if __name__=='__main__':
    # doc2bow_model(raw_documents)
    doc2vec_model(raw_documents)