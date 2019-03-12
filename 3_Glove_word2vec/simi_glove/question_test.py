#encoding:utf-8

import pandas as pd
import sys

from glove_wmd_gensim import *


QUESTION_PATH = '/opt/gongxf/python3_pj/Robot/original_data/金融测评问题集.xlsx'
OUTPUT_PATH='/opt/gongxf/python3_pj/Robot/simi_tfidf/QuestionTest_Result/tfidfModel_result.txt'



def word2vecmodel_test(text_list,text):
    file=open(OUTPUT_PATH,mode='w',encoding='utf-8')
    model = bulid_word2vec(model_path)
    word2vec_wmdistance_keyword(model=model, text_list=text_list,text=text)

def word2vecmodel_test_n(text_n,text):
    file=open(OUTPUT_PATH,mode='w',encoding='utf-8')
    model = bulid_word2vec(model_path)
    word2vec_wmdistance_nword(model=model, text_n=text_n,text=text)

def word2vecmodel_glove(text):
    file=open(OUTPUT_PATH,mode='w',encoding='utf-8')
    model = bulid_word2vec(model_path)
    word2vec_wmdistance(model=model,text=text)

if __name__=='__main__':
    # word2vec_tfidf_test("你好，我想开通任性付，上面的号码还是以前的号码，怎样改成现在的绑定号码？")
    df=pd.read_excel(QUESTION_PATH,sheet_name=0,ignore_index=False)
    for i in range(len(df['question'])):
        # tfidfmodel_test(df['question'][i])
        text=df['question'][i]
        # text_n=n_word(text)
        word2vecmodel_glove(df['question'][i])

