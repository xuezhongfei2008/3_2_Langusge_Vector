#!usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import csv
import re

#保留汉字字符
def extract_chinese(doc):
    pattern = '[^\u4e00-\u9fa5]+'
    return re.compile(pattern).sub('', doc.__str__())

#添加机器人知识库数据
def load_RobotKnowledge(path):
    #添加机器人知识库数据
    df=pd.read_excel(path,sheet_name=0,ignore_index=False)
    # df['content']=df[['分类名称','标准问题(必填)','默认答案(必填)','相似问题']].apply(tuple,axis=1)
    # print(df.columns)
    # df['content']=df[['分类名称','相似问题']].apply(tuple,axis=1)
    # df['content']=df['分类名称']
    # print(df[['相似问题','Unnamed: 9','Unnamed: 10','Unnamed: 11']])
    # df['content'].to_csv("RobotKnowledge.txt", header=False, index=False)
    # df['标准问题(必填)'].to_csv("Knowledge.txt", sep="\t",mode='a+',header=False, index=False,encoding='utf-8')
    df['标准问题(必填)'].to_csv("./data/Robot_Knowledge.txt",mode='w',header=False, index=False,encoding='utf-8')
#
def load_FinancialTerms(path):
    #添加机器人知识库数据
    df=pd.read_excel(path,sheet_name=0,ignore_index=True)
    df["话术"].to_csv("Knowledge.txt", mode='a+',header=False, index=False)
    # df[["场景","话术"]].to_csv("Knowledge.txt", mode='a+',header=False, index=False)

def load_Log(path):
    # df=pd.read_csv(path,engine='python',encoding='utf-8',sep='\t')
    df=pd.read_csv(path,header=None,names=['content'],engine='python',encoding='utf-8',error_bad_lines=False,)
    dff=df['content'].apply(extract_chinese)
    dfff=dff.drop(dff.index[dff.str.contains('http')==True])
    print(dfff)
    dfff.to_csv("./data/Knowledge.txt", mode='a+',header=False, index=False)

if __name__=='__main__':
    load_RobotKnowledge('/opt/gongxf/python3_pj/Robot/original_data/机器人知识库-2.23.xlsx')
    # load_FinancialTerms('F:\python3_pj\wordtovec\original_data\金融在线常见话术.xlsx')
    # load_Log('/opt/gongxf/python3_pj/Robot/13_Doc2Vec/data/15_16_17_question.csv')





















