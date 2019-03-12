# coding: utf-8

from collections import namedtuple
import random
import csv
import re
import string
import jieba

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

jieba.load_userdict("finWordDict.txt")

# step1:读取原始语料库数据
def read_data(doc_path):
    # 对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中,#读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    # print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    # 读取文本，预处理，分词，得到词典
    Doc_word_list = []
    with open(doc_path, "r", encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            while '\"' in line:
                line = line.replace('\"', '')
            if len(line) > 0:  # 如果句子非空
                line = re.sub("\d+", 'SZ', line)
                Doc_words = jieba.cut(line, cut_all=False)
                Doc_word_list.append(' '.join(word for word in Doc_words if word not in list(stop_words)))
            line = f.readline()
    LabelDoc = namedtuple('LabelDoc', 'words tags')  #定义一个namedtuple对象：LabelDoc包含words，tags属性
    # exclude = set(string.punctuation)  #包含所有标点的字符串,标点符号的集合
    all_docs = []
    count = 0
    for sen in Doc_word_list:
        word_list = sen.split()
        if len(word_list) < 3:
            # print("len(word_list)",len(word_list))
            continue
        tag = ['SEN_' + str(count)]
        count += 1
        all_docs.append(LabelDoc(sen.split(), tag))
    # print("all_docs:",all_docs)
    return all_docs

# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(input_data, min_cut_freq):
    # print("input_data:",input_data)
    words=[]
    LabelDoc = namedtuple('LabelDoc', 'words tags')
    for i in input_data:
        for j in i.words:
            words.append(j)
    # print("words:",words)
    count_org = [['UNK', -1]]
    count_org.extend(collections.Counter(words).most_common())
    # print("count_org",count_org)
    count = [['UNK', -1]]
    for word, c in count_org:
        word_tuple = [word, c]
        if word == 'UNK':
            count[0][1] = c
            continue
        if c >= min_cut_freq:           #出现频率大于min_cut_freq保存
            count.append(word_tuple)
    dictionary = dict()
    for word, _ in count:             #取大于min_cut_freq的单词作为词汇
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for tup in input_data:
        word_data = []
        for word in tup.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            word_data.append(index)
        # print("word_data",word_data)
        data.append(LabelDoc(word_data, tup.tags))
    count[0][1] = unk_count
    print("data:",data)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # print("count:",count)
    # print("dictionary:",dictionary)
    # print("reverse_dictionary",reverse_dictionary)
    return data, count, dictionary, reverse_dictionary    #data就是input_data数据进行 数字编码
    #count  [['UNK', 0], ['上报', 4], ['征信', 4], ['任性付', 3]]字符数字统计     #dictionary: {'UNK': 0, '上报': 1, '征信': 2, '任性付'] 字符构建的字典


class doc2vec():
    def __init__(self,
                 vocab_list=None,
                 doc_list=None,
                 batch_size=128,
                 embedding_size=128,  # Dimension of the embedding vector.
                 skip_window=1,  # How many words to consider left and right.
                 num_sampled=10,   #负采用个数
                 num_skips=2,
                 learning_rate=1.0,
                 logdir=None
                 ):
        print("初始化")
        if 1==None:
            self.load_model(model_path)
        else:
            # print(type(vocab_list))
            # assert type(vocab_list)==list
            self.vocab_list=vocab_list
            self.vocabulary_size=len(reverse_dictionary)
            self.paragraph_size=len(doc_list)
            self.batch_size=batch_size
            self.embedding_size=embedding_size
            self.skip_window=skip_window
            self.num_sampled=num_sampled
            self.num_skips=num_skips
            self.learning_rate=learning_rate
            self.logdir=logdir

            self.word2id={}         #word=>id 映射
            for i in range(self.vocabulary_size):
                # print("self.vocab_list[i]",self.vocab_list[i])
                self.word2id[self.vocab_list[i]]=i
            print("self.word2id",self.word2id)

            self.train_words_num=0  #训练的单词对数
            self.train_sents_num=0  #训练的句子数
            self.train_times_num=0  #训练的次数

            #训练误差记录
            self.train_loss_records=collections.deque(maxlen=10)  #保存最近10次的误差
            self.train_loss_k10=0


        self.build_graph()
        # self.init_op()

    # if model_path!=None:
    #     tf_model_path=os.path.join(model_path,'tf_vars')
    #     self.saver.restore(self.sess,tf_model_path)

    # def init_op(self):
    #     print("函数：init_op start")
    #     self.sess=tf.Session(graph=self.graph)
    #     self.sess.run(self.init)
    #     self.summary_writer=tf.summary.FileWriter(self.logdir,self.sess.graph)
    #     print("函数：init_op end")



    def build_graph(self):
        self.graph = tf.Graph()
        valid_size = 2  # Random set of words to evaluate similarity on.
        valid_window = 2  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        with self.graph.as_default():
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32,shape=[self.batch_size,self.num_skips])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            #paragraph vector place holder
            self.train_para_labels = tf.placeholder(tf.int32,shape=[self.batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            with tf.device('/cpu:0'):
                # Ops and variables pinned to the CPU because of missing GPU implementation
                # with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]: [200, 4] + [vocab_size - 1, embedding_size]
                embed_word = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
                print("embed_word:",embed_word)

                self.para_embeddings = tf.Variable(
                    tf.random_uniform([self.paragraph_size, self.embedding_size], -1.0, 1.0))
                embed_para = tf.nn.embedding_lookup(self.para_embeddings, self.train_para_labels)

                embed = tf.concat([embed_word, embed_para],1)

                reduced_embed = tf.div(tf.reduce_sum(embed, 1), self.skip_window*2 + 1)


                # Construct the variables for the NCE loss
                self.nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    inputs=reduced_embed,
                    labels=self.train_labels,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size
                )
            )

            # tensorboard 相关记录
            tf.summary.scalar('loss',self.loss)


            # Construct the SGD optimizer using a learning rate of 1.0.

            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate =self.learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   1000, 0.009, staircase=True)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / norm

            para_norm = tf.sqrt(tf.reduce_sum(tf.square(self.para_embeddings), 1, keepdims=True))
            self.normalized_para_embeddings = self.para_embeddings / para_norm

            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)

            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)
            # 变量初始化
            self.init = tf.global_variables_initializer()

    # word_index=0
    # sentence_index=0

    def generate_DM_batch(self):
        # word_index = 0
        # sentence_index = 0
        global word_index
        global sentence_index
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size, self.num_skips), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        para_labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32) # Paragraph Labels
        span = 2 * self.skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        # print("buffer:",buffer)
        for _ in range(span):
            # print("data[sentence_index].words[word_index]",data[sentence_index].words[word_index])
            buffer.append(data[sentence_index].words[word_index])
            # print(" buffer.append:", buffer)
            sen_len = len(data[sentence_index].words)
            # print("sen_len",sen_len)
            # print("word_index:",word_index)
            if sen_len - 1 == word_index: # reaching the end of a sentence
                word_index = 0
                sentence_index = (sentence_index + 1) % len(data)
            else: # increase the word_index by 1
                word_index += 1
        # print("2sen_len",sen_len)
        # print("2word_index",word_index)
        # print("2sentence_index",sentence_index)
        # print("2buffer",buffer)
        for i in range(self.batch_size):
            target = self.skip_window  # target label at the center of the buffer
            # print("target",target)
            targets_to_avoid = [ self.skip_window ]
            # print("targets_to_avoid",targets_to_avoid)
            batch_temp = np.ndarray(shape=(self.num_skips), dtype=np.int32)
            # print("batch_temp",batch_temp)
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    # print("target_before:",target)
                    target = random.randint(0, span - 1)
                    # print("target2:",target)
                targets_to_avoid.append(target)
                # print("targets_to_avoid",targets_to_avoid)
                # print("buffer2:",buffer)
                batch_temp[j] = buffer[target]
            # print("batch_temp",batch_temp)
            batch[i] = batch_temp
            # print("batch33",batch[i])
            labels[i,0] = buffer[self.skip_window]
            # print("labels33",labels[i,0])
            para_labels[i, 0] = sentence_index
            # print("para_labels",para_labels[i, 0])
            buffer.append(data[sentence_index].words[word_index])
            sen_len = len(data[sentence_index].words)
            if sen_len - 1 == word_index: # reaching the end of a sentence
                word_index = 0
                sentence_index = (sentence_index + 1) % len(data)
            else: # increase the word_index by 1
                word_index += 1
        # print("batch",batch)
        # print("labels",labels)
        # print("para_labels",para_labels)
        return batch, labels, para_labels

    def train_batch(self,num_steps = 1):
        # Add variable initializer.
        # init = tf.initialize_all_variables()
        # self.sess = tf.Session(graph=self.graph)
        # #     self.sess.run(self.init)
        # Step 5: Begin training.
        with tf.Session(graph=self.graph) as session:
            # We must initialize all variables before we use them.
            session.run(self.init)
            print("Initialized")

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels, batch_para_labels = self.generate_DM_batch()
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.train_para_labels: batch_para_labels}
                # print("feed_dict",feed_dict)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 1 == 0:
                    if step > 0:
                        average_loss /= 1
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                # if step % 1 == 0:
                #     sim = self.similarity.eval()
                #     for i in xrange(valid_size):
                #         valid_word = reverse_dictionary[valid_examples[i]]
                #         top_k = 8  # number of nearest neighbors
                #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #         log_str = "Nearest to %s:" % valid_word
                #         for k in xrange(top_k):
                #             close_word = reverse_dictionary[nearest[k]]
                #             log_str = "%s %s," % (log_str, close_word)
                #         print(log_str)
            final_embeddings = self.normalized_embeddings.eval()
            final_para_embeddings = self.normalized_para_embeddings.eval()
        return final_embeddings,final_para_embeddings
            # print("final_embeddings",final_embeddings)
            # print("final_para_embeddings:",final_para_embeddings)

def test_vec(reverse_dictionary,final_embeddings,final_para_embeddings,top_k = 5,doc_id = 2):
    # Testing final embedding
    input_dictionary = dict([(v, k) for (k, v) in reverse_dictionary.items()])
    # print("input_dictionary",input_dictionary)

    test_word_idx_a = np.random.randint(0, len(input_dictionary) - 1)
    # print("test_word_idx_a:",test_word_idx_a)
    a = final_embeddings[test_word_idx_a, :]
    # print("a",a)
    # print("final_embeddings",final_embeddings)
    similarity = final_embeddings.dot(a)                    #测试词的相关性
    # print("similarity",similarity)
    nearest = (-similarity).argsort()[0:top_k]
    # print("nearest",nearest)
    for k in range(top_k):
        close_word = reverse_dictionary[nearest[k]]
        # print(close_word)

    # doc_id = 2

    para_embedding = final_para_embeddings[doc_id, :]
    print(doc_id,para_embedding)
    similarity_para = final_para_embeddings.dot(para_embedding)
    nearest_para = (-similarity_para).argsort()[0:top_k]
    for k in range(top_k):
        close_sen = all_docs[nearest_para[k]]
        print(close_sen)


if __name__=='__main__':
    all_docs=read_data("F:\python3_pj\doc2vec\Knowledge.txt")
    data, count, dictionary, reverse_dictionary=build_dataset(all_docs,min_cut_freq=1)
    word_index = 0
    sentence_index = 0
    d2v=doc2vec(
        vocab_list=reverse_dictionary,
        doc_list=all_docs,
        batch_size=4,
        embedding_size=8,  # Dimension of the embedding vector.
        skip_window=2,  # How many words to consider left and right.
        num_sampled=10,  # 负采用个数
        num_skips=2,  # How many times to reuse an input to generate a label.
        learning_rate=0.8,
        logdir='tmp'
    )
    final_embeddings,final_para_embeddings=d2v.train_batch(num_steps=1)
    test_vec(reverse_dictionary,final_embeddings,final_para_embeddings)










