'''
author : fangtao
该脚本在Movie Reviews 数据集下
测试传统方法在短文本分类上的工作效果

'''
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from gensim import *
from scipy.sparse import csr_matrix
from sklearn.linear_model.logistic import  *
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from tradition_models import *

# Data loading params

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

'''
该脚本生成的是tfidf编码
'''
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    # 并且已经去除停用词了
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # 去掉停用词
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in x_text]

    dictionary = corpora.Dictionary(texts)  # 生成词典

    # 将文档存入字典，字典有很多功能，比如
    # diction.token2id 存放的是单词-id key-value对
    # diction.dfs 存放的是单词的出现频率
    dictionary.save('deerwester.dict')  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use

    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    corpus_lsi = lsi_model[corpus_tfidf]

    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_tfidf:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1

    tfidf_matrix = csr_matrix((data, (rows, cols))).toarray()

    # ''' 打乱样本'''
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = tfidf_matrix[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    labels = []
    for ele in y_shuffled:
        if ele[0] == 0 and ele[1] == 1:
            labels.append(1)
        else:
            labels.append(0)

    #
    # ''' 分割测试集和训练集 '''
    # # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.8 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = labels[:dev_sample_index], labels[dev_sample_index:]

    return x_train, y_train, x_dev, y_dev

x_train, y_train, x_dev, y_dev = preprocess()

LR_model(x_train, x_dev, y_train, y_dev)
DecisionTree_model(x_train, x_dev, y_train, y_dev)
RandomForest_model(x_train, x_dev, y_train, y_dev)
GaussianNB_model(x_train, x_dev, y_train, y_dev)
# SVM_model(x_train, x_dev, y_train, y_dev)