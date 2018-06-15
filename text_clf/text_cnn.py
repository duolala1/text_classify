import tensorflow as tf
import numpy as np
''' 
该脚本提供CNN模型用于文本分类
'''

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    sequence_length是单个句子向量的长度（所有句子向量具有相同长度）,这个向量是id序列表示矩阵中的向量[1,0,2...]这样
    vocab_size是所有不重复单词数量，即词典词汇数目
    embedding_size = 128
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        print(sequence_length,"numc",num_classes,'vb_size:',vocab_size,'eb_size:',embedding_size,'num_filt',num_filters)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        ''' 
        这一层将ID序列表示的文档信息转化成word2vwc形式，将词向量降到一个较小的维度
        这里没有用已有的embedding层参数，而是初始化为随机变量后训练得到word2vec参数
        '''
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            ''' 
            有vocab_size个embedding_size长度的向量，每个向量代表一个词语
            vocab_size是词典长度（不重复单词数量）
            向量中数值分布在-1到1之间
            也就是给词典中每个词语一个embedding_size长度的向量来表示该词语
            如:
            embedding_size = 3 
            词典：
            1: today : [0.2,0.3,0.9]
            2: go : [0.3,0.2,0.7]
            ...
            原文档 id 向量 [1,2,1]
            -> [[0.2,0.3,0.9],[0.3,0.2,0.7],[0.2,0.3,0.9]]
            因为原文档向量长度是固定的sequence_length
            因此生成的word2vec尺寸也是固定的：
            文档数目 * sequence_length行 * embedding_size列
            
            然后每个id会由lookup方法查找到一个向量对应这个id的词语，这些向量组合在一起就代表一篇文档
            
            word2vec方法给每个单词的编码长度比one-hot等编码方式长度要小（维度低），更为稠密（0少），
            同时更考虑了单词前后关系，这样就尽可能的将意思相近的词语在词向量表达后的空间上非常接近
            （如巧克力、朱古力两个词语被表达为128维度的词向量时两个向量（点）距离相对其他类别单词更近），
            非常适合短文本分析
            '''
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            '''
            tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
            tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
            '''
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 输出(?, 56, 128)
            # expand_dims增加维度，如shape:  [2] -> [2,1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # 输出(?, 56, 128, 1)
            print(self.embedded_chars_expanded.shape)
        # Create a convolution + maxpool layer for each filter size
        '''
        输入样本矩阵为：
        这里论文中有个很好的设计
        使用不同size的卷积核（3，4，5）
        embedding_size = 128 
        num_filters = 128
        这样能够读取不同个数量的连续单词
        感受野不同，提取的特征不同，类似N元语言模型，不过这里可以有很多个N元模型
        但只有一层
        最终将结果汇集在一起输出 pooled_outputs
        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                '''
                卷积核宽度要和embedding_size相同，即对一个词语取整个词向量
                因为如果只取一部分词向量的话没有意义
                行数为3,4,5
                '''
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                '''
                卷积输出结果为 (equence_length - filter_size + 1)行 * 1 列 * num_filters（厚度128）
                3:(?, 54, 1, 128)4:(?, 53, 1, 128)5:(?, 52, 1, 128)
                '''
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                ''' 
                注意这里conv没有用same参数，就是没有补0（补0意义就有变化了）
                因此size = 原长 - filtersize + 1（stride = 1）
                '''
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                '''
                pooled : 下采样在整个列上只取得一个最大值作为特征
                因此只有num_filters个特征(?, 1, 1, 128)
                '''
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        ''' 
        pooled_outputs是list类型：
        [<tf.Tensor 'conv-maxpool-3/pool:0' shape=(?, 1, 1, 128) dtype=float32>, <tf.Tensor 'conv-maxpool-4/pool:0' shape=(?, 1, 1, 128) dtype=float32>, <tf.Tensor 'conv-maxpool-5/pool:0' shape=(?, 1, 1, 128) dtype=float32>]
        因此需要将三个用concat方法结合在一起变为：(?, 1, 1, 384)
        '''
        self.h_pool = tf.concat(pooled_outputs, 3)
        print(self.h_pool.shape)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            ''' 
            自己加着玩的，又加了个全连接层
            对实际结果输出影响不大 
            '''
            fc1w = tf.get_variable('fc1w', shape=[num_filters_total, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc1b = tf.get_variable('fc1b', shape=[512], initializer= tf.constant_initializer(0))
            fc1  = tf.matmul(self.h_drop, fc1w) + fc1b

            # 全连接层到输出 w [384*2]
            W = tf.get_variable(
                "W",
                shape=[512, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(fc1, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            '''
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            '''
        # 多分类计算交叉熵
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 计算准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
