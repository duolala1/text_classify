CNN短文本分类效果和传统模型比较
===
train.py <br> 
---
训练CNN模型对文本进行分类， 过程可视化表格保存在run文件夹下，可用tensorboard查看<br> 
-bayes_test.py 负责训练和测试传统模型在该数据集下的工作效果（包括SVM，Naive Bayes和LR模型等）<br> 
text_cnn.py<br> 
---
保存cnn模型（tensorflow实现）<br> 
tradition_models.py<br> 
---
保存传统模型（用scikit-learn实现）<br> 

-data_helper.py<br> 
---
负责文本的分词、去除符号等工作<br> 

最终CNN在短文本分类效果达到75%左右正确率<br>

短文本分类首先用正则等方法去除不正规字词
编码方式：先建立中文词汇表和把文本转为词ID序列，
调用tensorflow中 learn.preprocessing.VocabularyProcessor
该方法和词袋等方法不一样，该方法需要先定义文档的最大长度（就是认为该数据集中文档最多能有多少个词）maxLength， 然后后续生成的所有文档向量长度都为maxLength，
如果文档i长度小于maxLength，则后续用0填充，如果比它大，后续的会被剪切掉，
然后每个不重复的单词作为一个词典，如
{1:'today',2:'to',3:'go',...}
对应一句话：'today go to'，如果最大长度为5，
该文档编码为 [1 3 2 0 0] 
这样不统计词频，但保留原始文档词汇和顺序信息。

然后将ID序列表示的文本转换为word2vec形式，将每个词向量降维到embedding_size大小，生成一个word2vec的矩阵（总的词典中词语数量行 * embedding_size），矩阵中每行是对应每个id词汇的word2vec向量表达。
因此只需要将id序列矩阵中的id在其中lookup所述的行，如id ：1 ->[0.1,0.3,...]把这个向量放入原先的文档矩阵中，每个文档就变成了
maxLength行*embedding_size列的矩阵。
由于词向量在embedding过程后长度统一，并且原文档编码长度也统一(maxLength)，因此每个文档都是一个maxLength行*embedding_size列的矩阵。

接下来对这个矩阵应用卷积层，这里只需要一层卷积层，但是单层卷积层中应用多个卷积核:（3, 4, 5）行数 * embedding_size列数*1的卷积核尺寸，列数要和embedding_size相同是因为如果小于这个数，每次卷积核计算卷积的都不是完整的一个词向量，就没有意义；另外这里padding = ‘VALID’，因为如果补0的话会影响结果输出；不同大小的卷积核感受野不同，类比于N元语言模型，这里一次能读入3、4、5个词语，提取出不同特征，卷积后输出
(equence_length - filter_size + 1)行 * 1 列 * num_filters（厚度128）
然后输入池化层下采样，下采样时kernel行数即为(equence_length - filter_size + 1)，采用max_pooling，也就是说对一个卷积核的输出，只取一个最大值作为特征值，一共有3个不同大小卷积核，每个大小有filter_num个卷积核数量，因此pooling后输出3个1 * 1 * filter_num特征向量，然后调用tf.concat方法将三个连在一起，特征数目为3 * filter_num。

接下来直接连接一个全连接层，W 为[ 3 * filter_num , class_num]大小的矩阵，输出分类预测

分类效果最高75%，平均72%左右

将该方法同传统方法对比：
总样本数目10600左右（2:8分割测试集训练集）
LogisticRegression Model:
Train accuracy:
precision recall f1-score support

0 0.96 0.98 0.97 1055
1 0.98 0.96 0.97 1078

avg / total 0.97 0.97 0.97 2133

Test Accuracy
precision recall f1-score support

0 0.72 0.71 0.71 4276
1 0.71 0.72 0.72 4253

avg / total 0.72 0.72 0.72 8529

DCTree Model:
Train accuracy:
precision recall f1-score support

0 1.00 1.00 1.00 1055
1 1.00 1.00 1.00 1078

avg / total 1.00 1.00 1.00 2133

Test Accuracy
precision recall f1-score support

0 0.57 0.59 0.58 4276
1 0.57 0.56 0.56 4253

avg / total 0.57 0.57 0.57 8529

Random Forest Model:
Train accuracy:
precision recall f1-score support

0 0.93 0.51 0.66 1055
1 0.67 0.96 0.79 1078

avg / total 0.80 0.74 0.72 2133

Test Accuracy
precision recall f1-score support

0 0.72 0.36 0.48 4276
1 0.57 0.86 0.69 4253

avg / total 0.64 0.61 0.58 8529

Naive Bayes Model:
Train accuracy:
precision recall f1-score support

0 0.99 1.00 0.99 1055
1 1.00 0.99 0.99 1078

avg / total 0.99 0.99 0.99 2133

Test Accuracy
precision recall f1-score support

0 0.66 0.67 0.66 4276
1 0.66 0.65 0.65 4253

avg / total 0.66 0.66 0.66 8529
SVM Model:
Train accuracy:
precision recall f1-score support

0 0.00 0.00 0.00 1055
1 0.51 1.00 0.67 1078

avg / total 0.26 0.51 0.34 2133

可以看出在MovieReviews这样二类别短文本分类工作上
CNN明显优于传统方法（81%左右）。
SVM训练时间明显长于其他方法，且效果较差（不知是不是kernel不太对）
决策树方法很容易就过拟合，训练集达到了100%左右的正确率，但测试集正确率仅仅有57%左右正确率；
随机森林略好于决策树方法，过拟合现象不严重但也只有64%左右正确率
朴素贝叶斯方法和随机森林结果相似，有66%的正确率
最好的是逻辑回归算法，达到了72%左右正确率，十分接近CNN效果
