from sklearn.linear_model.logistic import  *
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import tree

'''LR模型分类训练'''
def LR_model(x_train, x_dev, y_train, y_dev):

    print('LogisticRegression Model:')
    classifier = LogisticRegression()

    classifier.fit(x_train, y_train)
    #
    # with open('LR_model.pkl', 'wb') as f:
    #     pickle.dump(classifier, f)

    print("Train accuracy:")
    print(classification_report(y_train, classifier.predict(x_train)))
    print('Test Accuracy')
    print(classification_report(y_dev, classifier.predict(x_dev)))


''' 决策树模型 '''
def DecisionTree_model(x_train, x_dev, y_train, y_dev):

    print('DCTree Model:')
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    print("Train accuracy:")
    print(classification_report(y_train, classifier.predict(x_train)))
    print('Test Accuracy')
    print(classification_report(y_dev, classifier.predict(x_dev)))


''' 随机森林 '''
def RandomForest_model(x_train, x_dev, y_train, y_dev):
    print('Random Forest Model:')
    # 随机森林
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
    classifier.fit(x_train, y_train)

    print("Train accuracy:")
    print(classification_report(y_train, classifier.predict(x_train)))
    print('Test Accuracy')
    print(classification_report(y_dev, classifier.predict(x_dev)))


''' 朴素贝叶斯 '''
def GaussianNB_model(x_train, x_dev, y_train, y_dev):
    print('Naive Bayes Model:')

    classifier = GaussianNB()  # 默认priors=None
    classifier.fit(x_train, y_train)

    print("Train accuracy:")
    print(classification_report(y_train, classifier.predict(x_train)))
    print('Test Accuracy')
    print(classification_report(y_dev, classifier.predict(x_dev)))

from sklearn import svm
def SVM_model(x_train, x_dev, y_train, y_dev):
    print('SVM Model:')
    classifier = svm.SVC()
    classifier.fit(x_train, y_train)

    print("Train accuracy:")
    print(classification_report(y_train, classifier.predict(x_train)))
    print('Test Accuracy')
    print(classification_report(y_dev, classifier.predict(x_dev)))