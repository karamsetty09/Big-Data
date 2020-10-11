"""
Student Name: Venkata Sandeep Kumar Karamsetty
Student ID: 6228975
Student Mail ID: vskk033@uowmail.edu.au
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import itertools
import scipy.sparse as sps

#Confusion Matrix function

def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize = (20,10)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid('false')
    plt.show()
   
os.chdir("C:/Users/venka/Desktop/semester - 3/Big data Analytics/Assignment 2/PythonApplication1/PythonApplication1")

#Read the data
trainingData = pd.read_table('train.data',sep=" ",names=['docId', 'wordId', 'count'])
print(trainingData)
testingData = pd.read_table('test.data',sep=" ",names=['docId', 'wordId', 'count'])
print(testingData)
traininglabel = pd.read_table('train.label',sep=" ",names=['labelId'])
print(traininglabel)
testinglabel = pd.read_table('test.label',sep=" ",names=['labelId'])
print(testinglabel)
trainingmap = pd.read_table('train.map',sep=" ",names=['labelName','labelId'])
print(trainingmap)
testingmap = pd.read_table('test.map',sep=" ",names=['labelName','labelId'])
print(testingmap)

#Merging the training and testing sets
traininglabel['docId'] = list(range(1,11270))
train_label = traininglabel.merge(trainingmap, on="labelId")
train_final = train_label.merge(trainingData, on="docId")
train_final = train_final[["docId","wordId","count","labelName"]]
print("Merged training data shown below:")
print(train_final)

testinglabel['docId'] = list(range(1,7506))
test_label= testinglabel.merge(testingmap, on="labelId")
test_final = test_label.merge(testingData, on="docId")
test_final = test_final[["docId","wordId","count","labelName"]]
print("Merged testing data shown below:")
print(test_final)


#Document matrix for testing and training
mat = sps.coo_matrix((trainingData["count"].values, (trainingData["docId"].values-1, trainingData["wordId"].values-1)))
training_data_matrix = mat.tocsc()
training_data_matrix.shape 
print(training_data_matrix)

test_mat  = sps.coo_matrix((testingData["count"].values, (testingData["docId"].values-1, testingData["wordId"].values-1)))
testing_data_matrix = test_mat.tocsc()
testing_data_matrix = testing_data_matrix[:,:training_data_matrix.shape[1]] 
print(testing_data_matrix)

#Naive Bayes classifier
naiveBayesclassifier = MultinomialNB(alpha=.01, class_prior=None, fit_prior=True)
naiveBayesclassifier.fit(training_data_matrix, traininglabel["labelId"])
print("naive Bayes Classifier")
print(naiveBayesclassifier)

# Predict the test results
prediction = naiveBayesclassifier.predict(testing_data_matrix)
print(prediction)
#Printing accuracy
print("Accuracy = {0}".format(metrics.f1_score(testinglabel["labelId"], prediction, average='macro')* 100))

#Confusion Matrix
cm = confusion_matrix(testinglabel["labelId"], prediction)
plot_confusion_matrix(cm, classes=trainingmap["labelName"],title='Confusion matrix')

