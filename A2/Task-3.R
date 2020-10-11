# Task - 3
#Name: Venkata Sandeep Kumar Karamsetty
#Student ID: 6228975 
#Mail ID: vskk033@uowmail.edu.au

#import libraries
#install.packages("twitteR")
#install.packages(c('ROAuth','RCurl'))
#install.packages("dplyr")
#install.packages('purrr')
#install.packages('caret')
#install.packages('RTextTools')
#install.packages('maxent')
#install.packages('sos')
#install.packages(c('ROAuth','RCurl'))
#install.packages('RTextTools')
library(sos)
library(twitteR)
library('ROAuth')
library('RCurl')
library('dplyr')
library(purrr)
library(plyr)
library(stringr)
library(tm)
library(e1071)
library(caret)
library(RTextTools)
library(maxent)

library(twitteR)
library(RCurl)
#twitter authentication
consumerAPIKey <- "4XsqFw3PwDZWRNUtyF5WSuFlz"
consumerAPISecretKey <- "CuOT4iPxlyoMuk5cconiUpF0eX8e7LQ1ltxs27XbmYWCNJ8b6O"
accessTokenKey <- "1179219778579226630-hFXeJSppsWWY7JllGrtjOmkjvMUQap"
accessTokenSecretKey <- "utFFVPO8J711V6JL1JCJx0HNFgjqyCKQdniCLqBWspwmz"
reqURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
setup_twitter_oauth(consumerAPIKey,consumerAPISecretKey,accessTokenKey,accessTokenSecretKey)
TrumpTweets<- searchTwitter("@realDonaldTrump",n=200,lang="en")
TrumpTweets
#convert to data frame
TrumpTweets <- twListToDF(TrumpTweets)
#write to csv file and save the tweets - change file location
write.csv(Tweets,file="twitterList.csv")
TrumpResults<- read.csv("twitterList.csv")

# count the tagging done in file
countTagging <- table(TrumpResults$Tagging)
countTagging

#Corpus to know details in file
corpusVector <- Corpus(VectorSource(TrumpResults$text))
corpusVector

#pre-processing the tweets
Tweets_Pre_Processed <- tm_map(corpusVector,  content_transformer(tolower))
url <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
Tweets_Pre_Processed <- tm_map(Tweets_Pre_Processed, url)
Tweets_Pre_Processed <- tm_map(Tweets_Pre_Processed, removePunctuation)
Tweets_Pre_Processed <- tm_map(Tweets_Pre_Processed, stripWhitespace)
Tweets_Pre_Processed <- tm_map(Tweets_Pre_Processed, removeWords, stopwords("english"))
Tweets_Pre_Processed <- tm_map(Tweets_Pre_Processed, removeNumbers)
Tweets_Pre_Processed <-tm_map(Tweets_Pre_Processed,stemDocument)
inspect(Tweets_Pre_Processed[1:10])
#Extracting Document term matrix from pre_processed_data
DTM <- DocumentTermMatrix(Tweets_Pre_Processed)
DTM
#partitioning of training and testing data
training_Data<- TrumpResults$Tagging[1:160] #trainig set
training_Data
test_Data <- TrumpResults$Tagging[161:200] #test set
test_Data
#document term matrix TF-IDF
DTM_TRAIN_TFIDF <- DocumentTermMatrix(Tweets_Pre_Processed, control = list(weighting = weightTfIdf))
DTM_TRAINING_SET <- DTM_TRAIN_TFIDF[1:160,]
DTM_TRAINING_SET

DTM_TEST_SET <- DTM_TRAIN_TFIDF[161:200,]
DTM_TEST_SET

# use the NB classifier with Laplace smoothing
NAIVE_BAYES_RESULT <- naiveBayes(as.matrix(DTM_TRAINING_SET), training_Data, laplace=1)
NAIVE_BAYES_RESULT

# predict with testdata
PREDICTED_RESULT <- predict (NAIVE_BAYES_RESULT,as.matrix(DTM_TEST_SET))
PREDICTED_RESULT

#Confusion Matrix for Naive Bayes
xtab <- table( "Actual" = test_Data, "Predictions"= PREDICTED_RESULT)

#xtab
confMatrixNB <- confusionMatrix(xtab)
confMatrixNB


#apply svm and confusion matrix matrix
SVM_TRAIN <- svm(as.matrix(DTM_TRAINING_SET), training_Data, type='C-classification',kernel="radial", cost=10, gamma=0.5)
SVM_TRAIN
#Prediction Result
SVM_PREDICT <- predict(SVM_TRAIN, as.matrix(DTM_TEST_SET))
SVM_PREDICT

#Confusion Matrix
xSVM <- table("Actual" = test_Data, "Predictions"= SVM_PREDICT)
xSVM
confMatrixSVM <- confusionMatrix(xSVM)
confMatrixSVM


#Performance for Naive Bayes - Precison, Recall, Fmeasure
n1 = sum(xtab) # number of instances
nc1 = nrow(xtab) # number of classes
diag1 = diag(xtab) # number of correctly classified instances per class 
rowsums1 = apply(xtab, 1, sum) # number of instances per class
colsums1 = apply(xtab, 2, sum) # number of predictions per class
p1 = rowsums1 / n1 # distribution of instances over the actual classes
q1 = colsums1 / n1 # distribution of instances over the predicted classes

accuracy1 = sum(diag1) / n1 

precisionNB = diag1 / colsums1 
recallNB = diag1 / rowsums1 
f1NB = 2 * precisionNB * recallNB / (precisionNB + recallNB)
data.frame(precisionNB, recallNB, f1NB, accuracy1) 



#Performance for SVM - Precison, recall, fmeasure

n2 = sum(xSVM) # number of instances
nc2 = nrow(xSVM) # number of classes
diag2 = diag(xSVM) # number of correctly classified instances per class 
rowsums2 = apply(xSVM, 1, sum) # number of instances per class
colsums2 = apply(xSVM, 2, sum) # number of predictions per class
p2 = rowsums2 / n2 # distribution of instances over the actual classes
q2 = colsums2 / n2 # distribution of instances over the predicted classes

accuracy2 = sum(diag2) / n2 

precisionSVM = diag2 / colsums2 
precallSVM = diag2 / rowsums2 
f1SVM = 2 * precisionSVM * precallSVM / (precisionSVM + precisionSVM) 
data.frame(precisionSVM, recallSVM, f1SVM, accuracy2) 

