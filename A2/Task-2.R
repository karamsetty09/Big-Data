#Task 2 -
#Name: Venkata Sandeep Kumar Karamsetty
#Student ID: 6228975 
#Mail ID: vskk033@uowmail.edu.au

#install.packages('XML')
#install.packages("tm.corpus.Reuters21578", repos = "http://datacube.wu.ac.at")
#install.packages("wordcloud")
#install.packages('topicmodels')
require(topicmodels)
library(tm.corpus.Reuters21578)
library(wordcloud)
library(lda)
library(ggplot2)
library(reshape2)

data(Reuters21578)

#Summarizing the data of Reuters 
summary(Reuters21578)

#observation on reuters
cat(content(Reuters21578[[7]]))
cat(content(Reuters21578[[3]]))

#Pre-Processing the Reuters-21578
pre_processing_reuters <- tm_map(Reuters21578,  content_transformer(tolower))
pre_processing_reuters <- tm_map(pre_processing_reuters, stripWhitespace)
pre_processing_reuters <- tm_map(pre_processing_reuters, removeWords, stopwords("english"))
pre_processing_reuters <- tm_map(pre_processing_reuters, removePunctuation)
pre_processing_reuters <- tm_map(pre_processing_reuters, removeNumbers)
pre_processing_reuters <-tm_map(pre_processing_reuters,stemDocument)
pre_processing_reuters

#document term matrix - Bag Of Words(BOW)
DTM <- DocumentTermMatrix(pre_processing_reuters)
DTM

#document term matrix TF-IDF
DTM1 <- DocumentTermMatrix(pre_processing_reuters, control = list(weighting = weightTfIdf))
DTM1

#reducing the dimensions of the DTM 
#By removing the less frequent terms such that the sparsity is less than 0.95
REVISED_DTM = removeSparseTerms(DTM1, 0.99)
REVISED_DTM
#frequency - Simple word cloud
x <- findFreqTerms(DTM,15)
x


#LDA
# due to vocabulary pruning, we have empty rows in our dtm 
#LDA does not like this. So we remove those docs from the dtm and the metadata
selection_id <- slam::row_sums(DTM) > 0
DTM <- DTM[selection_id, ]

#number of topics
k <- 20

# compute the LDA model, inference via 25 iterations of Gibbs sampling
result <- LDA(DTM, k, method="Gibbs", control=list(iter = 25, verbose = 25, alpha = 0.1))
result

#Let's take a look at the 10 most likely terms within the term probabilities beta of the inferred topics.
terms(result, 10)

#five most likely terms of each topic to a string
top5termsPerTopic <- terms(result, 5)
nameOfTopics <- apply(top5termsPerTopic, 2, paste, collapse=" ")
nameOfTopics
# have a look a some of the results (posterior distributions)
tmposterior <- posterior(result)

# format of the resulting object
attributes(tmposterior)

#Let us first take a look at the contents of eight sample documents:
examples <- c(2, 150, 250, 450, 850, 1050, 1250, 1450)
lapply(pre_processing_reuters[examples], as.character)

theta <- tmposterior$topics
theta
N <- length(examples)
N

# get topic proportions form example documents
tpExamples <- theta[examples,]
colnames(tpExamples) <- nameOfTopics
vizDataFrame <- melt(cbind(data.frame(tpExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
vizDataFrame


#ggplot
ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
  coord_flip() +
  facet_wrap(~ document, ncol = N)
