# Task 3
# Stendent Number: 6228975
# student Name: Venkata Sandeep Kumar Karamsetty
# Student ID: vskk033@uowmail.edu.au


#load library
# use below if not installed
#install.packages('arules') 
#install.packages('arulesViz')
library('arules')
library('arulesViz')

#setting current directory to working directory

setwd('./')

#reading .csv file

DataSet <- read.csv("./A1_success_data.csv")

#Data Exploration

class(DataSet)
typeof(DataSet)
head(DataSet)
summary(DataSet)

#convert the data frame to transactions so we can use it in apriori algorithm

transactions_of_Dataset <- as(DataSet, 'transactions')


summary(transactions_of_Dataset)
transactions_of_Dataset

# 1.) Generate frequent itemsets by applying various support thresholds and inspect these itemsets by displaying their 
# support, confidence, and lift values

#support threshold at 0.01
itemset1 = apriori(transactions_of_Dataset, parameter = list(support = 0.01, confidence = 0.3))
summary(itemset1@quality)
itemset1
#Inspect the itemsets by displaying the property support, confidence, and lift
inspect(head(sort(itemset1, by = "support"), 10))

#support threshold at 0.02
itemset2 = apriori(transactions_of_Dataset, parameter = list(support = 0.02,confidence = 0.6))
summary(itemset2@quality)
itemset2
inspect(head(sort(itemset2, by = "support"), 10))

#support threshold at 0.6
itemset3 = apriori(transactions_of_Dataset, parameter = list(support = 0.6,confidence = 0.8,minlen = 1,maxlen = 8))
summary(itemset3@quality)
itemset3
inspect(head(sort(itemset3, by = "support"), 10))



rhs = apriori(transactions_of_Dataset, parameter = list(support = 0.02,confidence = 0.6),
              appearance = list(rhs = c('Success=No', 'Success=Yes'), default = 'lhs'))

inspect(head(sort(rhs, by = 'lift'), 15))

#3.) - Visualize the rules generated in the last step by 
#3-1) showing the relationship among support, confidence and lift

plot(rhs@quality)

#3-2) using the graph visualization based on the sorted lift value.

lift <- head(sort(rhs, by = "lift"), 5)
plot(lift, method = "graph", control = list(type = "items"))


