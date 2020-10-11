# Task 1
# Stendent Number: 6228975
# student Name: Venkata Sandeep Kumar Karamsetty
# Student ID: vskk033@uowmail.edu.au

# Reading file or data
# For setting current directing to working directory.
setwd("./")

# Reading of .CSV file
performancedata <- read.csv("./A1_performance_test.csv")

# Data Exploration of performance data collected.
class(performancedata)
typeof(performancedata)
summary(performancedata)
plot(performancedata)

# Slecting anova test as number of parameters > 3
anova <- aov(performance ~ approach, data = performancedata)

#summary of Annova Results
summary(anova)

#Calculate mean to check better approach
mean(performancedata$approach == "approach1")
mean(performancedata$approach == "approach2")
mean(performancedata$approach == "no_approach")

# Function module to perform pair-wise tests for difference of means.
TukeyHSD(anova)
