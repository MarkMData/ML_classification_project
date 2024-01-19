library(tidyverse)
library(caret)
library(glmnet)
library(GGally)
library(corrplot)
library(gridExtra)
library(vtable)
library(MASS)
library(class)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Pre-processing and exploratory analysis ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

drug <- read.csv("ML_classification_project_data.csv")
# Removing index column
drug <- drug[,-1]

# Looking for NA's or missing values
drug[!complete.cases(drug), ]
drug[is.na(drug)]

# Looking at data structure
str(drug)

# Data summary
summary(drug)

# Setting class to a factor variable
drug$Class <- as.factor(drug$Class)

# Splitting data into training, validation, and testing sets while maintaining
# distribution of response variable
set.seed(123)
trainIndex <- createDataPartition(drug$Class, p = .5, 
                                  list = FALSE, 
                                  times = 1)
train <- drug[trainIndex,]
valdTest <- drug[-trainIndex,]
set.seed(123)
valdIndex <- createDataPartition(valdTest$Class, p = .6, 
                                 list = FALSE, 
                                 times = 1)
valid <- valdTest[valdIndex,]
test <- valdTest[-valdIndex,]

# Checking the distribution of the response in the three datasets
summary(train$Class)
summary(valid$Class)
summary(test$Class)

# Exploratory analysis on training data
# Looking at pairs plots of predictors with class as a factor
ggpairs(train, columns = c(1:11), mapping = aes(color = Class),
        upper = list(continuous = wrap(
          "cor", stars = FALSE, digits = 2 , title = "Cor", size = 2.5)),
        lower = list(continuous = wrap(
          "points", alpha = 0.7, size=0.6)),
        axisLabels = "none")


# boxplots for predictors with class as factor
plt.1 <- ggplot(train, aes(x=Class, y = Age)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.2 <- ggplot(train, aes(x=Class, y = Education)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.3 <- ggplot(train, aes(x=Class, y = X.Country)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.4 <- ggplot(train, aes(x=Class, y = Ethnicity)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.5 <- ggplot(train, aes(x=Class, y = Nscore)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.6 <- ggplot(train, aes(x=Class, y = Escore)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.7 <- ggplot(train, aes(x=Class, y = Oscore)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.8 <- ggplot(train, aes(x=Class, y = Ascore)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.9 <- ggplot(train, aes(x=Class, y = Cscore)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.10 <- ggplot(train, aes(x=Class, y = Impulsive)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
plt.11 <- ggplot(train, aes(x=Class, y = SS)) +
  geom_boxplot(varwidth = TRUE) + theme(axis.title.x=element_blank())
grid.arrange(plt.1, plt.2, plt.3, plt.4, plt.5, plt.6, plt.7, plt.8, plt.9,
             plt.10, plt.11, ncol=3, nrow =4)


# Centering and scaling
preProc <- preProcess(train, method = c("center", "scale"))
train <- predict(preProc, train)
valid <- predict(preProc, valid)
test <- predict(preProc, test)

summary(train)


# Creating function to assess model accuracy, specificity and sensitivity
# on validation data

validClass <- function(x){
  print(paste("Accuracy: ", round(mean(x == valid$Class), digits = 3)))
  print(paste("Sensitivity: ",
              round(sweep(table(valid$Class, x),
                          1, apply(table(valid$Class, x), 1, sum),
                          "/")[2,2],
                    digits = 3)))
  print(paste("Specificity: ",
              round(sweep(table(valid$Class, x),
                          1, apply(table(valid$Class, x), 1, sum),
                          "/")[1,1],
                    digits = 3)))
}


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Logistic regression model ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#...............................................................................
# Creating logistic regression model with all predictors
#...............................................................................

logMod1 <- glm(Class ~ ., data = train, family = binomial)
summary(logMod1)
# Predicting on validation data
log1Pred <- predict(logMod1, newdata = valid, type = "response")
log1Pred <- ifelse(log1Pred > 0.5, 1, 0)

# Accuracy, specificity and sensitivity of full log model using validation data
validClass(log1Pred)

#...............................................................................
# Variable selection using stepwise AIC
#...............................................................................

logModStep <- stepAIC(logMod1, direction = "backward")
logModStep$formula

# Log model with reduced variables
logMod2 <- glm(Class ~ Age + X.Country + Nscore + Escore + Oscore + SS,
               data = train, family = binomial)
summary(logMod2)

# Predicting on validation data
log2Pred <- predict(logMod2,
                    newdata = valid[c(c("Age", "X.Country", "Nscore",
                                        "Escore", "Oscore", "SS", "Class"))],
                    type = "response")
log2Pred <- ifelse(log2Pred > 0.5, 1, 0)

# Accuracy, specificity and sensitivity of reduced log model using validation data
validClass(log2Pred)


# Creating datasets with reduced predictors selected by stepwise regression
train2 <- train[, c("Age", "X.Country", "Nscore", "Escore", "Oscore", "SS", "Class")]
valid2 <- valid[, c("Age", "X.Country", "Nscore", "Escore", "Oscore", "SS", "Class")]
test2 <- test[, c("Age", "X.Country", "Nscore", "Escore", "Oscore", "SS", "Class")]


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# K nearest neighbors models ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#...............................................................................
# KNN with all predictors
#...............................................................................

set.seed(123)

# Testing 50 values of K using 10 fold cross validation
knn1 <- train(Class ~ ., data = train, method = "knn",
              trControl = trainControl("cv", number = 10),
              tuneGrid=data.frame(k = seq(1, 199, length.out = 100)),
              metric = "Accuracy")

plot(knn1, type = "l", lwd = 1.8, xlab = "Values of K")
knn1$bestTune
knn1$results

# Make predictions on the validation data with k = 109
knn1Pred <- knn(
  train = train[,-12], test = valid[,-12], cl = train$Class, k = 109)

# Accuracy, specificity and sensitivity of KNN model using validation data

validClass(knn1Pred)

#...............................................................................
# KNN with stepwise selected predictors
#...............................................................................

set.seed(123)

# Testing 100 values of K using 10 fold cross validation
knn2 <- train(Class ~ ., data = train2, method = "knn",
              trControl = trainControl("cv", number = 10),
              tuneGrid=data.frame(k = seq(1, 199, length.out = 100)),
              metric = "Accuracy")

plot(knn2, type = "l", lwd = 1.8, xlab = "Values of K")
knn2$bestTune
knn2$results

# Make predictions on the validation data with k = 129
knn2Pred <- knn(
  train = train2[,-12], test = valid2[,-12], cl = train2$Class, k = 129)

# Accuracy, specificity and sensitivity of KNN model using validation data

validClass(knn2Pred)

# Creating plot of CV selection of k for both models
knnDF <- data.frame(k = seq(1, 199, length.out = 100),
                    All_Predictors = knn1$results$Accuracy,
                    Subset_of_Predictors = knn2$results$Accuracy)
knnDFlong <- pivot_longer(
  knnDF,cols = c(All_Predictors, Subset_of_Predictors))

colnames(knnDFlong)[2] <- "Model"
ggplot(data = knnDFlong, aes(x = k, y = value, colour = Model)) +
  geom_line(size = 0.7) +
  xlab("Values of k") + ylab("Cross-validation accuracy") +
  theme(legend.position="bottom")


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Classification tree model ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#...............................................................................
# Tree with all predictors
#...............................................................................
set.seed(123)

# Creating fully grown tree
fullTree1 <- rpart(Class ~ ., data = train, method = "class",cp = -1)
rpart.plot(fullTree1, type=2,extra=4)

# Looking at variable importance
fullTree1$variable.importance

# Looking at cross validation results
fullTree1$cptable
plotcp(fullTree1)

# Obtaining value of cp within one se of lowest cv error rate
cp1 <- max(fullTree1$cptable[
  fullTree1$cptable[,4] <
    fullTree1$cptable[fullTree1$cptable[,4] == min(fullTree1$cptable[,4]),4] +
    fullTree1$cptable[fullTree1$cptable[,4] == min(fullTree1$cptable[,4]),5],
  1])
cp1

# Smallest tree within one standard error of the min CV error
tree1Final <- rpart(Class ~ ., data = train, method = "class",cp = cp1)
rpart.plot(tree1Final, type=2,extra=4)

# Predicting on validation data
tree1Pred <- predict(tree1Final, newdata = valid, type = "class")

# Accuracy, specificity and sensitivity of tree model using validation data
validClass(tree1Pred)

#...............................................................................
# Tree with stepwise selected predictors
#...............................................................................
set.seed(123)

# Creating fully grown tree
fullTree2 <- rpart(Class ~ ., data = train2, method = "class",cp = -1)
rpart.plot(fullTree2, type=2,extra=4)

# Looking at variable importance
fullTree2$variable.importance

# Looking at cross validation results
fullTree2$cptable
plotcp(fullTree2)


# Obtaining value of cp within one se of lowest cv error rate
cp2 <- max(fullTree2$cptable[
  fullTree2$cptable[,4] <
    fullTree2$cptable[fullTree2$cptable[,4] == min(fullTree2$cptable[,4]),4] +
    fullTree2$cptable[fullTree2$cptable[,4] == min(fullTree2$cptable[,4]),5],
  1])
cp2

# No smaller fell within one standard error of the min CV error so min cp used
tree2Final <- rpart(Class ~ ., data = train2, method = "class",cp = 0.02)
rpart.plot(tree2Final, type=2,extra=4)

# Predicting on validation data
tree2Pred <- predict(tree2Final, newdata = valid, type = "class")

# Accuracy, specificity and sensitivity of tree model using validation data
validClass(tree2Pred)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Random forest model ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#...............................................................................
# Forest with all predictors
#...............................................................................
set.seed(123)

# Creating random forest model using 10 fold cross validation
tunegrid1 <- expand.grid(.mtry = (2:11)) 

forest1 <- train(
  Class ~., data = train, method = "rf",
  trControl = trainControl("cv", number = 10),
  tuneGrid = tunegrid1,
  importance = TRUE)


forest1$results
# finding best number of predictor variables sampled at each split
forest1$bestTune

# looking at variable importance
importance(forest1$finalModel)

# Predicting on validation data
forest1Pred <- predict(forest1, newdata = valid)

# Accuracy, specificity and sensitivity of random forest model using validation data
validClass(forest1Pred)

#...............................................................................
# Forest with stepwise selected predictors
#...............................................................................
set.seed(123)

# Creating random forest model using 10 fold cross validation
tunegrid2 <- expand.grid(.mtry = (2:6)) 
forest2 <- train(
  Class ~., data = train2, method = "rf",
  trControl = trainControl("cv", number = 10),
  tuneGrid = tunegrid2,
  importance = TRUE)

# finding best number of predictor variables sampled at each split
forest2$bestTune
forest2$results
# looking at variable importance
importance(forest2$finalModel)

# Predicting on validation data
forest2Pred <- predict(forest2, newdata = valid2)

# Accuracy, specificity and sensitivity of random forest model using validation data
validClass(forest2Pred)


# Creating dataframe of cv accuracy results for differnet values of m
forestDF <- data.frame(m = as.numeric(forest1$results[,1]),
                       All_Predictors = as.numeric(forest1$results[,2]),
                       Subset_of_Predictors = as.numeric(forest2$results[,2]))
forestDF[6:10,3] <- "na"
forestDF[,3] <- as.numeric(forestDF[, 3])
forestDFlong <- pivot_longer(
  forestDF,cols = c(All_Predictors, Subset_of_Predictors))

colnames(forestDFlong)[2] <- "Model"
ggplot(data = forestDFlong, aes(x = m, y = value, colour = Model)) +
  geom_line(size = 0.7) +
  xlab("Number of predictors included for random selection") + ylab("Cross-validation accuracy") +
  theme(legend.position="bottom")


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Support vector machines ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#...............................................................................
# SVM with all predictors
#...............................................................................

# Non-linear kernel used as the response does not look linearly separable
set.seed(123)
svm1Tune <- tune(svm, Class ~., data=train, type="C-classification",
                 kernel="radial",
                 ranges = list(
                   cost = c(0.01, 0.1, 1, 10, 100),
                   gamma = c(0.01, 0.1, 1, 10, 100)))

svm1Tune

# Trying again with smaller gamma and cost
set.seed(123)
svm1Tune <- tune(
  svm, Class ~., data=train, type="C-classification",
  kernel="radial",
  ranges = list(
    cost = c(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5),
    gamma = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)))

svm1Tune

# Creating line plot of tuning performance
svmDF1 <- as.data.frame(svm1Tune$performances)
ggplot(svmDF1, aes(x = cost, y = error, color = as.factor(gamma))) +
  geom_line(size = 0.7) +
  scale_x_continuous(breaks=svmDF1$cost) +
  xlab("Values for C parameter") + ylab("Cros-classifcation error") +
  guides(color = guide_legend(title = "Values for gamma"))

# Predicting on validation data
svm1Pred <- predict(svm1Tune$best.model, newdata = valid)

# Accuracy, specificity and sensitivity of random forest model using validation data
validClass(svm1Pred)

#...............................................................................
# SVM with stepwise selected predictors
#...............................................................................

# Trying same range for cost and gamma as was used in previous model
set.seed(123)
svm2Tune <- tune(
  svm, Class ~., data=train2, type="C-classification",
  kernel="radial",
  ranges = list(
    cost = c(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5),
    gamma = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)))

svm2Tune

# Creating line plot of tuning performance
svmDF2 <- as.data.frame(svm2Tune$performances)
ggplot(svmDF2, aes(x = cost, y = error, color = as.factor(gamma))) +
  geom_line(size = 0.7) +
  scale_x_continuous(breaks=svmDF2$cost) +
  xlab("Values for C parameter") + ylab("Cros-classifcation error") +
  guides(color = guide_legend(title = "Values for gamma"))

# Predicting on validation data
svm2Pred <- predict(svm2Tune$best.model, newdata = valid2)

# Accuracy, specificity and sensitivity of reduced random forest model
# using validation data
validClass(svm2Pred)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Discriminant analysis ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# CVA used as classes did not look normally distributed in some variables
#...............................................................................
# CVA with all predictors
#...............................................................................

cva1 <- lda(Class ~ ., data = train, prior = rep(1/2, 2))
coef(cva1)
cva1

# Predicting on validation data
cva1Pred <- predict(cva1, newdata = valid)
cva1Pred$class

# Accuracy, specificity and sensitivity of cva model using validation data
validClass(cva1Pred$class)

#...............................................................................
# CVA with stepwise selected predictors
#...............................................................................

cva2 <- lda(Class ~ ., data = train2, prior = rep(1/2, 2))
cva2

# Predicting on validation data
cva2Pred <- predict(cva2, newdata = valid2)
cva2Pred$class

# Accuracy, specificity and sensitivity of cva model using validation data
validClass(cva2Pred$class)



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# K nearest neighbors with reduced set of predictors on tests data ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Make predictions on the validation data with k = 129
knn2Test <- knn(
  train = train2[,-12], test = test2[,-12], cl = train2$Class, k = 129)

# Creating function to assess model accuracy, specificity and sensitivity
# on validation data

testClass <- function(x){
  print(paste("Accuracy: ", round(mean(x == test$Class), digits = 3)))
  print(paste("Sensitivity: ",
              round(sweep(table(test$Class, x),
                          1, apply(table(test$Class, x), 1, sum),
                          "/")[2,2],
                    digits = 3)))
  print(paste("Specificity: ",
              round(sweep(table(test$Class, x),
                          1, apply(table(test$Class, x), 1, sum),
                          "/")[1,1],
                    digits = 3)))
}

# Accuracy, specificity and sensitivity of KNN model using test data
testClass(knn2Test)


