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
library(viridis)

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


# Finding variables with near zero variance
nzv <- nearZeroVar(train, saveMetrics= TRUE)
nzv
var_to_drop <- nearZeroVar(train)
var_to_drop

# Dropping the ethnicity variable
train <- train[, -var_to_drop]
valid <- valid[, -var_to_drop]
test <- test[, -var_to_drop]


# Centering and scaling
preProc <- preProcess(train, method = c("center", "scale"))
train <- predict(preProc, train)
valid <- predict(preProc, valid)
test <- predict(preProc, test)

################################################################################
# Logistic regression with lasso penalty ----
################################################################################

# Creating training X matrix and Y vector
X.train <- model.matrix(Class ~ .-1, data = train)
Y.train <- train$Class

# Creating validation X matrix and Y vector
X.valid <- model.matrix(Class ~ .-1, data = valid)
Y.valid <- valid$Class

set.seed(123)

# Applying lasso regression
fitLasso <- glmnet(X.train, Y.train, family = "binomial", alpha=1)

# plotting coefficients against the log of lambda
plot(fitLasso, xvar = "lambda",label=TRUE)


set.seed(123)
# Using k-fold cross validation with miss-classification error as measure
lassoCV <- cv.glmnet(X.train, Y.train, family = "binomial", alpha=1,
                     type.measure = "class")
plot(lassoCV)

# Checking value of Lambda
lassoCV$lambda.1se

# Lasso model with lambda.1se
modLassoLam1se <- glmnet(X.train, Y.train, family = "binomial", alpha=1,
                         lambda = lassoCV$lambda.1se)
modLassoLam1se$beta

# Make prediction on validation data for lasso model with lambda.1se
lassoPredClassLam1se <- predict(modLassoLam1se, newx = X.valid, type = "class")
lassoPredClassLam1se<- as.factor(lassoPredClassLam1se)
# confusion matrix for lasso model
confusionMatrix(lassoPredClassLam1se, valid$Class)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# K nearest neighbors models ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

set.seed(123)
# Testing 50 values of K using 10 fold cross validation
knn1 <- train(Class ~ ., data = train, method = "knn",
              trControl = trainControl("cv", number = 10),
              tuneGrid=data.frame(k = seq(1, 99, by=2)),
              metric = "Accuracy")
plot(knn1, type = "l", lwd = 1.8, xlab = "Values of K")

knn1$bestTune
knn1$results

# Make predictions on the validation data
knn1Pred <- knn(
  train = train[,-12], test = valid[,-12], cl = train$Class, k = 97)

# Accuracy, specificity and sensitivity of KNN model using validation data
confusionMatrix(knn1Pred, valid$Class)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Classification tree model ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

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
tree1Final <- prune(fullTree1, cp=cp1)
rpart.plot(tree1Final, type=2,extra=4)

# Predicting on validation data
tree1Pred <- predict(tree1Final, newdata = valid, type = "class")

# Accuracy, specificity and sensitivity of tree model using validation data
confusionMatrix(tree1Pred, valid$Class)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Random forest model ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

set.seed(123)

# Creating random forest model using 10 fold cross validation
tunegrid1 <- expand.grid(.mtry = (2:11)) 
tunegrid1
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

plot(forest1)

# Predicting on validation data
forest1Pred <- predict(forest1, newdata = valid)

# Accuracy, specificity and sensitivity of random forest model using validation data
confusionMatrix(forest1Pred, valid$Class)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Support vector machines ----
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Non-linear kernel used as the response does not look linearly separable
set.seed(123)
cost <- 2**(-2:10)
gamma <- 10**(-7:0)
cost
gamma
svm1Tune <- tune(svm, Class ~., data=train, type="C-classification",
                 kernel="radial",
                 ranges = list(
                   cost = cost,
                   gamma = gamma))

# plot of error with different tuning parameters
plot(
  svm1Tune,
  transform.x = log2,  transform.y = log10,
  main = 'Average 10-fold CV error',
  xlab = 'Log2(cost)', ylab = 'Log10(gamma)',
  nlevels = 30,
  color.palette =turbo)
svm1Tune$best.parameters

# Predicting on validation data
svm1Pred <- predict(svm1Tune$best.model, newdata = valid)

# Accuracy, specificity and sensitivity of random forest model using validation data
confusionMatrix(svm1Pred, valid$Class)

################################################################################
# Assessing performance of the best model using the test data
################################################################################

# Make predictions on the test data using the KNN model
knn2Pred <- knn(
  train = train[,-12], test = test[,-12], cl = train$Class, k = 97)

# Accuracy, specificity and sensitivity of KNN model using test data
confusionMatrix(knn2Pred, test$Class)


