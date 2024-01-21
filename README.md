
# Machine Learning Classification Project using R
## Predicting drug use with five machine learning algorithms  
<br>  
***Note: This project was completed as part of an MSc in Data Analytics and the data provided to us was a modified version of the data used in the study by Fehrman et al., (2017) that had been externally scaled to to change some categorical variables to psuedo-continuous variables in order to simplify the analysis***  
<br>  

## Overview  

The aim of the project was to evaluate whether drug use could be predicted from personality trait and demographic variables using a number of classification methods (logistic regression with lasso penalty, k-nearest neighbours, classification trees, random forests, support vector machines and canonical variate analysis).  The classification threshold used for all models was 0.5 such that probabilities above the threshold were deemed to indicate drug use and values below the threshold no drug use and the metrics used to evaluate model performance were accuracy, sensitivity and specificity. The data was based on All analysis was completed using R.  

<br>  

## Exploratory data analysis  
The data set comprised of 600 observations with the dependent variable being a binary classifer indicating whether an individual had ever taken legal or illegal drugs (never taken drugs n = 300, has taken drugs n= 300) and the 11 independent variables all continuous (see Table 1 for an overview of the variables).  
<br>  

***Table 1. Independent variables used in the analysis***  

|     Variable     |     Description                                                              |
|------------------|------------------------------------------------------------------------------|
|     Age          |     Age of study participants                                                |
|     Education    |     Education level of study participants                                    |
|     X.country    |     Country of origin for study participants                                 |
|     Ethnicity    |     Ethnicity of study participants                                          |
|     Nscore       |     Participants neuroticism personality trait   score                       |
|     Escore       |     Participants extraversion personality trait   score                      |
|     Oscore       |     Participants openness to experience   personality trait score            |
|     Ascore       |     Participants agreeableness personality trait   score                     |
|     Cscore       |     Participants conscientiousness personality   trait score                 |
|     Impulsive    |     Participants impulsiveness personality trait   score                     |
|     SS           |     Participants sensation personality trait   seeking score                 |
|     Class        |     Whether participants had ever taken legal or   illegal drugs (yes/no)    |  
<br>  

The dataset was split into training (50% of observations), validation (30% of observations) and test (20% of observations) sets while preserving the distribution of the response variable, and this was done using the caret package. The training dataset was used for the exploratory analysis and model building, the validation dataset was used for comparing the performance of the different modelling approaches and the test dataset was used to assess the performance of the model that had the highest prediction accuracy on the validation data. The relationship between the response variable and the predictors are displayed in the boxplots of Figure 1. From the plots there appears to be an association between drug use and the variables Age, SS and X.Country, and several variables appear to have outliers.  
<br>  
![Figure1](https://github.com/MarkMData/ML_classification_project/blob/main/images/boxplots.jpeg)  
***Figure 1. Distributions for predictor variables by drug use (0 = never used, 1 = has used).***  
<br>  

Figure 2 contains correlations, scatterplots, and density plots for all the independent variables. There are no strong correlations between any of the predictor variables meaning multicollinearity was not a concern, but there is evidence that some are not normally distributed and there appears to be an extreme outlier in the ethnicity data.  
<br>  
![Figure2](https://github.com/MarkMData/ML_classification_project/blob/main/images/pairsplot.jpeg)  
***Figure 2. Relationships between predictor variables with drug use as a class (0 = never used, 1 = has used).***  
<br>  
From the boxplots and scatter plots it appeared that the ethicity variable was comprised of very few unique values. To identify if the ethnicity (or any other variables) had a near zero variance the percentage of unique values and the frequency ratio of the most prevelant to the second most prevelant value for each variable was calculated (using the caret package), with cut offs of 10% for the unique values and 20:1 for the frequency ratio, as recommended by Kuhn & Johnson (2013). The only variable to meet both criteria was the ethicity variable with a frequecy ratio of 30.44 and only 2.33% unique values, and was excluded from all the models. As some of the classification methods such as k-nearest neighbours are sensitive to differences in scale of the variables, the training, validation and test dataset were centered and scaled using the mean and standard deviation from the training data before being used in modelling, and this was also done using the caret package.  
<br>  

## Method and Results  
<br>  

### Logistic regression with lasso penalty  
The logistic regression model with a lasso penalty was fit using the glmnet package. The value for lambda was chosen using 10 fold cross validation with 100 values of λ evaluated using 10-fold cross validation to identify when the minimum misclassification error occurred and then the value within one standard error of this (λ = 0.196) was selected (see Figure 3).  
<br>  
![Figure3](https://github.com/MarkMData/ML_classification_project/blob/main/images/lassoCV.jpeg)  
<br>  
### K-nearest neighbours  
To identify the optimal value for k, 10-fold cross validation on the training data was used to iteratively assess the prediction accuracy for 50 values of k ranging from one to 199 (odd values only to prevent ties) and the results for the model with all predictors and the model with a subset of predictors are displayed in Figure 3. From the plot it can be seen that the model with fewer predictors generally has higher accuracy across all values for k. The model with all predictors had highest avarage cross-validation prediction accuracy on the training data of 0.807 at a value of k = 109.  For the model with the subset of predictors peak prediction accuracy was 0.817 and this occurred at a value of k = 129. Using the selected values of k the out of sample prediction performance for both models was assessed against the validation data with the accuracy, specificity and sensitivity displayed in Table 4. On the validation data the model with all predictors had accuracy of 0.783, sensitivity of 0.767 and specificity of 0.8 while the model with the subset of predictors had higher accuracy of 0.817, the same level of sensitivity of 0.767 and higher specificity of 0.867. 


### References  
Fehrman, E., Muhammad, A. K., Mirkes, E. M., Egan, V., & Gorban, A. N. (2017). The Five Factor Model of personality and evaluation of drug consumption risk (arXiv:1506.06297). arXiv. https://doi.org/10.48550/arXiv.1506.06297  
Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer. https://doi.org/10.1007/978-1-4614-6849-3
