
# Machine Learning Classification Project using R
## Predicting drug use with five machine learning algorithms  
<br>  
***Note: This project was completed as part of an MSc in Data Analytics and the data provided to us was a modified version of the data used in the study by Fehrman et al., (2017) that had been externally scaled to to change some categorical variables to psuedo-continuous variables in order to simplify the analysis***  
<br>  

## Overview  

The aim of the project was to evaluate whether drug use could be predicted from personality trait and demographic variables using a number of classification methods (logistic regression, k-nearest neighbours, classification tress, random forests, support vector machines and canonical variate analysis).  The data was based on All analysis was completed using R.  

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

Figure 2 contains correlations, scatterplots, and density plots for all the independent variables. There are no strong correlations between any of the predictor variables meaning multicollinearity was not a concern, but there is evidence that some are not normally distributed and there appears to be an extreme outlier in the ethnicity data.  
<br>  

As some of the classification methods such as k-nearest neighbours are sensitive to differences in scale of the variables, the training, validation and test dataset were centered and scaled using the mean and standard deviation from the training data before being used in modelling.  
<br>  


### References  
Fehrman, E., Muhammad, A. K., Mirkes, E. M., Egan, V., & Gorban, A. N. (2017). The Five Factor Model of personality and evaluation of drug consumption risk (arXiv:1506.06297). arXiv. https://doi.org/10.48550/arXiv.1506.06297
