# Qualitative Activity Recognition of Weight Lifting Exercises

## Abstract

This is a submission for the course project of the Practical Machine Learning class in the Data Science track at Coursera.

The objective is to detect the mistakes a person makes in the execution of a weight lifting exercise. The collected dataset comes from the execution of the Unilateral Dumbbell Biceps Curl in five different ways (or "classes"). Class A indicates the exercise is executed according to the specifications, while classes B, C, D and E are four different fashions of common mistakes in the execution. 

The data is collected using a total of 4 sensors attached to the executor's body and the dumbbell. The data of six participants (or "users") has been collected. The data consists of raw data and does not include summary statistics.

The goal of the project is to predict the manner in which the persons did the exercise. This is the "classe" variable in the training set. In the prediction model, one may use any of the other variables to predict with. One should create a report describing how one built the model, how one used cross validation, what one thinks the expected out of sample error is, and why one made the choices one did. One will also use the prediction model to predict 20 different test cases. 

## The data

The data consists of two files. The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
## read the training data file
data <- read.csv("pml-training.csv")
```

The dataset has 19622 rows of 160 variables.

Given the size of the dataset, we opt to do apply a simple validation set approach. 

```r
## Create a set for cross-validation
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(3433)
inTrain = createDataPartition(data$classe, p = .70)[[1]]
trainingset = data[ inTrain,]
trainingtest <- trainingset # used for testing the model later on
validation = data[-inTrain,]
remove(data)
```

Since most of the summary statistic columns have erroneous data, we will leave these columns out.

```r
## Clean up the dataset
colsSelected <- !sapply(strsplit(names(trainingset), "_"), 
                    function(x) x[1]) %in% c("kurtosis", "skewness", "max", "min", "amplitude", "var", "avg", "stddev")
trainingset <- trainingset[, colsSelected]
```

## Motivation for the appraoch

The objective of the exercise is to predict a classe. We will use a categorization approach. The first option we will investigate is random forest. 
Given the number of variables in the dataset and the computation requirements of random forests, we will reduce the number of variables with principal component analysis.


```r
## Motivation for the approach
pr <- prcomp(trainingset[,c(-(1:7),-60)], center=TRUE, scale=TRUE)
tr <- predict(pr, trainingset)
par(mfrow=c(1,2))
plot(tr[,1:2], col=trainingset$classe, main="PCA by classe")
legend(x="bottomright", levels(trainingset$classe), pch=1, col=1:5)
plot(tr[,1:2], col=trainingset$user_name, main="PCA by user_name")
legend(x="bottomright", levels(trainingset$user_name), pch=1, col=1:6)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

```r
remove(tr)
```

One might have expected to see some pattern in the data based on PC1 and PC2 with respect to the classes, this obviously is not the case (see chart on the left). However, we clearly see arise clusters based on the 6 users.
Most of the variability in the dataset is explained by the users (see chart on the right), not by the way the exercise is executed.
The approach we will follow is to filter out the user and to treat the six clusters separately.

One can also detect an outlier at the very right in the charts. For the moment will leave the outlier as is.

## Model building

We prefer to standardize the variables in the PCA. Since some of the variables have zero values, causing the standardization to err, we clean up these variables.

```r
# Select the columns given no error on standardized PCA
neglectCol <- function (x) {
        s <- split(x, trainingset$user_name)
        sum(sapply(s, sum) == 0) > 0
}
neglect <- c(rep(TRUE, 7), apply(trainingset[,8:59], 2, neglectCol), TRUE)
select <- names(trainingset)[!neglect]
```

As explained above, our first option is to elaborate a random forest model based on principal components. We use the principal components explaining at least 90% of the variance in the variables. And we do so per user.

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# Create a model per user_name
par(mfrow=c(3,2))
for (i in 1:6) {
        training <- subset(trainingset, user_name==levels(trainingset$user_name)[i])
        training2 <- training[,select]
        
        pr <- prcomp(training2, center=TRUE, scale=TRUE)
        training2 <- as.data.frame(predict(pr, training2))
        
        training2$classe <- training$classe
        training2$user_name <- training$user_name
        
        plot(training2[,1:2], col=training2$classe, main=paste("PCA for",levels(training$user_name)[i]))
        legend(x="topright", levels(training2$classe), pch=1, col=1:5)
        
        set.seed(124512)
        # select the principal components up to a certain variance % (e.g. 90%)
        s <- cumsum(pr$sdev^2)/sum(pr$sdev^2)
        ns <- seq_along(s)[s >= .9][1] 
        
        # Fit the model
        modFit <- train(classe ~ .,method="rf",data=training2[,c(paste("PC", 1:ns, sep=""), "classe", "user_name")], proxy=TRUE)
        
        # Memorize the results (could be done more nicely)
        if (i==1) {mod1 <- modFit; pr1 <- pr} 
        else if (i==2) {mod2 <- modFit; pr2 <- pr} 
        else if (i==3) {mod3 <- modFit; pr3 <- pr} 
        else if (i==4) {mod4 <- modFit; pr4 <- pr} 
        else if (i==5) {mod5 <- modFit; pr5 <- pr} 
        else {mod6 <- modFit; pr6 <- pr }  
}
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 

When we test the model, our results are very satisfactory:

```r
## Do the validation on the trainingset we did set apart earlier
# Create a funtion for predicting
testPrediction <- function (testset) {
        predicted <- rep(NA, nrow(testset))
        for (i in 1:6) {
                if (i==1) {mod <- mod1; pr <- pr1 } 
                else if (i==2) { mod <- mod2; pr <- pr2 } 
                else if (i==3) { mod <- mod3; pr <- pr3 } 
                else if (i==4) { mod <- mod4; pr <- pr4 } 
                else if (i==5) { mod <- mod5; pr <- pr5 } 
                else { mod <- mod6; pr <- pr6 }
        
                subset <- subset(testset, user_name == levels(trainingset$user_name)[i])
                subset2 <- subset[,select]
                val <- as.data.frame(predict(pr, subset))
                val$user_name <- subset$user_name
                tr <- predict(mod, val)
                predicted[testset$user_name == levels(trainingset$user_name)[i]] <- tr
        }
        predicted <- levels(trainingset$classe)[predicted]
        predicted
}
# Test the validation
confusionMatrix(testPrediction(trainingtest), trainingtest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
For now, we do not further need to improve the model.

## Validating the model

Now, we validate the model based on the validation set:

```r
confusionMatrix(testPrediction(validation), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1639   27    7   17    3
##          B    9 1075   25    7    6
##          C   17   30  984   28    6
##          D    8    4    5  910    6
##          E    1    3    5    2 1061
## 
## Overall Statistics
##                                         
##                Accuracy : 0.963         
##                  95% CI : (0.958, 0.968)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.954         
##  Mcnemar's Test P-Value : 4.11e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.979    0.944    0.959    0.944    0.981
## Specificity             0.987    0.990    0.983    0.995    0.998
## Pos Pred Value          0.968    0.958    0.924    0.975    0.990
## Neg Pred Value          0.992    0.987    0.991    0.989    0.996
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.279    0.183    0.167    0.155    0.180
## Detection Prevalence    0.288    0.191    0.181    0.159    0.182
## Balanced Accuracy       0.983    0.967    0.971    0.970    0.989
```

Our accuracy is above 95%. This seems satisfactory.

## Conclusion

Given all the available data, our model based on a random forest per individual, and based on the principal components explaining 90% of the variance is satisfactory to do our predictions on the test data.

```r
# Do the prediction on the test data
testing <- read.csv("pml-testing.csv")
# Submit the answers to Coursera
answers <- testPrediction(testing)
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(answers)
```
