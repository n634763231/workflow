
#Get Iris from R
library(caret)
library(dplyr)
library(rattle)

a <- read.csv("data_clean/iris_clean.csv") 

#Create Training  and Testing Sets
set.seed(42)
inTrain<-createDataPartition(y=a$Species, p=0.70, list=FALSE)
training.Iris<-a[inTrain,]
testing.Iris<-a[-inTrain,]

# Display a pairs plot for the selected variables. 
training.Iris %>%
  dplyr::mutate(Species=as.factor(Species)) %>%
  GGally::ggpairs(columns=c(1,2,3,4,5),
                  mapping=ggplot2::aes(colour=Species, alpha=0.5),
                  diag=list(continuous="density",
                            discrete="bar"),
                  upper=list(continuous="cor",
                             combo="box",
                             discrete="ratio"),
                  lower=list(continuous="points",
                             combo="denstrip",
                             discrete="facetbar")) +
  ggplot2::theme(panel.grid.major=ggplot2::element_blank())

#preProcess, center and scale the data

##training set
preObj<-preProcess(training.Iris[,-5], method = c("center", "scale"))
preObjData<-predict(preObj,training.Iris[,-5])
boxplot(preObjData, main="Normalized data" )
training.Iris_N <- transform(preObjData,Species=training.Iris$Species)
##testing set
preObj<-preProcess(testing.Iris[,-5], method = c("center", "scale"))
preObjData<-predict(preObj,testing.Iris[,-5])
boxplot(preObjData, main="Normalized data" )
testing.Iris_N <- transform(preObjData,Species=testing.Iris$Species)

# Decision Tree 

## The 'rpart' package provides the 'rpart' function.

library(rpart, quietly=TRUE)

## Reset the random number seed to obtain the same results each time.

set.seed(42)

## Build the Decision Tree model.

rpart <- rpart(Species ~ .,
                   training.Iris_N,
                   method="class",
                   parms=list(split="information"),
                   control=rpart.control(usesurrogate=0, 
                                         maxsurrogate=0),
                   model=TRUE)

## Generate a textual view of the Decision Tree model.

print(rpart)
printcp(rpart)

## Evaluate model performance on the testing dataset. 

### Generate an Error Matrix for the Decision Tree model.

#### Obtain the response from the Decision Tree model.

pr_rpart <- predict(rpart, newdata=testing.Iris_N,
                  type="class")

#### Generate the confusion matrix showing counts.

rattle::errorMatrix(testing.Iris_N$Species, pr_rpart, count=TRUE)

#### Generate the confusion matrix showing proportions.

(per_rpart <- rattle::errorMatrix(testing.Iris_N$Species, pr_rpart))

#### Calculate the overall error percentage.

cat("Calculate the overall error percentage:", 100-sum(diag(per_rpart), na.rm=TRUE))

#### Calculate the averaged class error percentage.

cat("Calculate the averaged class error percentage:", mean(per_rpart[,"Error"], na.rm=TRUE))


### ROC Curve: requires the ROCR package.

library(ROCR)

#### ROC Curve: requires the ggplot2 package.

library(ggplot2, quietly=TRUE)

#### Generate an ROC Curve for the rpart model on a [test].

pr_rpart_roc <- predict(rpart, newdata=testing.Iris_N, type= "prob")[,2]

#### Remove observations with missing target.

no.miss   <- na.omit(testing.Iris_N$Species)
miss.list <- attr(no.miss, "na.action")
attributes(no.miss) <- NULL

if (length(miss.list))
{
  pred_rpart <- prediction(pr_rpart_roc[-miss.list], no.miss)
} else
{
  pred_rpart <- prediction(pr_rpart_roc, no.miss)
}

pe <- performance(pred_rpart, "tpr", "fpr")
au <- performance(pred_rpart, "auc")@y.values[[1]]
pd <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
p <- ggplot(pd, aes(x=fpr, y=tpr))
p <- p + geom_line(colour="red")
p <- p + xlab("False Positive Rate") + ylab("True Positive Rate")
p <- p + ggtitle("ROC Curve Decision Tree a [test] Species")
p <- p + theme(plot.title=element_text(size=10))
p <- p + geom_line(data=data.frame(), aes(x=c(0,1), y=c(0,1)), colour="grey")
p <- p + annotate("text", x=0.50, y=0.00, hjust=0, vjust=0, size=5,
                  label=paste("AUC =", round(au, 2)))
print(p)

#### Calculate the area under the curve for the plot.

no.miss   <- na.omit(testing.Iris_N$Species)
miss.list <- attr(no.miss, "na.action")
attributes(no.miss) <- NULL

if (length(miss.list))
{
  pred_rpart <- prediction(pr_rpart_roc[-miss.list], no.miss)
} else
{
  pred_rpart <- prediction(pr_rpart_roc, no.miss)
}
performance(pred_rpart, "auc")

# Support vector machine. 

## The 'kernlab' package provides the 'ksvm' function.

library(kernlab, quietly=TRUE)

## Build a Support Vector Machine model.

set.seed(42)
ksvm <- ksvm(as.factor(Species) ~ .,
                 data=training.Iris_N,
                 kernel="rbfdot",
                 prob.model=TRUE)

## Generate a textual view of the SVM model.

ksvm

## Evaluate model performance on the testing dataset. 

### Generate an Error Matrix for the SVM model.

#### Obtain the response from the SVM model.

pr_ksvm <- kernlab::predict(ksvm, newdata=na.omit(testing.Iris_N))

#### Generate the confusion matrix showing counts.

rattle::errorMatrix(na.omit(testing.Iris_N)$Species, pr_ksvm, count=TRUE)

#### Generate the confusion matrix showing proportions.

(per_ksvm <- rattle::errorMatrix(na.omit(testing.Iris_N)$Species, pr_ksvm))

#### Calculate the overall error percentage.

cat("Calculate the overall error percentage:", 100-sum(diag(per_ksvm), na.rm=TRUE))

#### Calculate the averaged class error percentage.

cat("Calculate the averaged class error percentage:", mean(per_ksvm[,"Error"], na.rm=TRUE))

### ROC Curve: requires the ROCR package.

library(ROCR)

#### ROC Curve: requires the ggplot2 package.

library(ggplot2, quietly=TRUE)

#### Generate an ROC Curve for the ksvm model on a [test].

pr_ksvm_roc <- kernlab::predict(ksvm, newdata=na.omit(testing.Iris_N),
                           type    = "probabilities")[,2]

#### Remove observations with missing target.

no.miss   <- na.omit(na.omit(testing.Iris_N)$Species)
miss.list <- attr(no.miss, "na.action")
attributes(no.miss) <- NULL

if (length(miss.list))
{
  pred <- prediction(pr_ksvm_roc[-miss.list], no.miss)
} else
{
  pred <- prediction(pr_ksvm_roc, no.miss)
}

pe <- performance(pred, "tpr", "fpr")
au <- performance(pred, "auc")@y.values[[1]]
pd <- data.frame(fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))
p <- ggplot(pd, aes(x=fpr, y=tpr))
p <- p + geom_line(colour="red")
p <- p + xlab("False Positive Rate") + ylab("True Positive Rate")
p <- p + ggtitle("ROC Curve SVM a [test] Species")
p <- p + theme(plot.title=element_text(size=10))
p <- p + geom_line(data=data.frame(), aes(x=c(0,1), y=c(0,1)), colour="grey")
p <- p + annotate("text", x=0.50, y=0.00, hjust=0, vjust=0, size=5,
                  label=paste("AUC =", round(au, 2)))
print(p)

#### Calculate the area under the curve for the plot.

no.miss   <- na.omit(testing.Iris_N$Species)
miss.list <- attr(no.miss, "na.action")
attributes(no.miss) <- NULL

if (length(miss.list))
{
  pred_ksvm <- prediction(pr_ksvm_roc[-miss.list], no.miss)
} else
{
  pred_ksvm <- prediction(pr_ksvm_roc, no.miss)
}
performance(pred_ksvm, "auc")

