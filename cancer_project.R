setwd("D:/Nishant Documents/Data")
cancer_data<-read.csv("data.csv",header=TRUE)
head(cancer_data)
cancer_data<-cancer_data[,-1]
cancer_data<-cancer_data[,-32]
View(cancer_data)
head(cancer_data)
is.na(cancer_data)
dim(cancer_data)
sapply(cancer_data,class)


library(caret)
library(e1071)
split=0.80
trainIndex <- createDataPartition(cancer_data$diagnosis, p=split, list=FALSE)
data_train <- cancer_data[ trainIndex,]
data_test <- cancer_data[-trainIndex,]
# train a naive bayes model
model_nb <- naiveBayes(diagnosis~., data=data_train)
# make predictions

predictions_naive <- predict(model_nb, data_test)
# summarize results
confusionMatrix(predictions_naive,data_test$diagnosis)




# # -----------------------------------------------------------------------
trainIndex <- createDataPartition(cancer_data$diagnosis, p=split, list=FALSE)
data_train <- cancer_data[ trainIndex,]
data_test <- cancer_data[-trainIndex,]
logRegModel <- train(diagnosis ~ ., data=data_train, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, data_test)
logRegPredictionprob <- predict(logRegModel, data_test, type='prob')[2]
confusionMatrix(logRegPrediction, data_test[,"diagnosis"])
#ROC Curve
AUC=list()
Accuracy=list()
library(pROC)
AUC$logReg <- roc(as.numeric(data_test$Outcome),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']


# # -----------------------------------------------------------------------
library("e1071")
trainIndex <- createDataPartition(cancer_data$diagnosis, p=split, list=FALSE)
data_train <- cancer_data[ trainIndex,]
data_test <- cancer_data[-trainIndex,]
svm_model <- svm(diagnosis ~ ., data=data_train)
summary(svm_model)
pred_svm <- predict(svm_model,data_test)
confusionMatrix(pred_svm,data_test$diagnosis)

#-------------------------------------------------------------------------------

library(MASS)
trainIndex <- createDataPartition(cancer_data$diagnosis, p=split, list=FALSE)
data_train <- cancer_data[ trainIndex,]
data_test <- cancer_data[-trainIndex,]
tr_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 10)
lda_train = train(diagnosis ~ ., 
                  data=data_train, 
                  method="lda",
                  trControl=tr_ctrl)
pred_lda = predict(lda_train, data_test)
confusionMatrix(pred_lda,data_test$diagnosis)
# --------------------------------------------------------------------
forest_train = train(diagnosis ~ ., 
                     data=data_train, 
                     method="rf",
                     trControl=tr_ctrl)
pred_forest<-predict(forest_train,data_test)
confusionMatrix(pred_forest,data_test$diagnosis)
# -----------------------------------------------------------------

library(C50)
trainIndex <- createDataPartition(cancer_data$diagnosis, p=split, list=FALSE)
data_train <- cancer_data[ trainIndex,]
data_test <- cancer_data[-trainIndex,]
c50_model<-C5.0(diagnosis~.,data_train)
pred_c50<-predict(c50_model,data_test)

confusionMatrix(pred_c50,data_test$diagnosis)
#-----------------------------------------------------------------------
library(nnet)

nnet_model<-nnet(diagnosis~.,data_train,size=10)
nnet_pred<-predict(nnet_model, data_test, type="class")

confusionMatrix(nnet_pred,data_test$diagnosis)
#------------------------------------------------------------------------

library(adabag)
adaboost<-boosting(diagnosis~.,data_train)
ada_pred<-predict(adaboost,data_test)
confusionMatrix(ada_pred$class,data_test$diagnosis)
t1<-adaboost$trees[[1]]
library(tree)
plot(t1)
text(t1,pretty=1)
#-----------------------------------------------------------------------