setwd("D:/Nishant Documents/Data")
heart_data<-read.csv("Heart_Disease_Data.csv",header = TRUE)
is.na(heart_data)
sapply(heart_data,class)
summary(heart_data$ca)
summary(heart_data$thal)
summary(heart_data)

ind<-which(heart_data$ca=="?")
heart_data$ca[ind]<-NA
ind1<-which(heart_data$thal=="?")
heart_data$thal[ind1]<-NA

View(heart_data)

ind_pred2<-which(heart_data$pred_attribute==2)
heart_data$pred_attribute[ind_pred2]<-1
ind_pred3<-which(heart_data$pred_attribute==3)
heart_data$pred_attribute[ind_pred3]<-1
ind_pred4<-which(heart_data$pred_attribute==4)
heart_data$pred_attribute[ind_pred4]<-1

heart_data$pred_attribute<-as.factor(heart_data$pred_attribute)

summary(heart_data$pred_attribute)

md.pattern(heart_data)

heart_imputes = mice(heart_data, m=5, maxit = 40)

imputed_heart<-complete(heart_imputes,5)


heart_imputes$method



library(caret)
library(e1071)
split=0.80
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
# train a naive bayes model
model_nb <- naiveBayes(pred_attribute~., data=data_train)
# make predictions
predictions_naive <- predict(model_nb, data_test)
# summarize results
confusionMatrix(predictions_naive,data_test$pred_attribute)

# # -----------------------------------------------------------------------
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
logRegModel <- train(pred_attribute ~ ., data=data_train, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, data_test)
logRegPredictionprob <- predict(logRegModel, data_test, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, data_test[,"pred_attribute"])
#ROC Curve
AUC=list()
Accuracy=list()

library(pROC)


AUC$logReg <- roc(as.numeric(data_test$pred_attribute),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']


# # -----------------------------------------------------------------------
library("e1071")
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
svm_model <- svm(pred_attribute ~ ., data=data_train)
pred_svm <- predict(svm_model,data_test)
confusionMatrix(pred_svm,data_test$pred_attribute)

#-------------------------------------------------------------------------------
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
library(MASS)
tr_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 10)
lda_train = train(pred_attribute ~ ., 
                  data=data_train, 
                  method="lda",
                  trControl=tr_ctrl)
pred_lda = predict(lda_train, data_test)
confusionMatrix(pred_lda,data_test$pred_attribute)
# --------------------------------------------------------------------
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
forest_train = train(pred_attribute ~ ., 
                     data=data_train, 
                     method="rf",
                     trControl=tr_ctrl)
pred_forest<-predict(forest_train,data_test)
confusionMatrix(pred_forest,data_test$pred_attribute)

random_forest<-randomForest(pred_attribute~.,data=data_train)

varImpPlot(random_forest)


#---------------------
library(psych)
prin_comp <- prcomp(data_train[,-c(12,13,14)], scale. = T)

attributes(prin_comp)

prin_comp$center
trg<-predict(prin_comp,data_train)
trn<-data.frame(trg,data_train[12:14])
tst<-predict(prin_comp,data_test)

tst<-data.frame(tst,data_test[12:14])

prntr<-randomForest(pred_attribute~.,data=trn)

pred_pn<-predict(prntr,tst)

confusionMatrix(pred_pn,tst$pred_attribute)

# -----------------------------------------------------------------

library(C50)
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
c50_model<-C5.0(pred_attribute~.,data_train)
pred_c50<-predict(c50_model,data_test)

confusionMatrix(pred_c50,data_test$pred_attribute)
#-----------------------------------------------------------------------
library(nnet)
trainIndex <- createDataPartition(imputed_heart$pred_attribute, p=split, list=FALSE)
data_train <- imputed_heart[ trainIndex,]
data_test <- imputed_heart[-trainIndex,]
nnet_model<-nnet(pred_attribute~.,data_train,size=20)
nnet_pred<-predict(nnet_model, data_test, type="class")

confusionMatrix(nnet_pred,data_test$pred_attribute)
#------------------------------------------------------------------------

library(adabag)
adaboost<-boosting(pred_attribute~.,data_train)
ada_pred<-predict(adaboost,data_test)
confusionMatrix(ada_pred$class,data_test$pred_attribute)
t1<-adaboost$trees[[1]]
library(tree)
plot(t1)
text(t1,pretty=1)
#-----------------------------------------------------------------------