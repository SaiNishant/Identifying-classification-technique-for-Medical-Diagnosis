diabetes_data<-read.csv("pima-indians-diabetes.csv",header = FALSE)

View(diabetes_data)
colnames(diabetes_data)<-c("Preg","Gluc","BP","SkinThick","Insulin","BMI","DPF","Age","Outcome")

is.na(diabetes_data)
summary(diabetes_data)

sapply(diabetes_data,class)

library(caret)
diabetes_data$Outcome<-as.factor(diabetes_data$Outcome)
split=0.80
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
model_nb <- naiveBayes(Outcome~., data=data_train)
predictions_naive <- predict(model_nb, data_test)
confusionMatrix(predictions_naive,data_test$Outcome)

# # -----------------------------------------------------------------------
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
logRegModel <- train(Outcome ~ ., data=data_train, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, data_test)
logRegPredictionprob <- predict(logRegModel, data_test, type='prob')[2]
confusionMatrix(logRegPrediction, data_test[,"Outcome"])
#ROC Curve
AUC=list()
Accuracy=list()
library(pROC)
AUC$logReg <- roc(as.numeric(data_test$Outcome),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']


# # -----------------------------------------------------------------------
library("e1071")
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
svm_model <- svm(Outcome ~ ., data=data_train)
pred_svm <- predict(svm_model,data_test)
confusionMatrix(pred_svm,data_test$Outcome)

#-------------------------------------------------------------------------------

library(MASS)
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
tr_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 10)
lda_train = train(Outcome ~ ., 
                  data=data_train, 
                  method="lda",
                  trControl=tr_ctrl)
pred_lda = predict(lda_train, data_test)
confusionMatrix(pred_lda,data_test$Outcome)
# ----------------------------------------------------------------------------------
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
forest_train = train(Outcome ~ ., 
                     data=data_train, 
                     method="rf",
                     trControl=tr_ctrl)
pred_forest<-predict(forest_train,data_test)
confusionMatrix(pred_forest,data_test$Outcome)
# -----------------------------------------------------------------

library(C50)
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
c50_model<-C5.0(Outcome~.,data_train)
pred_c50<-predict(c50_model,data_test)
confusionMatrix(pred_c50,data_test$Outcome)
#-----------------------------------------------------------------------
library(nnet)
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
nnet_model<-nnet(Outcome~.,data_train,size=10)
nnet_pred<-predict(nnet_model, data_test, type="class")
confusionMatrix(nnet_pred,data_test$Outcome)
#------------------------------------------------------------------------
trainIndex <- createDataPartition(diabetes_data$Outcome, p=split, list=FALSE)
data_train <- diabetes_data[ trainIndex,]
data_test <- diabetes_data[-trainIndex,]
library(adabag)
adaboost<-boosting(Outcome~.,data_train)
ada_pred<-predict(adaboost,data_test)
confusionMatrix(ada_pred$class,data_test$Outcome)
t1<-adaboost$trees[[1]]
library(tree)
plot(t1)
text(t1,pretty=1)
#-----------------------------------------------------------------------