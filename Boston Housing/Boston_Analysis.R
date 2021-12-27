# Load data set
library(MASS)

################################# Exploratory data analysis #####################################

install.packages("ggplot2")
install.packages("reshape2")
require(ggplot2)
require(reshape2)

# Scatter plot each feature against crim rate
Boston[,c(1:6,14)]
Boston[,7:14]

bosmelt1 <- melt(Boston[,c(1:6,14)], id="medv")
ggplot(bosmelt1, aes(x=value, y=medv))+
  facet_wrap(~variable, scales="free")+
  geom_point()

bosmelt2 <- melt(Boston[,7:14], id="medv")
ggplot(bosmelt2, aes(x=value, y=medv))+
  facet_wrap(~variable, scales="free")+
  geom_point()

# Correlation
install.packages("corrplot")
library(corrplot)
corrplot(cor(Boston), type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


####################################### Method ##########################################
dim(Boston)[1]#  506
round(dim(Boston)[1]*0.8)

set.seed(19)
flag <- sort(sample(dim(Boston)[1],round(dim(Boston)[1]*0.8), replace = FALSE))
btrain <- Boston[flag,]
btest <- Boston[-flag,]
## Extra the true response value for training and testing data
y_train <- btrain$medv
y_test  <- btest$medv

############################### Results #################################
# (a)

# (1) Random Forest
install.packages("randomForest")
library(randomForest)
rf1 <- randomForest(medv ~., data=btrain, 
                    importance=TRUE)

rf1

## Check Important variables
importance(rf1)
## There are two types of importance measure 
##  (1=mean decrease in accuracy, 
##   2= mean decrease in node impurity)
importance(rf1, type=2)
varImpPlot(rf1)

## Prediction on the testing data set
rf.pred = predict(rf1, btest, type='class')
table(rf.pred, y_test)

mean((y_test-rf.pred)^2) # MSE

# Tuning

# Which mtry to choose?
tuned.mtry <- tuneRF(btrain[,-14], btrain$medv, mtryStart = 4, ntreeTry = 500, stepFactor = 2, improve = 0)

rf2 <- randomForest(medv ~., data=btrain,
                    mtry=8, importance=TRUE)
rf2

importance(rf2)
importance(rf2, type=2)
varImpPlot(rf2)

## Prediction on the testing data set
rf2.pred = predict(rf2, btest, type='class')
table(rf2.pred, y_test)
mean((y_test-rf2.pred)^2) # MSE

# (2) Boosting
library(gbm)
#(2.1) gbm.bos1
gbm.bos1 <- gbm(medv ~ .,data=btrain,
                 distribution = 'gaussian',
                 n.trees = 500, 
                 shrinkage = 0.01, 
                 interaction.depth = 4,
                 cv.folds = 10)

print(gbm.bos1)

## Model Inspection 
## Find the estimated optimal number of iterations
perf_gbm1 = gbm.perf(gbm.bos1, method="cv") 
perf_gbm1

## summary model
## Which variances are important
summary(gbm.bos1)

# Plot lstat
plot(gbm.bos1, i="lstat")
plot(gbm.bos1, i="rm")

## Make Prediction
## use "predict" to find the training or testing error

## Training error
pred1gbm <- predict(gbm.bos1,newdata = btrain, n.trees=perf_gbm1, type="response")
mean((y_train-pred1gbm)^2)


## Testing Error
pred1test<-predict(gbm.bos1,newdata = btest[,-14], n.trees=perf_gbm1, type="response")
mean((y_test-pred1test)^2)

#(2.2) Tuning
ntree_opt_cv<-gbm.perf(gbm.bos1,method = "cv")
ntree_opt_cv
ntree_opt_oob<-gbm.perf(gbm.bos1,method = "OOB")
ntree_opt_oob

#(2.3) gbm.bos2
gbm.bos2 <- gbm(medv ~ .,data=btrain,
                distribution = 'gaussian',
                n.trees = 396, 
                shrinkage = 0.01, 
                interaction.depth = 4,
                cv.folds = 10)

print(gbm.bos2)

## Model Inspection 
## Find the estimated optimal number of iterations
perf_gbm2 = gbm.perf(gbm.bos2, method="cv") 
perf_gbm2

## summary model
## Which variances are important
summary(gbm.bos2)


## Training error
pred2gbm <- predict(gbm.bos2,newdata = btrain, n.trees=perf_gbm2, type="response")
mean((y_train-pred2gbm)^2)


## Testing Error
pred2test<-predict(gbm.bos2,newdata = btest[,-14], n.trees=perf_gbm2, type="response")
mean((y_test-pred2test)^2)

# (3) Baseline methods

#(3.1) Liner regression with stepwise variable selection 
model_lm <- lm( medv ~ ., data = btrain)
model_step  <- step(model_lm)
model_step

model_lm <-lm( medv ~ crim + zn + chas + nox + rm + dis + rad + 
                 tax + ptratio + black + lstat, data = btrain)
summary(model_lm)

# Testing error
pred_lm<-predict(model_lm,btest[,-14],type = 'response')
mean((y_test-pred_lm)^2)

# (3.2) Regression tree
library(rpart)
library(rpart.plot)

set.seed(19)
Boston.tree <- rpart(medv ~ ., data = btrain,  cp = 0.001)
rpart.plot(Boston.tree, type = 1, fallen.leaves = FALSE)

### (b) Optimal tree size
plotcp(Boston.tree)
print(Boston.tree$cptable)
opt <- which.min(Boston.tree$cptable[, "xerror"])
opt
cp_opt <- Boston.tree$cptable[opt, "CP"]
cp_opt

## (c) Pruning
rpart.pruned <- prune(Boston.tree,cp=cp_opt)
#### Plot
rpart.plot(rpart.pruned, box.palette="RdBu", shadow.col="gray")

## (d) Error
##(di) Testing error
pred2<-predict(rpart.pruned,btest[,-14])
mean((y_test-pred2)^2)