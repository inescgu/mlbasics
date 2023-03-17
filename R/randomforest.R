# random forest
# by ines, taken from ML final project

rm(list=ls())

LoadPackages=function(packages){
  for (i in 1:length(packages)){
    suppressPackageStartupMessages(library(packages[i], character.only=TRUE))
  }
}

LoadPackages(c("rstudioapi","caret","randomForest","e1071",
               "tidyverse","dplyr"))

setwd(dirname(getActiveDocumentContext()$path))
# Note: if you run any scripts individually, make sur

#help(choose.k) # run for hyperparameter help tuning
# load processed data from python
X_train<-read.csv("../processed/X_train.csv")
X_test<-read.csv("../processed/X_test.csv")

y<-read.csv("../processed/couse_vector.csv")
y$couse<-as.factor(y$couse)

trainids <- read.csv("../processed/trainids.csv")
names(trainids)[2] <- "SEQN"

testids <- read.csv("../processed/testids.csv")
names(testids)[2] <- "SEQN"

y_train<- subset(y, y$X %in% trainids$SEQN)
y_test <- subset(y, y$X %in% testids$SEQN)

names(y_train)[1] <-"SEQN"
names(y_test)[1] <-"SEQN"

y_train<-left_join(y_train, trainids, by="SEQN")
y_test<-left_join(y_test, testids, by="SEQN")

X_train<-left_join(X_train, y_train, by="X")
X_train<-subset(X_train, select=-c(SEQN,X))

X_test<-left_join(X_test, y_test, by="X")
X_test<-subset(X_test, select=-c(SEQN,X))

# Create a Random Forest model with default parameters
model <- randomForest(couse~ ., data = X_train, importance = TRUE)
model

# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(X_train))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(couse~., data=X_train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

# https://www.r-bloggers.com/2018/01/how-to-implement-random-forests-in-r/

# tune random forest
### best mtry ----
# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(X_train))
rf_random <- train(couse~., data=X_train, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# Grid search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(couse~., data=X_train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

# try parameters
best_mtry<- rf_gridsearch$bestTune$mtry
max(rf_gridsearch$results$Accuracy)
### best max node ----
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(couse~.,
                      data = X_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

### best trees ----
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(couse~.,
                       data = X_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 24,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

### final model ----
fit <- train(couse~.,
             X_train,
             method = "rf",
             metric = "Accuracy",
             tuneGrid = tuneGrid,
             trControl = trControl,
             importance = TRUE,
             nodesize = 14,
             ntree = 800,
             maxnodes = 24)

# test on test set
predictForest <- predict(fit, newdata = X_test)
table(X_test, predictForest)

# confusion matrix
confusionMatrix(predictForest, X_test$couse)

# variable importance plot
varImpPlot(fit)

write.csv(X_train, file="../processed/Xy_train_couse.csv")
write.csv(X_test, file="../processed/Xy_test_couse.csv")
